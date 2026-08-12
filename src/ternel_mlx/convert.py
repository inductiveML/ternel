"""Turn the frozen TQ1_G128 sidecar into a loadable mlx-lm checkpoint.

The sidecar is the artifact every gate in this repository is defined over: 498
tensors of canonical 28-byte blocks, verified losslessly re-encodable from the
GGUF's Q2_0. This converter does not re-derive it. It reads those exact bytes,
does the two things the sidecar cannot know about, and writes safetensors:

* **Reorders the value heads.** llama.cpp and mlx-lm disagree on the order of
  the 48 linear-attention value heads. Eight tensor kinds carry that axis. The
  permutation is a pure gather over whole rows or whole 128-weight groups, so no
  block is opened and no weight is re-encoded -- see :mod:`ternel_mlx.naming`.
* **Tiles for the Metal kernels.** ``[row][group][byte]`` becomes
  ``[tile][group][code_slot][row_in_tile]``, so a threadgroup's 256 threads read
  one code slot across 256 rows as one coalesced load. The scale pair becomes a
  ``uint16`` by reinterpretation. Both are permutations of the same bytes.

The 353 non-quantised tensors are copied verbatim out of the official mlx-lm
checkpoint, because :mod:`ternel_mlx.crosscheck` measured them against the GGUF
and found them identical -- so taking mlx-lm's own bytes costs nothing in
fidelity and removes every convention question about norm shifts and conv1d
layout at a stroke.

Nothing here imports MLX. The artifact is written by ``numpy`` through
:mod:`ternel_mlx.shard_writer`, so the file that MLX will later load was not
produced by MLX, and the verifier that reads it back shares no code with either.

Every tensor is hashed at three points -- as it leaves the sidecar, after the
reorder, and as it lands on disk -- and the first of those is *compared* against
the frozen sidecar manifest rather than merely recorded, so a silent change in
the source cannot pass through.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bonsai_tq1.format import (
    ALIGNMENT,
    BLOCK_BYTES,
    BLOCK_SIZE,
    FormatError,
    read_sidecar,
    write_json_atomic,
)
from bonsai_tq1.gguf_utils import load_tensor_infos

from . import LAYOUT_NAME, LAYOUT_VERSION
from .layout import CODES_SUFFIX, PackedTensorLayout, SCALES_SUFFIX
from .naming import (
    HeadAxis,
    HeadLayout,
    MappedTensor,
    build_name_map,
    check_against_baseline,
)
from .packed_model import ACTIVATION_DTYPE_KEY
from .packing import pack_blocks, validate_packed_codes
from .remote_safetensors import LocalSafetensors
from .shard_writer import (
    PlannedTensor,
    expected_file_bytes,
    plan_shards,
    shard_name,
    write_index,
    write_shard,
)

SCHEMA_VERSION = 1
MANIFEST_NAME = "ternel_manifest.json"
CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"

# The file ``mlx_lm.utils.load_model`` imports ``Model`` and ``ModelArgs`` from,
# named by ``config["model_file"]``. It is loaded by path under the module name
# ``custom_model``, with no package and without the model directory on
# ``sys.path``, so it can only import from installed packages -- which is why it
# is a shim onto Ternel rather than a copy of it.
MODEL_FILE_NAME = "ternel_packed_model.py"

MODEL_FILE_TEMPLATE = '''"""The classes ``mlx_lm.utils.load_model`` imports for this checkpoint.

``config.json`` names this file in ``model_file``. mlx-lm loads it by path and
takes ``Model`` and ``ModelArgs`` from it.

Both come from the installed Ternel package, which this checkpoint requires
rather than merely benefits from: its weights are {layout_name} blocks in a
tiled layout that only Ternel's Metal kernels can read, and there is no
dequantised copy to fall back to. Vendoring those kernels beside the weights
would create a second definition of the format, free to drift from the one the
artifact was verified against.

    pip install ternel
    mlx_lm.generate --model <this directory> --prompt "..."

Written by ``ternel_mlx.convert`` for layout {layout_name} v{layout_version}.
"""

from ternel_mlx.packed_model import Model, ModelArgs

__all__ = ["Model", "ModelArgs"]
'''

# Config keys that assert a quantisation mlx-lm would act on. ``load_model``
# calls ``nn.quantize`` when it sees ``quantization``, and promotes
# ``text_config.quantization_config`` to the top level. Ternel stores its own
# arrays under those module paths, so every one of these must be gone.
QUANTISATION_KEYS = ("quantization", "quantization_config")

# Config keys describing the vision tower. The GGUF holds no vision weights and
# Ternel refuses to source them elsewhere, so the artifact says so in its config
# rather than shipping a config that promises a tower it does not have.
VISION_KEYS = (
    "vision_config",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)

# Files copied verbatim from the official repository. The vision preprocessor
# configs are deliberately absent; the licence and notice are not optional.
COMPANION_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "LICENSE.txt",
    "NOTICE.txt",
)

TEXT_ONLY_ARCHITECTURE = "Qwen3_5ForCausalLM"


def sha256_bytes(array: np.ndarray) -> str:
    """Hash an array's bytes through the buffer protocol, without copying them.

    ``ascontiguousarray`` is a no-op on the memmap windows and freshly built
    arrays this sees, so the 278 MB embedding table is hashed in place rather
    than duplicated by ``tobytes``.
    """
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


@dataclass(frozen=True)
class SidecarTensor:
    """Where one tensor's canonical blocks live in the frozen sidecar."""

    name: str
    offset: int
    groups: int
    elements: int
    payload_sha256: str
    logical_weight_sha256: str
    scale_bits_sha256: str

    @property
    def nbytes(self) -> int:
        return self.groups * BLOCK_BYTES


def index_sidecar(sidecar_path: Path) -> dict[str, SidecarTensor]:
    """The frozen sidecar's manifest, keyed by GGUF tensor name."""
    header, manifest = read_sidecar(sidecar_path)
    if manifest["block_bytes"] != BLOCK_BYTES or manifest["block_size"] != BLOCK_SIZE:
        raise FormatError(
            f"{sidecar_path} declares {manifest['block_bytes']} bytes per "
            f"{manifest['block_size']} weights, not {BLOCK_BYTES} per {BLOCK_SIZE}"
        )
    if header.alignment != ALIGNMENT:
        raise FormatError(f"{sidecar_path} is aligned to {header.alignment}, not {ALIGNMENT}")
    indexed = {
        entry["name"]: SidecarTensor(
            name=entry["name"],
            offset=entry["packed_offset"],
            groups=entry["groups"],
            elements=entry["elements"],
            payload_sha256=entry["packed_payload_sha256"],
            logical_weight_sha256=entry["logical_weight_sha256"],
            scale_bits_sha256=entry["scale_bits_sha256"],
        )
        for entry in manifest["tensors"]
    }
    if len(indexed) != len(manifest["tensors"]):
        raise FormatError(f"{sidecar_path} names a tensor twice")
    return indexed


def read_canonical_blocks(
    mapping: np.memmap, entry: SidecarTensor, *, layout: PackedTensorLayout
) -> np.ndarray:
    """One tensor's canonical blocks as ``(rows, groups_per_row, 28)`` uint8."""
    if entry.groups != layout.groups:
        raise FormatError(
            f"{entry.name}: the sidecar holds {entry.groups} groups, the layout needs "
            f"{layout.groups}"
        )
    if entry.elements != layout.rows * layout.columns:
        raise FormatError(
            f"{entry.name}: the sidecar holds {entry.elements} weights, the layout needs "
            f"{layout.rows * layout.columns}"
        )
    end = entry.offset + entry.nbytes
    if end > mapping.size:
        raise FormatError(f"{entry.name} ends at byte {end}, past the {mapping.size}-byte sidecar")
    window = mapping[entry.offset : end]
    return window.reshape(layout.rows, layout.groups_per_row, BLOCK_BYTES)


def reorder_blocks(
    blocks: np.ndarray, tensor: MappedTensor, *, head_layout: HeadLayout
) -> np.ndarray:
    """Put a tensor's value heads in mlx-lm's order, moving whole blocks only."""
    reorder = tensor.reorder
    if reorder is None:
        return blocks
    if reorder.axis is HeadAxis.ROWS:
        return blocks[head_layout.source_order(blocks.shape[0], reorder)]
    return blocks[
        :, head_layout.group_source_order(blocks.shape[1], reorder, group_size=BLOCK_SIZE)
    ]


@dataclass(frozen=True)
class ConvertedTensor:
    """One quantised tensor's accounting, from frozen sidecar to artifact."""

    gguf_name: str
    mlx_name: str
    layout: PackedTensorLayout
    reorder: str
    sidecar_offset: int
    sidecar_sha256: str
    canonical_sha256: str
    codes_sha256: str
    scales_sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "gguf_name": self.gguf_name,
            "mlx_name": self.mlx_name,
            "reorder": self.reorder,
            "sidecar_offset": self.sidecar_offset,
            "sidecar_payload_sha256": self.sidecar_sha256,
            "canonical_sha256": self.canonical_sha256,
            "codes_sha256": self.codes_sha256,
            "scales_sha256": self.scales_sha256,
            **self.layout.to_json(),
        }


@dataclass(frozen=True)
class CopiedTensor:
    """One plain tensor taken verbatim from the official checkpoint."""

    gguf_name: str
    mlx_name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "gguf_name": self.gguf_name,
            "mlx_name": self.mlx_name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sha256": self.sha256,
        }


def describe_reorder(tensor: MappedTensor) -> str:
    reorder = tensor.reorder
    return "none" if reorder is None else f"{reorder.axis.value}/{reorder.unit.value}"


def plan_tensors(
    mapped: list[MappedTensor], baseline: LocalSafetensors
) -> list[PlannedTensor]:
    """Every array the artifact will hold, in the order the sidecar stores them."""
    planned: list[PlannedTensor] = []
    for tensor in mapped:
        if tensor.quantised:
            layout = PackedTensorLayout.for_tensor(tensor.rows, tensor.columns)
            planned.append(
                PlannedTensor(tensor.codes_name, np.dtype(np.uint8), layout.codes_shape)
            )
            planned.append(
                PlannedTensor(tensor.scales_name, np.dtype("<u2"), layout.scales_shape)
            )
        else:
            entry = baseline.entry(tensor.mlx_name)
            planned.append(PlannedTensor(tensor.mlx_name, entry.numpy_dtype, entry.shape))
    return planned


def artifact_config(baseline_config: dict[str, object], *, activation_dtype: str) -> dict[str, object]:
    """The baseline's config, stripped of every claim the artifact does not meet."""
    config = json.loads(json.dumps(baseline_config))
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise FormatError("the baseline config has no text_config object")
    for key in QUANTISATION_KEYS:
        config.pop(key, None)
        text_config.pop(key, None)
    for key in VISION_KEYS:
        config.pop(key, None)
    config["architectures"] = [TEXT_ONLY_ARCHITECTURE]
    config["language_model_only"] = True
    config["model_file"] = MODEL_FILE_NAME
    text_config[ACTIVATION_DTYPE_KEY] = activation_dtype
    for key in QUANTISATION_KEYS:
        if key in config or key in text_config:
            raise FormatError(f"{key} survived the strip; mlx-lm would re-quantise the artifact")
    return config


def copy_companions(baseline_dir: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in COMPANION_FILES:
        source = baseline_dir / name
        if not source.is_file():
            raise FormatError(f"the official checkpoint is missing {name}")
        shutil.copyfile(source, out_dir / name)
        copied.append(name)
    return copied


def write_model_file(out_dir: Path) -> dict[str, str]:
    """Write the ``model_file`` mlx-lm imports, and hash what was written."""
    text = MODEL_FILE_TEMPLATE.format(layout_name=LAYOUT_NAME, layout_version=LAYOUT_VERSION)
    (out_dir / MODEL_FILE_NAME).write_text(text, encoding="utf-8")
    return {
        "name": MODEL_FILE_NAME,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "requires": f"ternel_mlx>={LAYOUT_VERSION}",
    }


def convert(
    gguf_path: Path,
    sidecar_path: Path,
    baseline_dir: Path,
    out_dir: Path,
    *,
    max_shard_bytes: int,
    activation_dtype: str,
) -> dict[str, object]:
    started = time.time()
    baseline_config = json.loads((baseline_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    text_config = baseline_config["text_config"]
    head_layout = HeadLayout.from_text_config(text_config)

    _, infos = load_tensor_infos(gguf_path)
    mapped = build_name_map(infos)
    baseline = LocalSafetensors(baseline_dir / "model.safetensors")
    check_against_baseline(mapped, set(baseline.names()))
    sidecar_entries = index_sidecar(sidecar_path)

    quantised = [tensor for tensor in mapped if tensor.quantised]
    missing = sorted(t.gguf.name for t in quantised if t.gguf.name not in sidecar_entries)
    if missing:
        raise FormatError(f"the sidecar is missing {len(missing)} tensors, e.g. {missing[:3]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    planned = plan_tensors(mapped, baseline)
    shards = plan_shards(planned, max_shard_bytes=max_shard_bytes)
    metadata = {
        "format": "pt",
        "layout_name": LAYOUT_NAME,
        "layout_version": str(LAYOUT_VERSION),
        "ternel_manifest": MANIFEST_NAME,
    }

    mapping = np.memmap(sidecar_path, dtype=np.uint8, mode="r")
    by_mlx_name = {tensor.mlx_name: tensor for tensor in mapped}
    converted: dict[str, ConvertedTensor] = {}
    copied: dict[str, CopiedTensor] = {}
    produced: dict[str, np.ndarray] = {}

    def build_quantised(tensor: MappedTensor) -> None:
        layout = PackedTensorLayout.for_tensor(tensor.rows, tensor.columns)
        entry = sidecar_entries[tensor.gguf.name]
        blocks = read_canonical_blocks(mapping, entry, layout=layout)
        source_digest = sha256_bytes(blocks)
        if source_digest != entry.payload_sha256:
            raise FormatError(
                f"{tensor.gguf.name}: the sidecar's bytes hash to {source_digest}, but its "
                f"manifest records {entry.payload_sha256}"
            )
        ordered = np.ascontiguousarray(reorder_blocks(blocks, tensor, head_layout=head_layout))
        # A tensor with no value-head axis is handed to the packer as the very
        # bytes that were hashed, so re-hashing it would measure nothing. Stating
        # the equality is the stronger claim, and the reordered ones still pay.
        canonical_digest = source_digest if tensor.reorder is None else sha256_bytes(ordered)
        codes, scales = pack_blocks(ordered, layout=layout)
        validate_packed_codes(codes)
        produced[tensor.codes_name] = codes
        produced[tensor.scales_name] = scales
        converted[tensor.gguf.name] = ConvertedTensor(
            gguf_name=tensor.gguf.name,
            mlx_name=tensor.mlx_name,
            layout=layout,
            reorder=describe_reorder(tensor),
            sidecar_offset=entry.offset,
            sidecar_sha256=source_digest,
            canonical_sha256=canonical_digest,
            codes_sha256=sha256_bytes(codes),
            scales_sha256=sha256_bytes(scales),
        )

    def produce(planned_tensor: PlannedTensor) -> np.ndarray:
        cached = produced.pop(planned_tensor.name, None)
        if cached is not None:
            return cached
        module, _, suffix = planned_tensor.name.rpartition(".")
        if suffix in (CODES_SUFFIX, SCALES_SUFFIX) and module in by_mlx_name:
            build_quantised(by_mlx_name[module])
            return produced.pop(planned_tensor.name)
        tensor = by_mlx_name[planned_tensor.name]
        array = np.ascontiguousarray(baseline.read(tensor.mlx_name))
        copied[tensor.gguf.name] = CopiedTensor(
            gguf_name=tensor.gguf.name,
            mlx_name=tensor.mlx_name,
            dtype=baseline.entry(tensor.mlx_name).dtype,
            shape=tuple(int(dim) for dim in array.shape),
            sha256=sha256_bytes(array),
        )
        return array

    shard_files: list[dict[str, object]] = []
    total_bytes = 0
    for index, shard in enumerate(shards):
        name = shard_name(index, len(shards))
        written = write_shard(out_dir / name, shard, produce, metadata=metadata)
        actual = (out_dir / name).stat().st_size
        expected = expected_file_bytes(shard, metadata)
        if actual != expected:
            raise FormatError(f"{name} is {actual} bytes on disk, expected {expected}")
        total_bytes += actual
        shard_files.append(
            {"name": name, "tensors": len(shard), "payload_bytes": written, "file_bytes": actual}
        )
    if produced:
        raise FormatError(f"{len(produced)} built arrays were never written: {sorted(produced)[:3]}")
    if len(converted) != len(quantised):
        raise FormatError(f"converted {len(converted)} quantised tensors, expected {len(quantised)}")
    if len(copied) != len(mapped) - len(quantised):
        raise FormatError(f"copied {len(copied)} plain tensors, expected {len(mapped) - len(quantised)}")

    write_index(out_dir / INDEX_NAME, shards, total_bytes=total_bytes)
    config = artifact_config(baseline_config, activation_dtype=activation_dtype)
    write_json_atomic(out_dir / CONFIG_NAME, config)
    companions = copy_companions(baseline_dir, out_dir)
    model_file = write_model_file(out_dir)

    payload_bytes = sum(item.layout.payload_bytes for item in converted.values())
    canonical_bytes = sum(item.layout.canonical_bytes for item in converted.values())
    if payload_bytes != canonical_bytes:
        raise FormatError(
            f"packed payload is {payload_bytes} bytes but the canonical format is "
            f"{canonical_bytes}; the layout is padding somewhere"
        )
    weights = sum(item.layout.rows * item.layout.columns for item in converted.values())

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "layout_name": LAYOUT_NAME,
        "layout_version": LAYOUT_VERSION,
        "source": {
            "gguf": str(gguf_path),
            "sidecar": str(sidecar_path),
            "baseline": str(baseline_dir),
        },
        "quantised": {
            "tensors": len(converted),
            "weights": weights,
            "groups": sum(item.layout.groups for item in converted.values()),
            "payload_bytes": payload_bytes,
            "canonical_bytes": canonical_bytes,
            "bits_per_weight": payload_bytes * 8 / weights,
            "reordered_tensors": sum(1 for item in converted.values() if item.reorder != "none"),
            "tensors_detail": [converted[name].to_json() for name in sorted(converted)],
        },
        "plain": {
            "tensors": len(copied),
            "elements": sum(int(np.prod(item.shape)) for item in copied.values()),
            "tensors_detail": [copied[name].to_json() for name in sorted(copied)],
        },
        "shards": shard_files,
        "files": {
            "config": CONFIG_NAME,
            "index": INDEX_NAME,
            "companions": companions,
            "model_file": model_file,
        },
        "artifact_bytes": total_bytes,
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(out_dir / MANIFEST_NAME, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-shard-bytes", type=int, required=True)
    parser.add_argument("--activation-dtype", type=str, required=True)
    args = parser.parse_args(argv)

    manifest = convert(
        args.gguf,
        args.sidecar,
        args.baseline,
        args.out,
        max_shard_bytes=args.max_shard_bytes,
        activation_dtype=args.activation_dtype,
    )
    quantised = manifest["quantised"]
    print(
        f"{quantised['tensors']} quantised tensors, {quantised['weights']} weights, "
        f"{quantised['payload_bytes']} payload bytes ({quantised['bits_per_weight']:.4f} bits/weight)"
    )
    print(
        f"{manifest['plain']['tensors']} plain tensors copied verbatim, "
        f"{quantised['reordered_tensors']} tensors reordered"
    )
    print(f"{len(manifest['shards'])} shards, {manifest['artifact_bytes']} bytes in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
