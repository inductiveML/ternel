"""Prove the packed artifact holds exactly the model it claims to.

The converter is a chain -- GGUF Q2_0, frozen TQ1_G128 sidecar, value-head
permutation, code-position-major tiling, safetensors shards -- and each link can
fail in a way the next one cannot see. So the artifact is checked against three
different things, chosen so that no single mistake can satisfy all three:

**Storage.** Read the shards back, undo the tiling and undo the permutation, and
require the bytes to equal the sidecar's *elementwise*, not by digest alone.
Then recompute the sidecar's own two frozen digests -- the logical ternary
weights and the raw FP16 scale bits, hashed exactly as ``bonsai_tq1.pack_model``
defines them -- and require those to match too.

**Source.** Decode the same chunk and compare it against the GGUF's Q2_0 blocks
decoded independently, so the result does not rest on the sidecar being what its
manifest says. If the sidecar had drifted, this leg would say so.

**Ground truth.** Compare the artifact against ``prism-ml/Ternary-Bonsai-27B-mlx-2bit``
*without applying any permutation at all*: both are already in mlx-lm's order, so
this leg is the one the value-head reordering cannot hide inside. It has to be
here. The first two legs apply the permutation and then invert it with the same
code, so a wrong permutation cancels out and passes them both -- which is the
textbook shape of a verification that proves nothing.

One divergence is expected and is not a defect of this artifact: row 178519 of
the embedding table, where the two upstream publications encoded a zero-absmax
row differently. :mod:`ternel_mlx.crosscheck` characterises it in full. The
weight-space gap it produces is measured here and compared against a bound the
caller states.

Nothing in here imports MLX. The artifact is read with ``numpy.memmap`` through
Ternel's own safetensors header parser, and the tiling is undone by
:func:`ternel_mlx.packing.unpack_blocks`, whose inverse relationship to the
packer is itself a unit-tested property.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from bonsai_tq1.format import (
    BLOCK_BYTES,
    BLOCK_SIZE,
    MAX_FULL_CODE_BYTE,
    MAX_TAIL_CODE_BYTE,
    FormatError,
    decode_tq1_blocks,
    write_json_atomic,
)
from bonsai_tq1.gguf_utils import load_tensor_infos
from bonsai_tq1.lut23_reorder import FULL_CODE_SLOTS

from .convert import (
    CONFIG_NAME,
    INDEX_NAME,
    MANIFEST_NAME,
    QUANTISATION_KEYS,
    VISION_KEYS,
    SidecarTensor,
    index_sidecar,
    sha256_bytes,
)
from .crosscheck import (
    NEGATIVE_ZERO_BITS,
    RowChunk,
    baseline_quantised_chunk,
    dequantised_difference,
    describe_reorder,
    gguf_quantised_chunk,
)
from .layout import PackedTensorLayout
from .naming import (
    HeadAxis,
    HeadLayout,
    MappedTensor,
    build_name_map,
    check_against_baseline,
)
from .packed_model import ACTIVATION_DTYPE_KEY
from .packing import unpack_blocks, validate_packed_codes
from .remote_safetensors import LocalSafetensors

SCHEMA_VERSION = 1

# A row permutation has no self-contained sub-range: mlx-lm row ``i`` is drawn
# from GGUF row ``order[i]``, which sits anywhere in the permuted block. Tensors
# carrying one are therefore read whole, and this is the ceiling on how large
# that is allowed to get before the verifier refuses rather than exhausting
# memory. The largest such tensor in Ternary Bonsai 27B is ``attn_qkv`` at
# 10240x5120 = 52,428,800 weights, so the ceiling is not close to binding.
MAX_UNCHUNKED_WEIGHTS = 1 << 30


@dataclass(frozen=True)
class TileChunk:
    """A range of whole tiles, and the rows they cover."""

    begin_tile: int
    end_tile: int
    tile: int

    @property
    def tiles(self) -> int:
        return self.end_tile - self.begin_tile

    @property
    def row_range(self) -> RowChunk:
        """The rows these tiles hold, in the artifact's own order."""
        return RowChunk(self.begin_tile * self.tile, self.end_tile * self.tile)


def tile_chunks(layout: PackedTensorLayout, *, weights_per_chunk: int, whole: bool) -> Iterator[TileChunk]:
    """Whole-tile chunks of at most ``weights_per_chunk`` weights.

    ``whole`` yields the tensor in one piece, for the tensors whose rows are
    permuted and so cannot be verified a slice at a time.
    """
    if weights_per_chunk <= 0:
        raise FormatError(f"chunk budget must be positive, got {weights_per_chunk}")
    if whole:
        if layout.rows * layout.columns > MAX_UNCHUNKED_WEIGHTS:
            raise FormatError(
                f"a row-permuted tensor of {layout.rows}x{layout.columns} weights exceeds the "
                f"{MAX_UNCHUNKED_WEIGHTS}-weight ceiling for reading a tensor whole"
            )
        yield TileChunk(0, layout.tiles, layout.tile)
        return
    step = max(1, weights_per_chunk // (layout.tile * layout.columns))
    for begin in range(0, layout.tiles, step):
        yield TileChunk(begin, min(begin + step, layout.tiles), layout.tile)


class ArtifactReader:
    """Every tensor of a sharded artifact, resolved through its index file.

    The index is *checked*, not trusted: a tensor is read from the shard the
    index names, and the shard's own header has to contain it.
    """

    def __init__(self, artifact_dir: Path) -> None:
        index = json.loads((artifact_dir / INDEX_NAME).read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise FormatError(f"{INDEX_NAME} carries no weight map")
        self.total_size = int(index["metadata"]["total_size"])
        self.shards: dict[str, LocalSafetensors] = {}
        self.owner: dict[str, str] = {}
        for name, shard in sorted(weight_map.items()):
            if shard not in self.shards:
                path = artifact_dir / shard
                if not path.is_file():
                    raise FormatError(f"{INDEX_NAME} names {shard}, which is not in {artifact_dir}")
                self.shards[shard] = LocalSafetensors(path)
            if name not in self.shards[shard]:
                raise FormatError(f"{INDEX_NAME} puts {name} in {shard}, whose header lacks it")
            self.owner[name] = shard

    def __contains__(self, name: str) -> bool:
        return name in self.owner

    @property
    def names(self) -> set[str]:
        return set(self.owner)

    def reader(self, name: str) -> LocalSafetensors:
        shard = self.owner.get(name)
        if shard is None:
            raise FormatError(f"{name} is not in the artifact index")
        return self.shards[shard]

    def read(self, name: str) -> np.ndarray:
        return self.reader(name).read(name)

    def read_rows(self, name: str, begin: int, end: int) -> np.ndarray:
        return self.reader(name).read_rows(name, begin, end)

    def unclaimed(self) -> set[str]:
        """Tensors present in a shard that the index does not account for."""
        held = {name for shard in self.shards.values() for name in shard.names()}
        return held - self.names


def inverse_permutation(order: np.ndarray) -> np.ndarray:
    """``order`` maps a target slot to its source slot; this maps back.

    ``target = source[order]``, so ``source = target[inverse_permutation(order)]``.
    """
    inverse = np.argsort(order, kind="stable")
    if not np.array_equal(order[inverse], np.arange(order.size)):
        raise FormatError("the value-head order is not a permutation of its own axis")
    return inverse


def sidecar_blocks(
    mapping: np.memmap, entry: SidecarTensor, chunk: RowChunk, *, groups_per_row: int
) -> np.ndarray:
    """The frozen sidecar's canonical blocks for one row range, in GGUF order."""
    stride = groups_per_row * BLOCK_BYTES
    begin = entry.offset + chunk.begin * stride
    end = entry.offset + chunk.end * stride
    if end > mapping.size:
        raise FormatError(f"{entry.name} rows {chunk.begin}:{chunk.end} run past the sidecar")
    return mapping[begin:end].reshape(chunk.rows, groups_per_row, BLOCK_BYTES)


@dataclass(frozen=True)
class QuantisedVerification:
    """What the three legs found for one packed tensor."""

    gguf_name: str
    mlx_name: str
    rows: int
    columns: int
    groups: int
    tile: int
    reorder: str
    sidecar_byte_mismatches: int
    gguf_code_mismatches: int
    gguf_scale_mismatches: int
    baseline_code_mismatches: int
    baseline_scale_mismatches: int
    baseline_bias_mismatches: int
    baseline_max_weight_difference: float
    logical_sha256: str
    logical_sha256_frozen: str
    scale_sha256: str
    scale_sha256_frozen: str
    max_trit_code: int
    max_full_code_byte: int
    max_tail_code_byte: int
    bounds_violation: str | None

    @property
    def bounds_safe(self) -> bool:
        """Whether the kernels' unclamped LUT indexing is safe on this tensor.

        Byte 255 divides to L3 row 28, one past a 27-entry table, so the fast
        path's lack of a clamp is only sound while every byte is legal. This is
        the load-time scan that makes that true, and it is measured here rather
        than asserted: the two maxima are in the report.
        """
        return (
            self.bounds_violation is None
            and self.max_full_code_byte <= MAX_FULL_CODE_BYTE
            and self.max_tail_code_byte <= MAX_TAIL_CODE_BYTE
        )

    @property
    def storage_exact(self) -> bool:
        return (
            self.sidecar_byte_mismatches == 0
            and self.logical_sha256 == self.logical_sha256_frozen
            and self.scale_sha256 == self.scale_sha256_frozen
        )

    @property
    def source_exact(self) -> bool:
        return self.gguf_code_mismatches == 0 and self.gguf_scale_mismatches == 0

    @property
    def baseline_identical(self) -> bool:
        return (
            self.baseline_code_mismatches == 0
            and self.baseline_scale_mismatches == 0
            and self.baseline_bias_mismatches == 0
        )

    def baseline_within(self, bound: float) -> bool:
        return self.baseline_max_weight_difference <= bound

    def to_json(self) -> dict[str, object]:
        return {
            "gguf_name": self.gguf_name,
            "mlx_name": self.mlx_name,
            "rows": self.rows,
            "columns": self.columns,
            "groups": self.groups,
            "tile": self.tile,
            "reorder": self.reorder,
            "sidecar_byte_mismatches": self.sidecar_byte_mismatches,
            "gguf_code_mismatches": self.gguf_code_mismatches,
            "gguf_scale_mismatches": self.gguf_scale_mismatches,
            "baseline_code_mismatches": self.baseline_code_mismatches,
            "baseline_scale_mismatches": self.baseline_scale_mismatches,
            "baseline_bias_mismatches": self.baseline_bias_mismatches,
            "baseline_max_weight_difference": self.baseline_max_weight_difference,
            "logical_weight_sha256": self.logical_sha256,
            "logical_weight_sha256_frozen": self.logical_sha256_frozen,
            "scale_bits_sha256": self.scale_sha256,
            "scale_bits_sha256_frozen": self.scale_sha256_frozen,
            "max_trit_code": self.max_trit_code,
            "max_full_code_byte": self.max_full_code_byte,
            "max_tail_code_byte": self.max_tail_code_byte,
            "bounds_violation": self.bounds_violation,
            "bounds_safe": self.bounds_safe,
            "storage_exact": self.storage_exact,
            "source_exact": self.source_exact,
            "baseline_identical": self.baseline_identical,
        }


def verify_quantised(
    tensor: MappedTensor,
    *,
    artifact: ArtifactReader,
    mapping: np.memmap,
    entry: SidecarTensor,
    gguf_path: Path,
    baseline: LocalSafetensors,
    head_layout: HeadLayout,
    weights_per_chunk: int,
) -> QuantisedVerification:
    layout = PackedTensorLayout.for_tensor(tensor.rows, tensor.columns)
    reorder = tensor.reorder
    row_inverse = (
        inverse_permutation(head_layout.source_order(layout.rows, reorder))
        if reorder is not None and reorder.axis is HeadAxis.ROWS
        else None
    )
    group_inverse = (
        inverse_permutation(
            head_layout.group_source_order(layout.groups_per_row, reorder, group_size=BLOCK_SIZE)
        )
        if reorder is not None and reorder.axis is HeadAxis.COLUMNS
        else None
    )

    logical_hash = hashlib.sha256()
    scale_hash = hashlib.sha256()
    sidecar_bad = 0
    gguf_code_bad = 0
    gguf_scale_bad = 0
    base_code_bad = 0
    base_scale_bad = 0
    base_bias_bad = 0
    max_gap = 0.0
    max_trit = 0
    max_full = 0
    max_tail = 0
    violation: str | None = None

    for chunk in tile_chunks(
        layout, weights_per_chunk=weights_per_chunk, whole=row_inverse is not None
    ):
        rows = chunk.row_range
        window = PackedTensorLayout(rows=rows.rows, columns=layout.columns, tile=layout.tile)
        codes = np.ascontiguousarray(
            artifact.read_rows(tensor.codes_name, chunk.begin_tile, chunk.end_tile)
        )
        scales = np.ascontiguousarray(
            artifact.read_rows(tensor.scales_name, chunk.begin_tile, chunk.end_tile)
        )
        # The shipped load-time gate, run so that a violation is *reported*
        # rather than thrown: one unsafe tensor should not cost the report on
        # the other 497. The maxima beside it are the same fact as a number.
        try:
            validate_packed_codes(codes)
        except FormatError as failure:
            violation = violation or str(failure)
        max_full = max(max_full, int(codes[:, :, :FULL_CODE_SLOTS, :].max()))
        max_tail = max(max_tail, int(codes[:, :, FULL_CODE_SLOTS, :].max()))

        # Leg 3 first, in mlx-lm's own order: no permutation is applied here, so
        # this is the one comparison a wrong permutation cannot survive.
        packed = unpack_blocks(codes, scales, layout=window)
        artifact_scale_bits, artifact_codes = decode_tq1_blocks(
            packed.reshape(-1, BLOCK_BYTES), validate=True
        )
        artifact_scale_bits = artifact_scale_bits.view("<u2").reshape(
            rows.rows, layout.groups_per_row
        )
        artifact_codes = artifact_codes.reshape(rows.rows, layout.columns)
        max_trit = max(max_trit, int(artifact_codes.max()))

        base_scales, base_codes, base_biases = baseline_quantised_chunk(
            baseline, tensor.mlx_name, layout.columns, rows
        )
        code_bad = artifact_codes != base_codes
        scale_bad = artifact_scale_bits != base_scales
        bias_bad = base_biases != (base_scales ^ NEGATIVE_ZERO_BITS)
        base_code_bad += int(np.count_nonzero(code_bad))
        base_scale_bad += int(np.count_nonzero(scale_bad))
        base_bias_bad += int(np.count_nonzero(bias_bad))
        rows_bad = np.flatnonzero(code_bad.any(axis=1) | scale_bad.any(axis=1) | bias_bad.any(axis=1))
        if rows_bad.size:
            max_gap = max(
                max_gap,
                dequantised_difference(
                    artifact_codes[rows_bad],
                    artifact_scale_bits[rows_bad],
                    base_codes[rows_bad],
                    base_scales[rows_bad],
                    base_biases[rows_bad],
                ),
            )

        # Legs 1 and 2, back in the GGUF's order.
        if group_inverse is not None:
            packed = packed[:, group_inverse]
        if row_inverse is not None:
            packed = packed[row_inverse]
        packed = np.ascontiguousarray(packed)
        sidecar_bad += int(
            np.count_nonzero(
                packed != sidecar_blocks(
                    mapping, entry, rows, groups_per_row=layout.groups_per_row
                )
            )
        )

        source_scale_bits, source_codes = decode_tq1_blocks(
            packed.reshape(-1, BLOCK_BYTES), validate=True
        )
        logical_hash.update(np.ascontiguousarray(source_codes))
        scale_hash.update(np.ascontiguousarray(source_scale_bits))

        gguf_scales, gguf_codes = gguf_quantised_chunk(gguf_path, tensor, rows)
        gguf_code_bad += int(
            np.count_nonzero(source_codes.reshape(rows.rows, layout.columns) != gguf_codes)
        )
        gguf_scale_bad += int(
            np.count_nonzero(
                source_scale_bits.view("<u2").reshape(rows.rows, layout.groups_per_row)
                != gguf_scales
            )
        )

    return QuantisedVerification(
        gguf_name=tensor.gguf.name,
        mlx_name=tensor.mlx_name,
        rows=layout.rows,
        columns=layout.columns,
        groups=layout.groups,
        tile=layout.tile,
        reorder=describe_reorder(reorder),
        sidecar_byte_mismatches=sidecar_bad,
        gguf_code_mismatches=gguf_code_bad,
        gguf_scale_mismatches=gguf_scale_bad,
        baseline_code_mismatches=base_code_bad,
        baseline_scale_mismatches=base_scale_bad,
        baseline_bias_mismatches=base_bias_bad,
        baseline_max_weight_difference=max_gap,
        logical_sha256=logical_hash.hexdigest(),
        logical_sha256_frozen=entry.logical_weight_sha256,
        scale_sha256=scale_hash.hexdigest(),
        scale_sha256_frozen=entry.scale_bits_sha256,
        max_trit_code=max_trit,
        max_full_code_byte=max_full,
        max_tail_code_byte=max_tail,
        bounds_violation=violation,
    )


@dataclass(frozen=True)
class PlainVerification:
    """Whether a copied tensor is still byte for byte what it was copied from."""

    gguf_name: str
    mlx_name: str
    dtype: str
    shape: tuple[int, ...]
    elements: int
    mismatches: int
    sha256: str
    baseline_sha256: str

    @property
    def identical(self) -> bool:
        return self.mismatches == 0 and self.sha256 == self.baseline_sha256

    def to_json(self) -> dict[str, object]:
        return {
            "gguf_name": self.gguf_name,
            "mlx_name": self.mlx_name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "elements": self.elements,
            "mismatches": self.mismatches,
            "sha256": self.sha256,
            "baseline_sha256": self.baseline_sha256,
            "identical": self.identical,
        }


def verify_plain(
    tensor: MappedTensor, *, artifact: ArtifactReader, baseline: LocalSafetensors
) -> PlainVerification:
    """A copied tensor is verified against what it was copied from.

    The copy applies no transform, so there is no arithmetic here to be wrong --
    only the possibility that the wrong bytes were written, or written somewhere
    the index does not point. That the baseline's own values are the GGUF's is
    what :mod:`ternel_mlx.crosscheck` established over all 353 of them.
    """
    got = np.ascontiguousarray(artifact.read(tensor.mlx_name))
    want = np.ascontiguousarray(baseline.read(tensor.mlx_name))
    if got.dtype != want.dtype or got.shape != want.shape:
        raise FormatError(
            f"{tensor.mlx_name} is {got.dtype}{got.shape} in the artifact but "
            f"{want.dtype}{want.shape} in the official checkpoint"
        )
    return PlainVerification(
        gguf_name=tensor.gguf.name,
        mlx_name=tensor.mlx_name,
        dtype=str(got.dtype),
        shape=tuple(int(dim) for dim in got.shape),
        elements=int(got.size),
        mismatches=int(np.count_nonzero(got.view("<u2") != want.view("<u2")))
        if got.dtype.itemsize == 2
        else int(np.count_nonzero(got != want)),
        sha256=sha256_bytes(got),
        baseline_sha256=sha256_bytes(want),
    )


def verify_structure(
    artifact_dir: Path, artifact: ArtifactReader, mapped: list[MappedTensor]
) -> dict[str, object]:
    """Everything about the artifact that is not a weight.

    A checkpoint can hold perfect tensors and still be unloadable, or loadable
    and silently re-quantised. Both are failures, so both are checked here.
    """
    expected = {name for tensor in mapped for name in tensor.parameter_names}
    problems: list[str] = []

    missing = sorted(expected - artifact.names)
    extra = sorted(artifact.names - expected)
    unclaimed = sorted(artifact.unclaimed())
    if missing:
        problems.append(f"{len(missing)} tensors are missing, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} tensors are not in the model, e.g. {extra[:3]}")
    if unclaimed:
        problems.append(f"{len(unclaimed)} shard tensors are absent from the index, e.g. {unclaimed[:3]}")

    config = json.loads((artifact_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise FormatError(f"{CONFIG_NAME} has no text_config object")
    for key in QUANTISATION_KEYS:
        if key in config or key in text_config:
            problems.append(f"config still declares {key}; mlx-lm would re-quantise the artifact")
    for key in VISION_KEYS:
        if key in config:
            problems.append(f"config still declares {key}, but the artifact ships no vision tower")
    if not isinstance(text_config.get(ACTIVATION_DTYPE_KEY), str):
        problems.append(f"config carries no string {ACTIVATION_DTYPE_KEY}")

    model_file = config.get("model_file")
    if not isinstance(model_file, str) or not (artifact_dir / model_file).is_file():
        problems.append(f"config names model_file {model_file!r}, which is not in the artifact")

    shard_bytes = {
        name: reader.path.stat().st_size for name, reader in sorted(artifact.shards.items())
    }
    manifest = json.loads((artifact_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    recorded = {item["name"]: int(item["file_bytes"]) for item in manifest["shards"]}
    if recorded != shard_bytes:
        problems.append(f"shard sizes on disk {shard_bytes} do not match the manifest {recorded}")
    if artifact.total_size != sum(shard_bytes.values()):
        problems.append(
            f"{INDEX_NAME} totals {artifact.total_size} bytes, the shards weigh "
            f"{sum(shard_bytes.values())}"
        )

    return {
        "expected_tensors": len(expected),
        "artifact_tensors": len(artifact.names),
        "missing": missing,
        "unexpected": extra,
        "unindexed": unclaimed,
        "shards": shard_bytes,
        "config_keys": sorted(config),
        "activation_dtype": text_config.get(ACTIVATION_DTYPE_KEY),
        "model_file": model_file,
        "problems": problems,
        "complete": not problems,
    }


def verify(
    artifact_dir: Path,
    sidecar_path: Path,
    gguf_path: Path,
    baseline_dir: Path,
    *,
    weights_per_chunk: int,
    max_weight_difference: float,
) -> dict[str, object]:
    baseline_config = json.loads((baseline_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    head_layout = HeadLayout.from_text_config(baseline_config["text_config"])
    _, infos = load_tensor_infos(gguf_path)
    mapped = build_name_map(infos)
    baseline = LocalSafetensors(baseline_dir / "model.safetensors")
    check_against_baseline(mapped, set(baseline.names()))

    artifact = ArtifactReader(artifact_dir)
    structure = verify_structure(artifact_dir, artifact, mapped)
    entries = index_sidecar(sidecar_path)
    mapping = np.memmap(sidecar_path, dtype=np.uint8, mode="r")

    quantised: list[QuantisedVerification] = []
    plain: list[PlainVerification] = []
    for tensor in mapped:
        if tensor.quantised:
            quantised.append(
                verify_quantised(
                    tensor,
                    artifact=artifact,
                    mapping=mapping,
                    entry=entries[tensor.gguf.name],
                    gguf_path=gguf_path,
                    baseline=baseline,
                    head_layout=head_layout,
                    weights_per_chunk=weights_per_chunk,
                )
            )
        else:
            plain.append(verify_plain(tensor, artifact=artifact, baseline=baseline))

    storage_bad = [item for item in quantised if not item.storage_exact]
    source_bad = [item for item in quantised if not item.source_exact]
    baseline_bad = [item for item in quantised if not item.baseline_identical]
    over_bound = [item for item in quantised if not item.baseline_within(max_weight_difference)]
    unsafe = [item for item in quantised if not item.bounds_safe]
    plain_bad = [item for item in plain if not item.identical]

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "artifact": str(artifact_dir),
            "sidecar": str(sidecar_path),
            "gguf": str(gguf_path),
            "baseline": str(baseline_dir),
        },
        "structure": structure,
        "quantised": {
            "tensors": len(quantised),
            "weights": sum(item.rows * item.columns for item in quantised),
            "groups": sum(item.groups for item in quantised),
            "reordered_tensors": sum(1 for item in quantised if item.reorder != "none"),
            "max_trit_code": max((item.max_trit_code for item in quantised), default=0),
            "max_full_code_byte": max((item.max_full_code_byte for item in quantised), default=0),
            "max_full_code_byte_allowed": MAX_FULL_CODE_BYTE,
            "max_tail_code_byte": max((item.max_tail_code_byte for item in quantised), default=0),
            "max_tail_code_byte_allowed": MAX_TAIL_CODE_BYTE,
            "sidecar_byte_mismatches": sum(item.sidecar_byte_mismatches for item in quantised),
            "gguf_code_mismatches": sum(item.gguf_code_mismatches for item in quantised),
            "gguf_scale_mismatches": sum(item.gguf_scale_mismatches for item in quantised),
            "baseline_code_mismatches": sum(item.baseline_code_mismatches for item in quantised),
            "baseline_scale_mismatches": sum(item.baseline_scale_mismatches for item in quantised),
            "baseline_bias_mismatches": sum(item.baseline_bias_mismatches for item in quantised),
            "baseline_max_weight_difference": max(
                (item.baseline_max_weight_difference for item in quantised), default=0.0
            ),
            "baseline_max_weight_difference_allowed": max_weight_difference,
            "frozen_digest_mismatches": sum(
                1
                for item in quantised
                if item.logical_sha256 != item.logical_sha256_frozen
                or item.scale_sha256 != item.scale_sha256_frozen
            ),
            "storage_failures": [item.to_json() for item in storage_bad],
            "source_failures": [item.to_json() for item in source_bad],
            "baseline_differences": [item.to_json() for item in baseline_bad],
            "over_bound": [item.to_json() for item in over_bound],
            "bounds_unsafe": [item.to_json() for item in unsafe],
            "verifications": [item.to_json() for item in quantised],
        },
        "plain": {
            "tensors": len(plain),
            "elements": sum(item.elements for item in plain),
            "mismatches": sum(item.mismatches for item in plain),
            "failures": [item.to_json() for item in plain_bad],
        },
        "verdict": {
            "structure_complete": structure["complete"],
            "storage_exact": not storage_bad,
            "source_exact": not source_bad,
            "baseline_bit_identical": not baseline_bad,
            "baseline_within_bound": not over_bound,
            "plain_identical": not plain_bad,
            "codes_within_lut_bounds": not unsafe,
            "pass": (
                structure["complete"]
                and not storage_bad
                and not source_bad
                and not over_bound
                and not unsafe
                and not plain_bad
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--weights-per-chunk", type=int, required=True)
    parser.add_argument(
        "--max-weight-difference",
        type=float,
        required=True,
        help="largest dequantised weight gap against the official checkpoint that still passes",
    )
    args = parser.parse_args(argv)

    result = verify(
        args.artifact,
        args.sidecar,
        args.gguf,
        args.baseline,
        weights_per_chunk=args.weights_per_chunk,
        max_weight_difference=args.max_weight_difference,
    )
    write_json_atomic(args.result, result)

    quantised = result["quantised"]
    verdict = result["verdict"]
    print(
        f"structure: {result['structure']['artifact_tensors']} tensors, complete="
        f"{result['structure']['complete']} {result['structure']['problems'] or ''}"
    )
    print(
        f"storage:   {quantised['sidecar_byte_mismatches']} byte mismatches against the sidecar, "
        f"{quantised['frozen_digest_mismatches']} frozen-digest mismatches over "
        f"{quantised['tensors']} tensors"
    )
    print(
        f"source:    {quantised['gguf_code_mismatches']} code and "
        f"{quantised['gguf_scale_mismatches']} scale mismatches against the GGUF over "
        f"{quantised['weights']} weights and {quantised['groups']} groups"
    )
    print(
        f"baseline:  {quantised['baseline_code_mismatches']} code, "
        f"{quantised['baseline_scale_mismatches']} scale, "
        f"{quantised['baseline_bias_mismatches']} bias mismatches, largest weight gap "
        f"{quantised['baseline_max_weight_difference']:.6g} "
        f"(allowed {quantised['baseline_max_weight_difference_allowed']:.6g})"
    )
    print(
        f"plain:     {result['plain']['tensors']} tensors, {result['plain']['mismatches']} "
        f"mismatches against the official checkpoint"
    )
    print(
        f"bounds:    max full byte {quantised['max_full_code_byte']} of "
        f"{quantised['max_full_code_byte_allowed']}, max tail byte "
        f"{quantised['max_tail_code_byte']} of {quantised['max_tail_code_byte_allowed']}, "
        f"max trit code {quantised['max_trit_code']}"
    )
    print(f"pass = {verdict['pass']}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
