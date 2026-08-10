"""Audit whether the official MLX affine 2-bit distribution is lossless ternary.

`prism-ml/Ternary-Bonsai-27B-mlx-2bit` is the natural Apple Silicon baseline for
Ternel. MLX affine quantisation dequantises as ``w = scale * q + bias`` with
``q in {0,1,2,3}``. Exact ternary is representable in that scheme
(``scale = s``, ``bias = -s``, ``q in {0,1,2}`` giving ``{-s, 0, +s}``), but a
naive min/max quantiser would instead produce four levels
``{-s, -s/3, +s/3, +s}`` and silently destroy the ternary structure.

Which of the two it actually is decides how the whole Ternel MLX result is
framed, so it is measured here rather than assumed:

* four levels  -> TQ1_G128 is both exact and 22.2% smaller than the baseline;
* three levels -> the baseline is already lossless and the TQ1_G128 claim is a
  memory-density result (1.75 vs 2.25 bits/weight) plus whatever the Metal
  kernel delivers in speed.

The audit reads only the safetensors header plus the byte ranges of the audited
tensors, so it costs a few MB rather than the full 8.49 GB shard.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bonsai_tq1.format import BLOCK_SIZE, FormatError, write_json_atomic

SCHEMA_VERSION = 1

BASELINE_REPO = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
BASELINE_REVISION = "main"
BASELINE_FILENAME = "model.safetensors"

# MLX affine quantisation constants for this checkpoint.
BASELINE_BITS = 2
BASELINE_GROUP_SIZE = 128
VALUES_PER_UINT32 = 32 // BASELINE_BITS
QUANT_MASK = (1 << BASELINE_BITS) - 1

# Storage cost of one 128-weight group, in bytes.
BASELINE_GROUP_BYTES = (BASELINE_GROUP_SIZE * BASELINE_BITS) // 8 + 2 + 2  # packed + scale + bias
TQ1_GROUP_BYTES = 28

# Tensors audited by default: both layer types, both tile regimes (rows=48 and
# rows divisible by 256), and all three group counts (40, 48, 136).
AUDIT_TENSORS = (
    "language_model.model.layers.0.linear_attn.in_proj_a.weight",
    "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
    "language_model.model.layers.0.linear_attn.out_proj.weight",
    "language_model.model.layers.0.mlp.down_proj.weight",
    "language_model.model.layers.3.self_attn.k_proj.weight",
    "language_model.model.layers.3.self_attn.q_proj.weight",
)


def resolve_url(repo: str, revision: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def unpack_affine_codes(packed: np.ndarray, *, columns: int) -> np.ndarray:
    """Unpack MLX affine 2-bit codes: 16 values per uint32, low bits first."""
    if packed.dtype != np.dtype("<u4"):
        raise FormatError(f"expected little-endian uint32 codes, got {packed.dtype}")
    if packed.ndim != 2:
        raise FormatError(f"expected a 2-D packed weight, got shape {packed.shape}")
    shifts = (np.arange(VALUES_PER_UINT32, dtype=np.uint32) * BASELINE_BITS)
    codes = ((packed[:, :, None] >> shifts) & QUANT_MASK).astype(np.uint8)
    flat = codes.reshape(packed.shape[0], packed.shape[1] * VALUES_PER_UINT32)
    if flat.shape[1] != columns:
        raise FormatError(f"unpacked {flat.shape[1]} columns, expected {columns}")
    return flat


@dataclass(frozen=True)
class TensorFidelity:
    name: str
    rows: int
    columns: int
    groups_per_row: int
    code_histogram: dict[int, int]
    max_code: int
    uses_fourth_code: bool
    bias_over_scale_min: float
    bias_over_scale_max: float
    bias_equals_negative_scale: bool
    zero_scale_groups: int
    distinct_values_per_group: dict[int, int]
    is_lossless_ternary: bool

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": self.columns,
            "groups_per_row": self.groups_per_row,
            "code_histogram": {str(k): v for k, v in sorted(self.code_histogram.items())},
            "max_code": self.max_code,
            "uses_fourth_code": self.uses_fourth_code,
            "bias_over_scale_min": self.bias_over_scale_min,
            "bias_over_scale_max": self.bias_over_scale_max,
            "bias_equals_negative_scale": self.bias_equals_negative_scale,
            "zero_scale_groups": self.zero_scale_groups,
            "distinct_values_per_group": {
                str(k): v for k, v in sorted(self.distinct_values_per_group.items())
            },
            "is_lossless_ternary": self.is_lossless_ternary,
        }


def audit_tensor(remote, weight_name: str) -> TensorFidelity:
    if not weight_name.endswith(".weight"):
        raise FormatError(f"expected a '.weight' tensor name, got {weight_name!r}")
    stem = weight_name[: -len(".weight")]
    packed = remote.read(weight_name)
    scales = remote.read(f"{stem}.scales")
    biases = remote.read(f"{stem}.biases")

    rows, groups_per_row = scales.shape
    if biases.shape != scales.shape:
        raise FormatError(f"{stem}: scales {scales.shape} and biases {biases.shape} differ")
    if packed.shape[0] != rows:
        raise FormatError(f"{stem}: packed rows {packed.shape[0]} != scale rows {rows}")
    columns = groups_per_row * BASELINE_GROUP_SIZE
    if columns % BLOCK_SIZE:
        raise FormatError(f"{stem}: {columns} columns is not a multiple of {BLOCK_SIZE}")

    codes = unpack_affine_codes(packed, columns=columns)
    values, counts = np.unique(codes, return_counts=True)
    histogram = {int(v): int(c) for v, c in zip(values, counts)}
    max_code = int(codes.max())

    # float64 so the ratio test is exact rather than fp16-rounded.
    scale64 = scales.astype(np.float64)
    bias64 = biases.astype(np.float64)
    zero_scale = int(np.count_nonzero(scale64 == 0.0))
    nonzero = scale64 != 0.0
    if not nonzero.any():
        raise FormatError(f"{stem}: every scale is zero")
    ratio = bias64[nonzero] / scale64[nonzero]
    ratio_min = float(ratio.min())
    ratio_max = float(ratio.max())
    bias_is_negative_scale = bool(np.array_equal(bias64, -scale64))

    # Count distinct dequantised values per group, exactly.
    dequantised = scale64.repeat(BASELINE_GROUP_SIZE, axis=1) * codes.astype(np.float64)
    dequantised += bias64.repeat(BASELINE_GROUP_SIZE, axis=1)
    grouped = dequantised.reshape(rows, groups_per_row, BASELINE_GROUP_SIZE)
    grouped.sort(axis=2)
    distinct = 1 + np.count_nonzero(np.diff(grouped, axis=2), axis=2)
    distinct_values, distinct_counts = np.unique(distinct, return_counts=True)
    distinct_histogram = {int(v): int(c) for v, c in zip(distinct_values, distinct_counts)}

    lossless = max_code <= 2 and bias_is_negative_scale and zero_scale == 0

    return TensorFidelity(
        name=weight_name,
        rows=int(rows),
        columns=int(columns),
        groups_per_row=int(groups_per_row),
        code_histogram=histogram,
        max_code=max_code,
        uses_fourth_code=max_code > 2,
        bias_over_scale_min=ratio_min,
        bias_over_scale_max=ratio_max,
        bias_equals_negative_scale=bias_is_negative_scale,
        zero_scale_groups=zero_scale,
        distinct_values_per_group=distinct_histogram,
        is_lossless_ternary=lossless,
    )


def audit(repo: str, revision: str, filename: str, tensors: tuple[str, ...], *, timeout_seconds: float) -> dict:
    from .remote_safetensors import RemoteSafetensors

    url = resolve_url(repo, revision, filename)
    remote = RemoteSafetensors(url, timeout_seconds=timeout_seconds)
    audited = [audit_tensor(remote, name) for name in tensors]

    all_lossless = all(item.is_lossless_ternary for item in audited)
    any_fourth = any(item.uses_fourth_code for item in audited)
    if all_lossless == any_fourth:
        raise FormatError("contradictory fidelity findings across audited tensors")

    saving = (BASELINE_GROUP_BYTES - TQ1_GROUP_BYTES) / BASELINE_GROUP_BYTES
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": {
            "repo": repo,
            "revision": revision,
            "filename": filename,
            "url": url,
            "safetensors_metadata": remote.metadata,
            "tensor_count": len(remote),
            "bits": BASELINE_BITS,
            "group_size": BASELINE_GROUP_SIZE,
        },
        "density": {
            "baseline_bytes_per_group": BASELINE_GROUP_BYTES,
            "baseline_bits_per_weight": BASELINE_GROUP_BYTES * 8 / BASELINE_GROUP_SIZE,
            "tq1_bytes_per_group": TQ1_GROUP_BYTES,
            "tq1_bits_per_weight": TQ1_GROUP_BYTES * 8 / BASELINE_GROUP_SIZE,
            "tq1_saving_fraction": saving,
        },
        "tensors": [item.to_json() for item in audited],
        "verdict": {
            "baseline_is_lossless_ternary": all_lossless,
            "framing": (
                "memory-density result: the baseline already stores exact ternary weights, so "
                "TQ1_G128 competes on bits/weight and kernel speed, not on accuracy"
                if all_lossless
                else "accuracy and density result: the baseline quantises ternary weights to four "
                "levels, so TQ1_G128 is exact where the baseline is not"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--revision", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--tensor", type=str, action="append")
    args = parser.parse_args(argv)

    tensors = tuple(args.tensor) if args.tensor else AUDIT_TENSORS
    result = audit(args.repo, args.revision, args.filename, tensors, timeout_seconds=args.timeout_seconds)
    write_json_atomic(args.result, result)

    verdict = result["verdict"]
    print(f"audited {len(tensors)} tensors from {args.repo}")
    print(f"baseline_is_lossless_ternary = {verdict['baseline_is_lossless_ternary']}")
    print(
        f"density: baseline {result['density']['baseline_bits_per_weight']:.3f} bits/weight -> "
        f"TQ1 {result['density']['tq1_bits_per_weight']:.3f} bits/weight "
        f"({result['density']['tq1_saving_fraction'] * 100:.2f}% smaller)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
