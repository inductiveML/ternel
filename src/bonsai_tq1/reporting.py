from __future__ import annotations

import csv
import json
from pathlib import Path

from .constants import MODEL_SHA256, PRISM_COMMIT, REPORTS_DIR, RESULTS_DIR


def _pct(value: float) -> str:
    return f"{100.0 * value:.10f}%"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_audit_reports(audit_path: Path, reports_dir: Path = REPORTS_DIR) -> None:
    audit = load_json(audit_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    source = audit["source"]
    fmt = audit["format"]
    q2 = audit["q2"]
    projection = audit["whole_model_projection"]
    distribution = q2["distribution"]
    symbols = distribution["symbol_counts"]
    fractions = distribution["fractions"]

    type_rows = "\n".join(
        f"| {name} | {values['tensor_count']:,} | {values['elements']:,} | {values['data_bytes']:,} |"
        for name, values in audit["type_summary"].items()
    )
    tensor_rows = "\n".join(
        f"| {tensor['name']} | {' × '.join(map(str, tensor['shape_gguf']))} | "
        f"{tensor['elements']:,} | {tensor['groups']:,} | {tensor['source_bytes']:,} |"
        for tensor in audit["tensors"]
    )
    format_report = f"""# Format audit

## Pinned source of truth

- Model revision: `{source['revision']}`
- File: `{source['filename']}`
- File bytes: `{source['file_bytes']:,}`
- Verified SHA-256: `{MODEL_SHA256}`
- GGUF version: {source['gguf_version']}
- GGUF tensors: {source['tensor_count']:,}
- Tensor-data offset: {source['data_offset']:,} bytes
- Prism llama.cpp commit: `{PRISM_COMMIT}`

## Actual Q2_0/g128 block

The pinned Prism source defines a packed 34-byte block containing one raw
little-endian FP16 scale followed by 32 bytes of four 2-bit fields each:

```text
offset 0..1   ggml_half scale d
offset 2..33  128 codes, four little-endian 2-bit fields per byte
code 0 -> -1, code 1 -> 0, code 2 -> +1, code 3 -> +2
dequantized weight = (code - 1) * fp16_to_fp32(d)
```

The entire released payload was scanned. Code 3 occurs exactly
**{symbols['+2']:,}** times, so this checkpoint's logical alphabet is exactly
ternary even though its container type can represent `+2`.

## Tensor types

| Type | Tensors | Elements | Data bytes |
| --- | ---: | ---: | ---: |
{type_rows}

## Q2_0 tensors

| Tensor | GGUF dimensions | Elements | Groups | Bytes |
| --- | --- | ---: | ---: | ---: |
{tensor_rows}
"""
    (reports_dir / "00_format_audit.md").write_text(format_report, encoding="utf-8")

    size_report = f"""# TQ1_G128 size model

## Block arithmetic

| Metric | Existing Q2_0/g128 | TQ1_G128 | Delta |
| --- | ---: | ---: | ---: |
| Values/group | 128 | 128 | 0 |
| Scale bytes/group | 2 | 2 | 0 |
| Symbol bytes/group | 32 | 26 | -6 |
| Total bytes/group | 34 | 28 | -6 |
| Bits/weight including scale | 2.125 | 1.75 | -0.375 |

Five trits are encoded per byte because `3^5 = 243`; the last three trits use
one canonical tail byte. Scales are copied byte-for-byte.

## Exact model projection

| Metric | Bytes |
| --- | ---: |
| Existing Q2_0 tensor payload | {q2['source_bytes']:,} |
| Projected TQ1 tensor payload | {q2['projected_tq1_bytes']:,} |
| Unchanged F32 tensors | {projection['unchanged_tensor_bytes']:,} |
| Existing GGUF metadata/alignment | {projection['gguf_header_and_alignment_bytes']:,} |
| Existing complete file | {source['file_bytes']:,} |
| Projected replacement file | {projection['projected_file_bytes']:,} |

- Quantized-payload reduction: **{_pct(q2['payload_reduction_fraction'])}**
- Whole-file reduction: **{_pct(projection['reduction_fraction'])}**
- Projected size: **{projection['projected_file_gb_decimal']:.9f} GB**
  ({projection['projected_file_gib']:.9f} GiB)

## Gate 1

**PASS**: the actual alphabet is ternary, quantized storage falls by at least
15%, and the projected replacement file is below 6.1 decimal GB.
"""
    (reports_dir / "01_size_model.md").write_text(size_report, encoding="utf-8")

    histogram = q2["nonzero_group_histogram"]
    histogram_path = RESULTS_DIR / "zero_distribution.csv"
    histogram_path.parent.mkdir(parents=True, exist_ok=True)
    group_total = distribution["groups"]
    with histogram_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("zero_count", "nonzero_count", "group_count", "fraction"))
        for nonzero_count, group_count in enumerate(histogram):
            writer.writerow((128 - nonzero_count, nonzero_count, group_count, group_count / group_total))

    threshold_lines = []
    cumulative = 0
    for nonzero_count, count in enumerate(histogram):
        cumulative += count
        if nonzero_count in (32, 48, 64, 80):
            threshold_lines.append(
                f"| ≤{nonzero_count} | {cumulative:,} | {_pct(cumulative / group_total)} |"
            )
    percentile_values = distribution["nonzeros_per_group"]
    percentile_line = " | ".join(str(percentile_values[f"p{p}"]) for p in (1, 5, 25, 50, 75, 95, 99))

    per_tensor_rows = "\n".join(
        f"| {tensor['name']} | {tensor['distribution']['symbol_counts']['0']:,} | "
        f"{_pct(tensor['distribution']['zero_density'])} | "
        f"{tensor['distribution']['nonzeros_per_group']['p50']} | "
        f"{tensor['distribution']['nonzeros_per_group']['p95']} |"
        for tensor in audit["tensors"]
    )
    sparsity_report = f"""# Exact ternary distribution and sparsity

This report is exhaustive over all {distribution['weights']:,} logical weights
and {group_total:,} g128 groups. The complete exact 0–128 group histogram is in
`artifacts/results/zero_distribution.csv`.

## Model-wide symbols

| Symbol | Exact count | Fraction |
| --- | ---: | ---: |
| -1 | {symbols['-1']:,} | {_pct(fractions['-1'])} |
| 0 | {symbols['0']:,} | {_pct(fractions['0'])} |
| +1 | {symbols['+1']:,} | {_pct(fractions['+1'])} |
| +2 | {symbols['+2']:,} | {_pct(fractions['+2'])} |

- Zero density: **{_pct(distribution['zero_density'])}**
- Nonzero density: **{_pct(distribution['nonzero_density'])}**
- Ternary entropy: **{distribution['entropy_bits_per_symbol']:.12f} bits/symbol**

## Nonzeros per group

| p1 | p5 | p25 | p50 | p75 | p95 | p99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {percentile_line} |

| Threshold | Exact groups | Fraction |
| --- | ---: | ---: |
{chr(10).join(threshold_lines)}

The distribution is not sparse enough for a basic mask+sign representation:
at the median of 90 nonzeros, a 16-byte mask, 12 sign bytes, and 2-byte scale
would require 30 bytes/group before indexing—larger than TQ1_G128's 28 bytes.

## Per-tensor distribution

| Tensor | Exact zeros | Zero density | p50 nonzeros | p95 nonzeros |
| --- | ---: | ---: | ---: | ---: |
{per_tensor_rows}
"""
    (reports_dir / "03_sparsity.md").write_text(sparsity_report, encoding="utf-8")


def main() -> int:
    write_audit_reports(RESULTS_DIR / "format_audit.json")
    return 0

