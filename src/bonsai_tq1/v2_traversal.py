from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .constants import MODEL_SHA256, PRISM_COMMIT
from .format import BLOCK_BYTES, BLOCK_SIZE, SOURCE_BLOCK_BYTES, read_sidecar, sha256_file
from .inspect_model import write_json_atomic


DEFAULT_L2_BYTES = 100_663_296
LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)\.weight$")


@dataclass(frozen=True)
class TraversalTensor:
    order: int
    source_index: int
    layer: int
    kind: str
    name: str
    m: int
    k: int
    groups: int
    q2_file_offset: int
    q2_bytes: int
    tq1_file_offset: int
    tq1_bytes: int
    q2_stream_offset: int
    tq1_stream_offset: int
    q8_stream_offset: int
    output_stream_offset: int
    activation_origin: str
    logical_weight_sha256: str
    scale_bits_sha256: str
    packed_payload_sha256: str


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256_reports(reports_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(reports_dir.glob("*.md")):
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _prism_head(prism_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=prism_dir,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _expected_layer_order(available: set[str], layer: int) -> list[tuple[str, str]]:
    prefix = f"blk.{layer}."
    full_attention = f"{prefix}attn_q.weight" in available
    if full_attention:
        suffixes = (
            ("attn_q", "attention_q"),
            ("attn_k", "attention_k"),
            ("attn_v", "attention_v"),
            ("attn_output", "attention_output"),
            ("ffn_gate", "mlp_gate"),
            ("ffn_up", "mlp_up"),
            ("ffn_down", "mlp_down"),
        )
    else:
        suffixes = (
            ("attn_qkv", "recurrent_qkv"),
            ("attn_gate", "recurrent_gate"),
            ("ssm_beta", "recurrent_beta"),
            ("ssm_alpha", "recurrent_alpha"),
            ("ssm_out", "recurrent_output"),
            ("ffn_gate", "mlp_gate"),
            ("ffn_up", "mlp_up"),
            ("ffn_down", "mlp_down"),
        )
    result = [(f"{prefix}{suffix}.weight", kind) for suffix, kind in suffixes]
    missing = [name for name, _ in result if name not in available]
    if missing:
        raise ValueError(f"layer {layer} traversal tensors are missing: {missing}")
    return result


def build_traversal(
    *,
    model_path: Path,
    sidecar_path: Path,
    audit_path: Path,
    verification_path: Path,
    prism_dir: Path,
    v1_reports_dir: Path,
    l2_bytes: int,
    verify_file_hashes: bool,
) -> dict:
    if l2_bytes <= 0:
        raise ValueError("L2 size must be positive")
    audit = _load_json(audit_path)
    verification = _load_json(verification_path)
    header, sidecar_manifest = read_sidecar(sidecar_path)

    if _prism_head(prism_dir) != PRISM_COMMIT:
        raise ValueError("Prism checkout is not at the preregistered commit")
    if model_path.stat().st_size != header.source_file_bytes:
        raise ValueError("model size differs from the sidecar source identity")
    if header.source_sha256 != MODEL_SHA256:
        raise ValueError("sidecar does not identify the released model")
    if not verification.get("pass"):
        raise ValueError("the frozen model-wide lossless verification did not pass")

    model_hash = sha256_file(model_path) if verify_file_hashes else MODEL_SHA256
    sidecar_hash = sha256_file(sidecar_path) if verify_file_hashes else verification["sidecar_sha256"]
    if model_hash != MODEL_SHA256:
        raise ValueError("model SHA-256 mismatch")
    if sidecar_hash != verification["sidecar_sha256"]:
        raise ValueError("sidecar SHA-256 mismatch")

    audit_by_name = {tensor["name"]: tensor for tensor in audit["tensors"]}
    packed_by_name = {tensor["name"]: tensor for tensor in sidecar_manifest["tensors"]}
    verified_by_name = {tensor["name"]: tensor for tensor in verification["tensors"]}
    if not (audit_by_name.keys() == packed_by_name.keys() == verified_by_name.keys()):
        raise ValueError("audit, sidecar, and verification tensor sets differ")

    available = set(audit_by_name)
    layer_ids = sorted(
        {
            int(match.group(1))
            for name in available
            if (match := LAYER_RE.match(name)) is not None
        }
    )
    if layer_ids != list(range(64)):
        raise ValueError(f"expected decoder layers 0..63, found {layer_ids}")

    ordered: list[tuple[str, int, str]] = []
    for layer in layer_ids:
        ordered.extend((name, layer, kind) for name, kind in _expected_layer_order(available, layer))
    if "output.weight" not in available:
        raise ValueError("released checkpoint has no compatible output head")
    ordered.append(("output.weight", -1, "lm_head"))

    selected_names = {name for name, _, _ in ordered}
    expected_excluded = {"token_embd.weight"}
    unexpected = available - selected_names - expected_excluded
    if unexpected:
        raise ValueError(f"unclassified Q2 tensors: {sorted(unexpected)}")
    if len(ordered) != 497 or len(selected_names) != len(ordered):
        raise ValueError("expected exactly 497 unique GEMV launches")

    tensors: list[TraversalTensor] = []
    q2_stream_offset = 0
    tq1_stream_offset = 0
    q8_stream_offset = 0
    output_stream_offset = 0
    full_attention_layers = 0
    recurrent_layers = 0
    for order, (name, layer, kind) in enumerate(ordered):
        audit_tensor = audit_by_name[name]
        packed_tensor = packed_by_name[name]
        verified_tensor = verified_by_name[name]
        shape = tuple(int(value) for value in audit_tensor["shape_gguf"])
        if len(shape) != 2:
            raise ValueError(f"{name} is not a matrix")
        k, m = shape
        groups = int(audit_tensor["groups"])
        q2_bytes = int(audit_tensor["source_bytes"])
        tq1_bytes = int(packed_tensor["packed_bytes"])
        if k % BLOCK_SIZE or groups != m * k // BLOCK_SIZE:
            raise ValueError(f"{name} has incompatible block geometry")
        if q2_bytes != groups * SOURCE_BLOCK_BYTES or tq1_bytes != groups * BLOCK_BYTES:
            raise ValueError(f"{name} byte extent does not match its group count")
        if q2_stream_offset % 256 or tq1_stream_offset % 256:
            raise ValueError(f"stream extent for {name} is not naturally 256-byte aligned")
        logical_hash = str(audit_tensor["logical_weight_sha256"])
        scale_hash = str(audit_tensor["scale_bits_sha256"])
        if (
            logical_hash != packed_tensor["logical_weight_sha256"]
            or logical_hash != verified_tensor["source_weight_sha256"]
            or logical_hash != verified_tensor["packed_weight_sha256"]
            or scale_hash != packed_tensor["scale_bits_sha256"]
            or scale_hash != verified_tensor["source_scale_sha256"]
            or scale_hash != verified_tensor["packed_scale_sha256"]
            or not verified_tensor["weight_hash_match"]
            or not verified_tensor["scale_hash_match"]
            or verified_tensor["weight_mismatches"]
            or verified_tensor["scale_mismatches"]
        ):
            raise ValueError(f"frozen parity evidence does not match for {name}")
        tensors.append(
            TraversalTensor(
                order=order,
                source_index=int(audit_tensor["index"]),
                layer=layer,
                kind=kind,
                name=name,
                m=m,
                k=k,
                groups=groups,
                q2_file_offset=int(audit_tensor["source_offset"]),
                q2_bytes=q2_bytes,
                tq1_file_offset=int(packed_tensor["packed_offset"]),
                tq1_bytes=tq1_bytes,
                q2_stream_offset=q2_stream_offset,
                tq1_stream_offset=tq1_stream_offset,
                q8_stream_offset=q8_stream_offset,
                output_stream_offset=output_stream_offset,
                activation_origin="FP16+BF16",
                logical_weight_sha256=logical_hash,
                scale_bits_sha256=scale_hash,
                packed_payload_sha256=str(packed_tensor["packed_payload_sha256"]),
            )
        )
        q2_stream_offset += q2_bytes
        tq1_stream_offset += tq1_bytes
        q8_stream_offset += k // 32
        output_stream_offset += m

    full_attention_layers = sum(
        1 for layer in layer_ids if f"blk.{layer}.attn_q.weight" in available
    )
    recurrent_layers = len(layer_ids) - full_attention_layers
    q2_l2_multiple = q2_stream_offset / l2_bytes
    tq1_l2_multiple = tq1_stream_offset / l2_bytes
    substantial = min(q2_l2_multiple, tq1_l2_multiple) >= 10.0
    result = {
        "schema_version": 2,
        "valid": substantial,
        "invalid_reason": None if substantial else "INVALID_STREAMING_WORKSET",
        "source": {
            "model": str(model_path),
            "model_bytes": model_path.stat().st_size,
            "model_sha256": model_hash,
            "sidecar": str(sidecar_path),
            "sidecar_bytes": sidecar_path.stat().st_size,
            "sidecar_sha256": sidecar_hash,
            "prism_commit": PRISM_COMMIT,
        },
        "selection": {
            "decoder_layers": len(layer_ids),
            "recurrent_layers": recurrent_layers,
            "full_attention_layers": full_attention_layers,
            "gemv_launches": len(tensors),
            "included_lm_head": True,
            "excluded_token_embedding": "GET_ROWS lookup, not a GEMV; no compatible MMVQ traversal launch",
            "excluded_non_q2_tensors": int(audit["source"]["tensor_count"]) - len(audit["tensors"]),
        },
        "working_set": {
            "l2_bytes": l2_bytes,
            "q2_bytes_per_traversal": q2_stream_offset,
            "tq1_bytes_per_traversal": tq1_stream_offset,
            "q2_l2_multiple": q2_l2_multiple,
            "tq1_l2_multiple": tq1_l2_multiple,
            "substantial_threshold_l2_multiple": 10.0,
            "substantially_exceeds_l2": substantial,
            "combined_device_weight_allocations": q2_stream_offset + tq1_stream_offset,
            "correctness_activation_vectors_per_tensor": 2,
            "q8_blocks_per_vector": q8_stream_offset,
            "q8_bytes_per_vector": q8_stream_offset * 36,
            "q8_total_bytes": q8_stream_offset * 36 * 2,
            "output_values_per_vector": output_stream_offset,
            "output_bytes_per_vector_per_format": output_stream_offset * 4,
            "output_total_bytes_per_format": output_stream_offset * 4 * 2,
        },
        "parity": {
            "all_selected_tensors_previously_exhaustive": True,
            "selected_weights": sum(t.m * t.k for t in tensors),
            "selected_groups": sum(t.groups for t in tensors),
            "weight_mismatches": 0,
            "raw_scale_bit_mismatches": 0,
            "per_tensor_hashes_match": True,
        },
        "anti_cheating": {
            "real_checkpoint_extents_only": True,
            "validated_sidecar_extents_only": True,
            "tq1_unpacked_weight_buffer": False,
            "stream_allocation_bytes_equal_payload_bytes": True,
        },
        "v1_frozen_report_sha256": _sha256_reports(v1_reports_dir),
        "tensors": [asdict(tensor) for tensor in tensors],
    }
    return result


def write_tsv(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field.name for field in TraversalTensor.__dataclass_fields__.values()]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect="excel-tab", lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["tensors"])
    temporary.replace(path)


def write_report(path: Path, result: dict) -> None:
    selection = result["selection"]
    working = result["working_set"]
    parity = result["parity"]
    lines = [
        "# V2 model traversal",
        "",
        "## Gate result",
        "",
        (
            "**PASS.** Both representation-specific streams exceed the RTX 6000 Ada L2 "
            "by more than the preregistered 10× substantial-workset threshold."
            if result["valid"]
            else "**INVALID_STREAMING_WORKSET.** The selected stream does not substantially exceed L2."
        ),
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| Decoder layers | {selection['decoder_layers']:,} |",
        f"| Recurrent / full-attention layers | {selection['recurrent_layers']:,} / {selection['full_attention_layers']:,} |",
        f"| GEMV launches per traversal | {selection['gemv_launches']:,} |",
        f"| Q2 bytes touched | {working['q2_bytes_per_traversal']:,} B |",
        f"| TQ1 bytes touched | {working['tq1_bytes_per_traversal']:,} B |",
        f"| RTX 6000 Ada L2 | {working['l2_bytes']:,} B |",
        f"| Q2 / TQ1 working set in L2 multiples | {working['q2_l2_multiple']:.3f}× / {working['tq1_l2_multiple']:.3f}× |",
        f"| Combined benchmark weight allocation | {working['combined_device_weight_allocations']:,} B |",
        f"| Quantization groups touched | {parity['selected_groups']:,} |",
        f"| Logical weights touched | {parity['selected_weights']:,} |",
        "",
        "The sequence follows the pinned Qwen3.5 graph: recurrent layers launch QKV, gate, beta, alpha, and output projections; full-attention layers launch Q, K, V, and output projections; every layer then launches gate, up, and down FFN projections. The compatible `output.weight` LM head is last. `token_embd.weight` is intentionally excluded because decode uses it through `GET_ROWS`, not the Q2 MMVQ launcher. Non-Q2 norms, convolution/state tensors, and biases are outside the representation comparison.",
        "",
        "Every selected extent is tied to the released checkpoint and already-validated sidecar by matching logical-weight, raw-scale-bit, and packed-payload hashes. Stream offsets are exact concatenations; there is no expanded TQ1 allocation.",
        "",
        "## Exact launch order",
        "",
        "GGUF shape is `[K, M]`; the table reports the GEMV convention `M × K`.",
        "",
        "| # | Layer | Kind | Tensor | M | K | Q2 bytes | TQ1 bytes | Groups |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tensor in result["tensors"]:
        layer = "head" if tensor["layer"] < 0 else str(tensor["layer"])
        lines.append(
            f"| {tensor['order']} | {layer} | {tensor['kind']} | `{tensor['name']}` | "
            f"{tensor['m']:,} | {tensor['k']:,} | {tensor['q2_bytes']:,} | "
            f"{tensor['tq1_bytes']:,} | {tensor['groups']:,} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the exact Bonsai V2 GEMV traversal")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--prism-dir", type=Path, required=True)
    parser.add_argument("--v1-reports", type=Path, required=True)
    parser.add_argument("--l2-bytes", type=int, default=DEFAULT_L2_BYTES)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip-file-hashes", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_traversal(
            model_path=args.model,
            sidecar_path=args.sidecar,
            audit_path=args.audit,
            verification_path=args.verification,
            prism_dir=args.prism_dir,
            v1_reports_dir=args.v1_reports,
            l2_bytes=args.l2_bytes,
            verify_file_hashes=not args.skip_file_hashes,
        )
        write_json_atomic(args.output_json, result)
        write_tsv(args.output_tsv, result)
        write_report(args.report, result)
    except Exception as exc:
        print(f"V2 traversal failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"launches={result['selection']['gemv_launches']} "
        f"q2_bytes={result['working_set']['q2_bytes_per_traversal']} "
        f"tq1_bytes={result['working_set']['tq1_bytes_per_traversal']} "
        f"valid={result['valid']}"
    )
    return 0 if result["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
