from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .constants import MODEL_SHA256, REPORTS_DIR
from .format import (
    BLOCK_BYTES,
    SOURCE_BLOCK_BYTES,
    decode_q2_blocks,
    decode_tq1_blocks,
    read_sidecar,
    sha256_file,
)
from .gguf_utils import load_tensor_infos
from .inspect_model import DEFAULT_CHUNK_GROUPS, write_json_atomic


def verify_model(
    model_path: Path,
    sidecar_path: Path,
    *,
    chunk_groups: int = DEFAULT_CHUNK_GROUPS,
) -> dict:
    started = time.time()
    source_hash = sha256_file(model_path)
    if source_hash != MODEL_SHA256:
        raise ValueError(f"source SHA-256 mismatch: {source_hash}")
    header, manifest = read_sidecar(sidecar_path)
    if header.source_sha256 != source_hash or header.source_file_bytes != model_path.stat().st_size:
        raise ValueError("sidecar source identity does not match the GGUF")
    reader, tensors = load_tensor_infos(model_path)
    del reader
    source_by_name = {tensor.name: tensor for tensor in tensors if tensor.tensor_type == "Q2_0"}

    total_weights = 0
    total_groups = 0
    weight_mismatches = 0
    scale_mismatches = 0
    tensor_results: list[dict] = []
    progress = tqdm(manifest["tensors"], desc="Verifying TQ1 tensors", unit="tensor")
    for packed_info in progress:
        name = packed_info["name"]
        progress.set_postfix_str(name[:42])
        source = source_by_name.get(name)
        if source is None:
            raise ValueError(f"sidecar tensor {name} is absent from source")
        groups = int(packed_info["groups"])
        if groups != source.groups:
            raise ValueError(f"group count mismatch for {name}")
        source_map = np.memmap(
            model_path,
            mode="r",
            dtype=np.uint8,
            offset=source.data_offset,
            shape=(groups, SOURCE_BLOCK_BYTES),
        )
        packed_map = np.memmap(
            sidecar_path,
            mode="r",
            dtype=np.uint8,
            offset=int(packed_info["packed_offset"]),
            shape=(groups, BLOCK_BYTES),
        )
        source_weight_hash = hashlib.sha256()
        packed_weight_hash = hashlib.sha256()
        source_scale_hash = hashlib.sha256()
        packed_scale_hash = hashlib.sha256()
        source_packed_hash = hashlib.sha256()
        tensor_weight_mismatches = 0
        tensor_scale_mismatches = 0
        try:
            for start in range(0, groups, chunk_groups):
                end = min(start + chunk_groups, groups)
                source_scales, source_codes = decode_q2_blocks(np.asarray(source_map[start:end]))
                packed_bytes = np.asarray(packed_map[start:end])
                packed_scales, packed_codes = decode_tq1_blocks(packed_bytes, validate=True)
                tensor_weight_mismatches += int(np.count_nonzero(source_codes != packed_codes))
                tensor_scale_mismatches += int(
                    np.count_nonzero(np.any(source_scales != packed_scales, axis=1))
                )
                source_weight_hash.update(source_codes.tobytes(order="C"))
                packed_weight_hash.update(packed_codes.tobytes(order="C"))
                source_scale_hash.update(source_scales.tobytes(order="C"))
                packed_scale_hash.update(packed_scales.tobytes(order="C"))
                source_packed_hash.update(packed_bytes.tobytes(order="C"))
        finally:
            del source_map
            del packed_map
        source_weight_digest = source_weight_hash.hexdigest()
        packed_weight_digest = packed_weight_hash.hexdigest()
        source_scale_digest = source_scale_hash.hexdigest()
        packed_scale_digest = packed_scale_hash.hexdigest()
        payload_digest = source_packed_hash.hexdigest()
        manifest_hashes_match = (
            source_weight_digest == packed_info["logical_weight_sha256"]
            and source_scale_digest == packed_info["scale_bits_sha256"]
            and payload_digest == packed_info["packed_payload_sha256"]
        )
        if not manifest_hashes_match:
            raise ValueError(f"manifest hash mismatch for {name}")
        tensor_results.append(
            {
                "name": name,
                "weights": source.elements,
                "groups": groups,
                "weight_mismatches": tensor_weight_mismatches,
                "scale_mismatches": tensor_scale_mismatches,
                "source_weight_sha256": source_weight_digest,
                "packed_weight_sha256": packed_weight_digest,
                "source_scale_sha256": source_scale_digest,
                "packed_scale_sha256": packed_scale_digest,
                "weight_hash_match": source_weight_digest == packed_weight_digest,
                "scale_hash_match": source_scale_digest == packed_scale_digest,
            }
        )
        total_weights += source.elements
        total_groups += groups
        weight_mismatches += tensor_weight_mismatches
        scale_mismatches += tensor_scale_mismatches

    if len(source_by_name) != len(manifest["tensors"]):
        raise ValueError("not every source Q2_0 tensor is present in the sidecar")
    passed = weight_mismatches == 0 and scale_mismatches == 0 and all(
        tensor["weight_hash_match"] and tensor["scale_hash_match"] for tensor in tensor_results
    )
    return {
        "schema_version": 1,
        "source_sha256": source_hash,
        "sidecar_sha256": sha256_file(sidecar_path),
        "sidecar_file_bytes": sidecar_path.stat().st_size,
        "tensor_count": len(tensor_results),
        "weights_checked": total_weights,
        "groups_checked": total_groups,
        "weight_mismatches": weight_mismatches,
        "scale_mismatches": scale_mismatches,
        "pass": passed,
        "verdict_if_stopped": None if passed else "FAIL_NOT_LOSSLESS",
        "tensors": tensor_results,
        "elapsed_seconds": time.time() - started,
    }


def write_verification_report(result: dict, path: Path | None = None) -> None:
    path = path or REPORTS_DIR / "02_lossless_verification.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| {tensor['name']} | {tensor['weights']:,} | {tensor['groups']:,} | "
        f"{tensor['source_weight_sha256']} | {tensor['source_scale_sha256']} |"
        for tensor in result["tensors"]
    )
    status = "PASS" if result["pass"] else "FAIL_NOT_LOSSLESS"
    report = f"""# Lossless verification

**{status}**

| Check | Result |
| --- | ---: |
| Tensors checked | {result['tensor_count']:,} |
| Groups checked | {result['groups_checked']:,} |
| Weights checked | {result['weights_checked']:,} |
| Weight mismatches | {result['weight_mismatches']:,} |
| Scale mismatches | {result['scale_mismatches']:,} |
| Sidecar bytes | {result['sidecar_file_bytes']:,} |
| Sidecar SHA-256 | `{result['sidecar_sha256']}` |

Weights are hashed as canonical logical code bytes in original tensor order;
scales are hashed as their untouched two raw bytes per g128 group. Original and
TQ1 representations were decoded independently in this verification pass.

| Tensor | Weights | Groups | Logical weight SHA-256 | Scale-bit SHA-256 |
| --- | ---: | ---: | --- | --- |
{rows}
"""
    path.write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exhaustively verify a TQ1_G128 sidecar")
    parser.add_argument("model", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "02_lossless_verification.md")
    parser.add_argument("--chunk-groups", type=int, default=DEFAULT_CHUNK_GROUPS)
    args = parser.parse_args(argv)
    try:
        result = verify_model(
            args.model,
            args.sidecar,
            chunk_groups=args.chunk_groups,
        )
        write_json_atomic(args.result, result)
        write_verification_report(result, args.report)
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: value for key, value in result.items() if key != "tensors"}, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

