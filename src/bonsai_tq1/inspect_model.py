from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .constants import MODEL_FILENAME, MODEL_REPO, MODEL_REVISION, MODEL_SHA256, MODEL_SIZE
from .format import (
    BLOCK_BYTES,
    BLOCK_SIZE,
    SOURCE_BLOCK_BYTES,
    decode_q2_blocks,
    iter_memmap_blocks,
    write_json_atomic,
)
from .gguf_utils import load_tensor_infos, summarize_types
from .stats import summarize_symbols

DEFAULT_CHUNK_GROUPS = 1 << 18


def inspect_model(model_path: Path, *, chunk_groups: int = DEFAULT_CHUNK_GROUPS) -> dict:
    started = time.time()
    actual_size = model_path.stat().st_size
    if actual_size != MODEL_SIZE:
        raise ValueError(f"source size mismatch: expected {MODEL_SIZE}, got {actual_size}")

    reader, tensors = load_tensor_infos(model_path)
    q2_tensors = [tensor for tensor in tensors if tensor.tensor_type == "Q2_0"]
    type_summary = summarize_types(tensors)
    total_symbols = np.zeros(4, dtype=np.int64)
    total_nonzero_histogram = np.zeros(BLOCK_SIZE + 1, dtype=np.int64)
    tensor_results: list[dict] = []

    progress = tqdm(q2_tensors, desc="Scanning Q2_0 tensors", unit="tensor")
    for tensor in progress:
        progress.set_postfix_str(tensor.name[:42])
        symbols = np.zeros(4, dtype=np.int64)
        histogram = np.zeros(BLOCK_SIZE + 1, dtype=np.int64)
        weight_hash = hashlib.sha256()
        scale_hash = hashlib.sha256()
        groups_seen = 0
        for blocks in iter_memmap_blocks(
            model_path,
            offset=tensor.data_offset,
            block_count=tensor.groups,
            block_bytes=SOURCE_BLOCK_BYTES,
            chunk_groups=chunk_groups,
        ):
            scales, codes = decode_q2_blocks(blocks)
            symbols += np.bincount(codes.reshape(-1), minlength=4).astype(np.int64)
            nonzero_counts = np.count_nonzero(codes != 1, axis=1)
            histogram += np.bincount(nonzero_counts, minlength=BLOCK_SIZE + 1).astype(np.int64)
            weight_hash.update(codes.tobytes(order="C"))
            scale_hash.update(scales.tobytes(order="C"))
            groups_seen += codes.shape[0]
        if groups_seen != tensor.groups:
            raise AssertionError(f"group scan mismatch for {tensor.name}")
        total_symbols += symbols
        total_nonzero_histogram += histogram
        tensor_results.append(
            {
                "index": tensor.index,
                "name": tensor.name,
                "shape_gguf": list(tensor.shape),
                "elements": tensor.elements,
                "groups": tensor.groups,
                "source_offset": tensor.data_offset,
                "source_bytes": tensor.data_bytes,
                "logical_weight_sha256": weight_hash.hexdigest(),
                "scale_bits_sha256": scale_hash.hexdigest(),
                "distribution": summarize_symbols(symbols, histogram),
                "nonzero_group_histogram": histogram.tolist(),
            }
        )

    q2_elements = sum(tensor.elements for tensor in q2_tensors)
    q2_bytes = sum(tensor.data_bytes for tensor in q2_tensors)
    q2_groups = sum(tensor.groups for tensor in q2_tensors)
    projected_q2_bytes = q2_groups * BLOCK_BYTES
    unchanged_tensor_bytes = sum(tensor.data_bytes for tensor in tensors if tensor.tensor_type != "Q2_0")
    gguf_header_and_alignment = actual_size - sum(tensor.data_bytes for tensor in tensors)
    projected_file_bytes = gguf_header_and_alignment + unchanged_tensor_bytes + projected_q2_bytes
    q2_reduction = 1.0 - projected_q2_bytes / q2_bytes
    whole_reduction = 1.0 - projected_file_bytes / actual_size
    has_fourth_symbol = int(total_symbols[3]) != 0
    packing_gate_pass = (
        not has_fourth_symbol
        and q2_reduction >= 0.15
        and projected_file_bytes <= 6_100_000_000
    )

    result = {
        "schema_version": 1,
        "source": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "filename": MODEL_FILENAME,
            "expected_sha256": MODEL_SHA256,
            "file_bytes": actual_size,
            "gguf_version": int(reader.fields["GGUF.version"].contents()),
            "tensor_count": len(tensors),
            "data_offset": int(reader.data_offset),
        },
        "format": {
            "source_type": "Q2_0",
            "block_size": BLOCK_SIZE,
            "source_block_bytes": SOURCE_BLOCK_BYTES,
            "source_scale_bytes": 2,
            "source_quant_bytes": 32,
            "codes": {"0": -1, "1": 0, "2": 1, "3": 2},
            "candidate_type": "TQ1_G128",
            "candidate_block_bytes": BLOCK_BYTES,
            "candidate_scale_bytes": 2,
            "candidate_trit_bytes": 26,
        },
        "type_summary": type_summary,
        "q2": {
            "tensor_count": len(q2_tensors),
            "elements": q2_elements,
            "groups": q2_groups,
            "source_bytes": q2_bytes,
            "projected_tq1_bytes": projected_q2_bytes,
            "payload_reduction_fraction": q2_reduction,
            "distribution": summarize_symbols(total_symbols, total_nonzero_histogram),
            "nonzero_group_histogram": total_nonzero_histogram.tolist(),
        },
        "whole_model_projection": {
            "unchanged_tensor_bytes": unchanged_tensor_bytes,
            "gguf_header_and_alignment_bytes": gguf_header_and_alignment,
            "projected_file_bytes": projected_file_bytes,
            "projected_file_gb_decimal": projected_file_bytes / 1e9,
            "projected_file_gib": projected_file_bytes / 2**30,
            "reduction_fraction": whole_reduction,
        },
        "gate_1": {
            "source_alphabet_is_ternary": not has_fourth_symbol,
            "q2_payload_reduction_at_least_15_percent": q2_reduction >= 0.15,
            "projected_file_at_most_6_1_gb": projected_file_bytes <= 6_100_000_000,
            "pass": packing_gate_pass,
            "verdict_if_stopped": None if packing_gate_pass else "FAIL_PACKING_NOT_MATERIAL",
        },
        "tensors": tensor_results,
        "elapsed_seconds": time.time() - started,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exhaustively inspect the Bonsai Q2_0 GGUF")
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-groups", type=int, default=DEFAULT_CHUNK_GROUPS)
    args = parser.parse_args(argv)
    try:
        result = inspect_model(args.model, chunk_groups=args.chunk_groups)
        write_json_atomic(args.output, result)
    except Exception as exc:
        print(f"inspection failed: {exc}", file=sys.stderr)
        return 2
    distribution = result["q2"]["distribution"]
    print(json.dumps({"gate_1": result["gate_1"], "distribution": distribution}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

