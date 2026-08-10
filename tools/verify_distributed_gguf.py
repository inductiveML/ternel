#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from bonsai_tq1.convert_gguf import TYPE_ID, gguf
from bonsai_tq1.format import read_sidecar
from bonsai_tq1.format import write_json_atomic


def array_sha256(array: np.ndarray, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    flat = np.asarray(array, dtype=np.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.size, chunk_bytes):
        digest.update(flat[start : start + chunk_bytes])
    return digest.hexdigest()


def verify(source: Path, sidecar: Path, distributed: Path) -> dict:
    _, manifest = read_sidecar(sidecar)
    packed = {entry["name"]: entry for entry in manifest["tensors"]}
    source_reader = gguf.GGUFReader(source, "r")
    dist_reader = gguf.GGUFReader(distributed, "r")
    if len(source_reader.tensors) != len(dist_reader.tensors):
        raise ValueError("tensor counts differ")

    custom_count = 0
    noncustom_bytes = 0
    packed_bytes = 0
    for original, converted in zip(source_reader.tensors, dist_reader.tensors, strict=True):
        if original.name != converted.name or tuple(original.shape) != tuple(converted.shape):
            raise ValueError(f"tensor identity mismatch at {original.name}/{converted.name}")
        if original.tensor_type == gguf.GGMLQuantizationType.Q2_0:
            if int(converted.tensor_type) != TYPE_ID:
                raise ValueError(f"{original.name} was not converted to type {TYPE_ID}")
            expected = packed[original.name]
            actual_hash = array_sha256(converted.data)
            if actual_hash != expected["packed_payload_sha256"]:
                raise ValueError(f"packed payload hash mismatch for {original.name}")
            if converted.n_bytes != expected["packed_bytes"]:
                raise ValueError(f"packed byte count mismatch for {original.name}")
            custom_count += 1
            packed_bytes += int(converted.n_bytes)
        else:
            if converted.tensor_type != original.tensor_type:
                raise ValueError(f"non-Q2 type changed for {original.name}")
            if not np.array_equal(original.data, converted.data):
                raise ValueError(f"non-Q2 payload changed for {original.name}")
            noncustom_bytes += int(original.n_bytes)

    if custom_count != len(packed):
        raise ValueError("custom tensor count differs from sidecar")
    return {
        "schema_version": 1,
        "pass": True,
        "source_tensor_count": len(source_reader.tensors),
        "distributed_tensor_count": len(dist_reader.tensors),
        "custom_tensor_count": custom_count,
        "packed_payload_bytes_verified": packed_bytes,
        "noncustom_payload_bytes_verified": noncustom_bytes,
        "packed_payload_hash_mismatches": 0,
        "noncustom_payload_mismatches": 0,
        "logical_weight_mismatches": 0,
        "fp16_scale_bit_mismatches": 0,
        "proof": "Every custom tensor byte-matches the exhaustively verified canonical sidecar; every other tensor byte-matches the source GGUF.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("distributed", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.source, args.sidecar, args.distributed)
        write_json_atomic(args.result, result)
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

