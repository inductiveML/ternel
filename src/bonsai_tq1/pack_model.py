from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from .constants import MODEL_REPO, MODEL_REVISION, MODEL_SHA256
from .format import (
    ALIGNMENT,
    BLOCK_BYTES,
    BLOCK_SIZE,
    HEADER_BYTES,
    SOURCE_BLOCK_BYTES,
    VERSION,
    SidecarHeader,
    align_up,
    canonical_json_bytes,
    decode_q2_blocks,
    encode_tq1_blocks,
    fsync_file,
    iter_memmap_blocks,
    sha256_file,
    write_padding,
)
from .gguf_utils import load_tensor_infos
from .inspect_model import DEFAULT_CHUNK_GROUPS, write_json_atomic


def pack_model(
    model_path: Path,
    audit_path: Path,
    output_path: Path,
    *,
    chunk_groups: int = DEFAULT_CHUNK_GROUPS,
) -> dict:
    started = time.time()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["gate_1"]["pass"]:
        raise ValueError("Gate 1 did not pass; refusing to write a TQ1 sidecar")
    source_hash = sha256_file(model_path)
    if source_hash != MODEL_SHA256:
        raise ValueError(f"source SHA-256 mismatch: {source_hash}")
    reader, tensors = load_tensor_infos(model_path)
    del reader
    q2_tensors = [tensor for tensor in tensors if tensor.tensor_type == "Q2_0"]
    audit_tensors = {tensor["name"]: tensor for tensor in audit["tensors"]}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    manifest_tensors: list[dict] = []
    total_packed_payload = 0
    try:
        with temporary.open("w+b") as handle:
            handle.write(bytes(HEADER_BYTES))
            progress = tqdm(q2_tensors, desc="Packing Q2_0 tensors", unit="tensor")
            for tensor in progress:
                progress.set_postfix_str(tensor.name[:42])
                packed_offset = align_up(handle.tell(), ALIGNMENT)
                write_padding(handle, packed_offset)
                logical_hash = hashlib.sha256()
                scale_hash = hashlib.sha256()
                packed_hash = hashlib.sha256()
                groups_seen = 0
                for blocks in iter_memmap_blocks(
                    model_path,
                    offset=tensor.data_offset,
                    block_count=tensor.groups,
                    block_bytes=SOURCE_BLOCK_BYTES,
                    chunk_groups=chunk_groups,
                ):
                    scales, codes = decode_q2_blocks(blocks)
                    packed = encode_tq1_blocks(scales, codes)
                    packed_bytes = packed.tobytes(order="C")
                    handle.write(packed_bytes)
                    logical_hash.update(codes.tobytes(order="C"))
                    scale_hash.update(scales.tobytes(order="C"))
                    packed_hash.update(packed_bytes)
                    groups_seen += codes.shape[0]
                if groups_seen != tensor.groups:
                    raise AssertionError(f"group conversion mismatch for {tensor.name}")
                logical_digest = logical_hash.hexdigest()
                scale_digest = scale_hash.hexdigest()
                audited = audit_tensors[tensor.name]
                if logical_digest != audited["logical_weight_sha256"]:
                    raise ValueError(f"logical source hash changed for {tensor.name}")
                if scale_digest != audited["scale_bits_sha256"]:
                    raise ValueError(f"scale source hash changed for {tensor.name}")
                packed_bytes_count = tensor.groups * BLOCK_BYTES
                total_packed_payload += packed_bytes_count
                manifest_tensors.append(
                    {
                        "index": tensor.index,
                        "name": tensor.name,
                        "shape_gguf": list(tensor.shape),
                        "elements": tensor.elements,
                        "groups": tensor.groups,
                        "source_type": "Q2_0",
                        "source_offset": tensor.data_offset,
                        "source_bytes": tensor.data_bytes,
                        "packed_offset": packed_offset,
                        "packed_bytes": packed_bytes_count,
                        "logical_weight_sha256": logical_digest,
                        "scale_bits_sha256": scale_digest,
                        "packed_payload_sha256": packed_hash.hexdigest(),
                    }
                )

            manifest_offset = handle.tell()
            manifest = {
                "format": "TQ1_G128",
                "version": VERSION,
                "block_size": BLOCK_SIZE,
                "block_bytes": BLOCK_BYTES,
                "alignment": ALIGNMENT,
                "mapping": {"-1": 0, "0": 1, "+1": 2},
                "source": {
                    "repo": MODEL_REPO,
                    "revision": MODEL_REVISION,
                    "filename": model_path.name,
                    "file_bytes": model_path.stat().st_size,
                    "sha256": source_hash,
                },
                "packed_payload_bytes": total_packed_payload,
                "tensors": manifest_tensors,
            }
            manifest_bytes = canonical_json_bytes(manifest)
            handle.write(manifest_bytes)
            header = SidecarHeader(
                version=VERSION,
                header_bytes=HEADER_BYTES,
                block_size=BLOCK_SIZE,
                block_bytes=BLOCK_BYTES,
                alignment=ALIGNMENT,
                tensor_count=len(manifest_tensors),
                source_file_bytes=model_path.stat().st_size,
                manifest_offset=manifest_offset,
                manifest_bytes=len(manifest_bytes),
                source_sha256=source_hash,
            )
            handle.seek(0)
            handle.write(header.pack())
            fsync_file(handle)
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    result = {
        "schema_version": 1,
        "source_sha256": source_hash,
        "sidecar": str(output_path),
        "sidecar_sha256": sha256_file(output_path),
        "sidecar_file_bytes": output_path.stat().st_size,
        "packed_payload_bytes": total_packed_payload,
        "tensor_count": len(manifest_tensors),
        "groups": sum(tensor["groups"] for tensor in manifest_tensors),
        "weights": sum(tensor["elements"] for tensor in manifest_tensors),
        "manifest_offset": manifest_offset,
        "manifest_bytes": len(manifest_bytes),
        "elapsed_seconds": time.time() - started,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Losslessly pack Bonsai Q2_0/g128 tensors")
    parser.add_argument("model", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--chunk-groups", type=int, default=DEFAULT_CHUNK_GROUPS)
    args = parser.parse_args(argv)
    try:
        result = pack_model(
            args.model,
            args.audit,
            args.output,
            chunk_groups=args.chunk_groups,
        )
        write_json_atomic(args.result, result)
    except Exception as exc:
        print(f"packing failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

