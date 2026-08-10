from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# The distribution patch extends the pinned Prism gguf-py enum with type 43.
# Prefer that exact local source without changing the project's uv pin.
_WORKSPACE = Path(__file__).resolve().parents[2]
_LOCAL_GGUF_PY = _WORKSPACE / "artifacts" / "src" / "llama.cpp" / "gguf-py"
if _LOCAL_GGUF_PY.is_dir():
    sys.path.insert(0, str(_LOCAL_GGUF_PY))

import gguf  # noqa: E402

from .constants import MODEL_REVISION, MODEL_SHA256  # noqa: E402
from .format import BLOCK_BYTES, BLOCK_SIZE, read_sidecar, sha256_file  # noqa: E402
from .inspect_model import write_json_atomic  # noqa: E402


TYPE_NAME = "TQ1_G128"
TYPE_ID = 43


def _copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)

    writer.add_string("bonsai.tq1_g128.format", TYPE_NAME)
    writer.add_uint32("bonsai.tq1_g128.version", 1)
    writer.add_uint32("bonsai.tq1_g128.block_size", BLOCK_SIZE)
    writer.add_uint32("bonsai.tq1_g128.block_bytes", BLOCK_BYTES)
    writer.add_uint32("bonsai.tq1_g128.cuda_m_tile", 256)
    writer.add_string("bonsai.tq1_g128.source_sha256", MODEL_SHA256)
    writer.add_string("bonsai.tq1_g128.source_revision", MODEL_REVISION)
    writer.add_string("bonsai.tq1_g128.runtime", "Prism llama.cpp + TQ1_LUT23_SMEM_STREAMK")


def _arch(reader: gguf.GGUFReader) -> str:
    field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if field is None:
        raise ValueError("source GGUF has no general.architecture")
    value = field.contents()
    if not isinstance(value, str):
        raise ValueError(f"unexpected architecture value {value!r}")
    return value


def convert_to_tq1_gguf(
    source_path: Path,
    sidecar_path: Path,
    output_path: Path,
    *,
    verify_source_hash: bool = True,
) -> dict[str, Any]:
    started = time.time()
    if output_path.resolve() in {source_path.resolve(), sidecar_path.resolve()}:
        raise ValueError("output must differ from source and sidecar")
    if verify_source_hash:
        digest = sha256_file(source_path)
        if digest != MODEL_SHA256:
            raise ValueError(f"source SHA-256 mismatch: {digest}")
    else:
        digest = sha256_file(source_path)

    sidecar_header, sidecar_manifest = read_sidecar(sidecar_path)
    if sidecar_header.source_sha256 != digest:
        raise ValueError("sidecar source identity does not match GGUF")
    packed_by_name = {entry["name"]: entry for entry in sidecar_manifest["tensors"]}

    reader = gguf.GGUFReader(source_path, "r")
    custom_type = gguf.GGMLQuantizationType(TYPE_ID)
    writer_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if writer_path.exists():
        writer_path.unlink()

    writer = gguf.GGUFWriter(writer_path, arch=_arch(reader), endianess=reader.endianess)
    alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        writer.data_alignment = int(alignment_field.contents())
    _copy_metadata(reader, writer)

    descriptors: list[tuple[Any, dict[str, Any] | None]] = []
    q2_count = 0
    packed_bytes = 0
    for tensor in reader.tensors:
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q2_0:
            entry = packed_by_name.get(tensor.name)
            if entry is None:
                raise ValueError(f"sidecar is missing Q2 tensor {tensor.name}")
            columns = int(tensor.shape[0])
            rows = int(tensor.shape[1])
            expected_bytes = rows * (columns // BLOCK_SIZE) * BLOCK_BYTES
            if int(entry["packed_bytes"]) != expected_bytes:
                raise ValueError(f"packed extent mismatch for {tensor.name}")
            byte_shape = (rows, columns // BLOCK_SIZE * BLOCK_BYTES)
            writer.add_tensor_info(
                tensor.name,
                byte_shape,
                np.dtype(np.uint8),
                expected_bytes,
                raw_dtype=custom_type,
            )
            descriptors.append((tensor, entry))
            q2_count += 1
            packed_bytes += expected_bytes
        else:
            writer.add_tensor_info(
                tensor.name,
                tensor.data.shape,
                tensor.data.dtype,
                tensor.data.nbytes,
                raw_dtype=tensor.tensor_type,
            )
            descriptors.append((tensor, None))

    if q2_count != sidecar_header.tensor_count or q2_count != len(packed_by_name):
        raise ValueError("source/sidecar Q2 tensor counts differ")

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        progress = tqdm(descriptors, desc="Writing distributable TQ1 GGUF", unit="tensor")
        for tensor, entry in progress:
            progress.set_postfix_str(tensor.name[:45])
            if entry is None:
                writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
                continue
            columns = int(tensor.shape[0])
            rows = int(tensor.shape[1])
            packed = np.memmap(
                sidecar_path,
                mode="r",
                dtype=np.uint8,
                offset=int(entry["packed_offset"]),
                shape=(rows, columns // BLOCK_SIZE * BLOCK_BYTES),
            )
            try:
                writer.write_tensor_data(packed, tensor_endianess=reader.endianess)
            finally:
                del packed
        writer.close()
        os.replace(writer_path, output_path)
    except BaseException:
        try:
            writer.close()
        finally:
            if writer_path.exists():
                writer_path.unlink()
        raise

    output_hash = sha256_file(output_path)
    result = {
        "schema_version": 1,
        "format": TYPE_NAME,
        "ggml_type_id": TYPE_ID,
        "source": str(source_path),
        "source_sha256": digest,
        "sidecar": str(sidecar_path),
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": output_hash,
        "tensor_count": len(descriptors),
        "converted_tensor_count": q2_count,
        "packed_payload_bytes": packed_bytes,
        "elapsed_seconds": time.time() - started,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a single user-loadable TQ1_G128 GGUF from the canonical sidecar"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--skip-source-hash", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = convert_to_tq1_gguf(
            args.source,
            args.sidecar,
            args.output,
            verify_source_hash=not args.skip_source_hash,
        )
        if args.result:
            write_json_atomic(args.result, result)
    except Exception as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

