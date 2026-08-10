from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import numpy.typing as npt

BLOCK_SIZE = 128
SOURCE_BLOCK_BYTES = 34
BLOCK_BYTES = 28
TRIT_BYTES = 26
# Largest legal code bytes: 3**5 - 1 for a regular slot's five trits, 3**3 - 1
# for the tail slot's three. Anything above is unrepresentable, and the LUT23
# kernels index their tables without a bounds check, so these are load-bearing.
MAX_FULL_CODE_BYTE = 3**5 - 1
MAX_TAIL_CODE_BYTE = 3**3 - 1
ALIGNMENT = 256
HEADER_BYTES = 256
MAGIC = b"TQ1G128\0"
VERSION = 1
POWERS5 = np.asarray([1, 3, 9, 27, 81], dtype=np.uint16)
POWERS3 = np.asarray([1, 3, 9], dtype=np.uint16)
Q2_SHIFTS = np.asarray([0, 2, 4, 6], dtype=np.uint8)
HEADER_STRUCT = struct.Struct("<8sIIIIIIQQQ32s")


class FormatError(ValueError):
    pass


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def decode_q2_blocks(blocks: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Decode Prism Q2_0/g128 blocks to raw scale bytes and logical codes 0..3."""
    source = np.asarray(blocks, dtype=np.uint8)
    if source.ndim == 1:
        if source.size % SOURCE_BLOCK_BYTES:
            raise FormatError("Q2 byte length is not divisible by 34")
        source = source.reshape(-1, SOURCE_BLOCK_BYTES)
    if source.ndim != 2 or source.shape[1] != SOURCE_BLOCK_BYTES:
        raise FormatError(f"expected Q2 blocks shaped (N, 34), got {source.shape}")
    scales = source[:, :2].copy()
    quant_bytes = source[:, 2:]
    codes = ((quant_bytes[:, :, None] >> Q2_SHIFTS) & 0x03).reshape(-1, BLOCK_SIZE)
    return scales, codes.astype(np.uint8, copy=False)


def encode_tq1_blocks(scales: npt.ArrayLike, codes: npt.ArrayLike) -> np.ndarray:
    """Encode ternary logical codes 0/1/2 as 28-byte TQ1_G128 blocks."""
    scale_array = np.asarray(scales, dtype=np.uint8)
    code_array = np.asarray(codes, dtype=np.uint8)
    if scale_array.ndim == 1:
        scale_array = scale_array.reshape(-1, 2)
    if code_array.ndim == 1:
        code_array = code_array.reshape(-1, BLOCK_SIZE)
    if scale_array.ndim != 2 or scale_array.shape[1] != 2:
        raise FormatError(f"expected scale bytes shaped (N, 2), got {scale_array.shape}")
    if code_array.ndim != 2 or code_array.shape[1] != BLOCK_SIZE:
        raise FormatError(f"expected codes shaped (N, 128), got {code_array.shape}")
    if scale_array.shape[0] != code_array.shape[0]:
        raise FormatError("scale and code block counts differ")
    invalid = np.count_nonzero(code_array > 2)
    if invalid:
        raise FormatError(f"source contains {invalid} non-ternary code(s)")

    count = code_array.shape[0]
    result = np.empty((count, BLOCK_BYTES), dtype=np.uint8)
    result[:, :2] = scale_array
    groups = code_array[:, :125].reshape(count, 25, 5).astype(np.uint16, copy=False)
    result[:, 2:27] = np.sum(groups * POWERS5, axis=2, dtype=np.uint16).astype(np.uint8)
    tail = code_array[:, 125:].astype(np.uint16, copy=False)
    result[:, 27] = np.sum(tail * POWERS3, axis=1, dtype=np.uint16).astype(np.uint8)
    return result


def decode_tq1_blocks(blocks: npt.ArrayLike, *, validate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Independently decode 28-byte TQ1_G128 blocks."""
    packed = np.asarray(blocks, dtype=np.uint8)
    if packed.ndim == 1:
        if packed.size % BLOCK_BYTES:
            raise FormatError("TQ1 byte length is not divisible by 28")
        packed = packed.reshape(-1, BLOCK_BYTES)
    if packed.ndim != 2 or packed.shape[1] != BLOCK_BYTES:
        raise FormatError(f"expected TQ1 blocks shaped (N, 28), got {packed.shape}")
    payload = packed[:, 2:]
    if validate:
        invalid_full = int(np.count_nonzero(payload[:, :25] > MAX_FULL_CODE_BYTE))
        invalid_tail = int(np.count_nonzero(payload[:, 25] > MAX_TAIL_CODE_BYTE))
        if invalid_full or invalid_tail:
            raise FormatError(
                f"invalid base-3 bytes: regular={invalid_full}, tail={invalid_tail}"
            )

    count = packed.shape[0]
    codes = np.empty((count, BLOCK_SIZE), dtype=np.uint8)
    full = payload[:, :25].astype(np.uint16, copy=False)
    decoded = np.empty((count, 25, 5), dtype=np.uint8)
    for index, power in enumerate((1, 3, 9, 27, 81)):
        decoded[:, :, index] = ((full // power) % 3).astype(np.uint8)
    codes[:, :125] = decoded.reshape(count, 125)
    tail = payload[:, 25].astype(np.uint16, copy=False)
    for index, power in enumerate((1, 3, 9)):
        codes[:, 125 + index] = ((tail // power) % 3).astype(np.uint8)
    return packed[:, :2].copy(), codes


def iter_memmap_blocks(
    path: Path,
    *,
    offset: int,
    block_count: int,
    block_bytes: int,
    chunk_groups: int,
) -> Iterator[np.ndarray]:
    if offset < 0 or block_count < 0 or block_bytes <= 0 or chunk_groups <= 0:
        raise ValueError("invalid block mapping parameters")
    raw = np.memmap(
        path,
        mode="r",
        dtype=np.uint8,
        offset=offset,
        shape=(block_count, block_bytes),
    )
    try:
        for start in range(0, block_count, chunk_groups):
            yield np.asarray(raw[start : min(start + chunk_groups, block_count)])
    finally:
        del raw


@dataclass(frozen=True)
class SidecarHeader:
    version: int
    header_bytes: int
    block_size: int
    block_bytes: int
    alignment: int
    tensor_count: int
    source_file_bytes: int
    manifest_offset: int
    manifest_bytes: int
    source_sha256: str

    def pack(self) -> bytes:
        if len(self.source_sha256) != 64:
            raise FormatError("source SHA-256 must contain 64 hexadecimal characters")
        try:
            source_hash = bytes.fromhex(self.source_sha256)
        except ValueError as exc:
            raise FormatError("source SHA-256 is not hexadecimal") from exc
        prefix = HEADER_STRUCT.pack(
            MAGIC,
            self.version,
            self.header_bytes,
            self.block_size,
            self.block_bytes,
            self.alignment,
            self.tensor_count,
            self.source_file_bytes,
            self.manifest_offset,
            self.manifest_bytes,
            source_hash,
        )
        if len(prefix) > HEADER_BYTES:
            raise AssertionError("header struct exceeds fixed header")
        return prefix + bytes(HEADER_BYTES - len(prefix))

    @classmethod
    def unpack(cls, data: bytes) -> "SidecarHeader":
        if len(data) < HEADER_BYTES:
            raise FormatError("truncated TQ1 header")
        if any(data[HEADER_STRUCT.size:HEADER_BYTES]):
            raise FormatError("nonzero reserved TQ1 header bytes")
        fields = HEADER_STRUCT.unpack_from(data)
        if fields[0] != MAGIC:
            raise FormatError("invalid TQ1 magic")
        header = cls(
            version=fields[1],
            header_bytes=fields[2],
            block_size=fields[3],
            block_bytes=fields[4],
            alignment=fields[5],
            tensor_count=fields[6],
            source_file_bytes=fields[7],
            manifest_offset=fields[8],
            manifest_bytes=fields[9],
            source_sha256=fields[10].hex(),
        )
        if header.version != VERSION:
            raise FormatError(f"unsupported TQ1 version {header.version}")
        if (header.header_bytes, header.block_size, header.block_bytes, header.alignment) != (
            HEADER_BYTES,
            BLOCK_SIZE,
            BLOCK_BYTES,
            ALIGNMENT,
        ):
            raise FormatError("unsupported TQ1 geometry")
        return header


def read_sidecar(path: Path) -> tuple[SidecarHeader, dict]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = SidecarHeader.unpack(handle.read(HEADER_BYTES))
        end = header.manifest_offset + header.manifest_bytes
        if header.manifest_offset < HEADER_BYTES or end > file_size:
            raise FormatError("manifest points outside the sidecar")
        handle.seek(header.manifest_offset)
        try:
            manifest = json.loads(handle.read(header.manifest_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatError("invalid sidecar manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != "TQ1_G128":
        raise FormatError("unexpected sidecar manifest format")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != header.tensor_count:
        raise FormatError("sidecar tensor count does not match manifest")
    previous_end = HEADER_BYTES
    for tensor in tensors:
        offset = int(tensor["packed_offset"])
        size = int(tensor["packed_bytes"])
        groups = int(tensor["groups"])
        if offset % ALIGNMENT or offset < align_up(previous_end) or size != groups * BLOCK_BYTES:
            raise FormatError(f"invalid tensor extent for {tensor.get('name', '<unknown>')}")
        if offset + size > header.manifest_offset:
            raise FormatError("tensor data overlaps the manifest")
        previous_end = offset + size
    return header, manifest


def write_padding(handle: BinaryIO, target_offset: int) -> None:
    current = handle.tell()
    if target_offset < current:
        raise ValueError("cannot pad backwards")
    remaining = target_offset - current
    zeroes = bytes(min(1024 * 1024, max(remaining, 1)))
    while remaining:
        amount = min(remaining, len(zeroes))
        handle.write(zeroes[:amount])
        remaining -= amount


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def fsync_file(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())
