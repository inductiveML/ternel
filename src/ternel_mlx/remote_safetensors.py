"""Range-read individual tensors out of a remote safetensors file.

The official MLX distribution of Ternary Bonsai 27B is a single 8.49 GB shard.
Auditing a handful of tensors does not justify downloading it, so this module
reads the safetensors header and then only the byte ranges of the tensors that
are actually requested.
"""

from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

import ml_dtypes
import numpy as np

from bonsai_tq1.format import FormatError

SAFETENSORS_LENGTH_BYTES = 8
MAX_HEADER_BYTES = 100 * 1024 * 1024

# safetensors dtype tag -> numpy dtype. Explicitly little-endian: safetensors
# is defined as little-endian regardless of host byte order.
DTYPE_MAP: dict[str, np.dtype] = {
    "BOOL": np.dtype("?"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "U16": np.dtype("<u2"),
    "I16": np.dtype("<i2"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "U64": np.dtype("<u8"),
    "I64": np.dtype("<i8"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "BF16": np.dtype(ml_dtypes.bfloat16),
}


@dataclass(frozen=True)
class TensorEntry:
    """One tensor's location inside the remote file, in absolute file offsets."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numpy_dtype(self) -> np.dtype:
        mapped = DTYPE_MAP.get(self.dtype)
        if mapped is None:
            raise FormatError(f"unsupported safetensors dtype {self.dtype!r} for {self.name}")
        return mapped

    def validate(self) -> None:
        if self.begin < 0 or self.end < self.begin:
            raise FormatError(f"tensor {self.name} has a negative or inverted extent")
        count = 1
        for dim in self.shape:
            if dim < 0:
                raise FormatError(f"tensor {self.name} has a negative dimension")
            count *= dim
        expected = count * self.numpy_dtype.itemsize
        if expected != self.nbytes:
            raise FormatError(
                f"tensor {self.name} declares {self.nbytes} bytes but shape/dtype imply {expected}"
            )


def _http_get_range(url: str, begin: int, end_inclusive: int, *, timeout_seconds: float) -> bytes:
    if begin < 0 or end_inclusive < begin:
        raise ValueError("invalid byte range")
    request = urllib.request.Request(url, headers={"Range": f"bytes={begin}-{end_inclusive}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status != 206:
                raise FormatError(
                    f"server ignored the range request for {url} (status {response.status}); "
                    "refusing to read a full 8 GB shard"
                )
            payload = response.read()
    except urllib.error.URLError as exc:
        raise FormatError(f"range request to {url} failed: {exc}") from exc
    expected = end_inclusive - begin + 1
    if len(payload) != expected:
        raise FormatError(f"range request returned {len(payload)} bytes, expected {expected}")
    return payload


class RemoteSafetensors:
    """A remote safetensors file addressed by HTTP range requests."""

    def __init__(self, url: str, *, timeout_seconds: float) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        prefix = _http_get_range(url, 0, SAFETENSORS_LENGTH_BYTES - 1, timeout_seconds=timeout_seconds)
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES:
            raise FormatError(f"implausible safetensors header length {header_bytes}")
        raw = _http_get_range(
            url,
            SAFETENSORS_LENGTH_BYTES,
            SAFETENSORS_LENGTH_BYTES + header_bytes - 1,
            timeout_seconds=timeout_seconds,
        )
        try:
            header = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatError("remote safetensors header is not valid JSON") from exc
        if not isinstance(header, dict):
            raise FormatError("remote safetensors header is not a JSON object")

        self.data_start = SAFETENSORS_LENGTH_BYTES + header_bytes
        self.metadata = header.get("__metadata__")
        entries: dict[str, TensorEntry] = {}
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(value, dict):
                raise FormatError(f"header entry {name} is not an object")
            offsets = value["data_offsets"]
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise FormatError(f"header entry {name} has malformed data_offsets")
            entry = TensorEntry(
                name=name,
                dtype=str(value["dtype"]),
                shape=tuple(int(dim) for dim in value["shape"]),
                begin=self.data_start + int(offsets[0]),
                end=self.data_start + int(offsets[1]),
            )
            entry.validate()
            entries[name] = entry
        if not entries:
            raise FormatError("remote safetensors header declares no tensors")
        self.entries = entries

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> Iterator[str]:
        return iter(sorted(self.entries))

    def entry(self, name: str) -> TensorEntry:
        found = self.entries.get(name)
        if found is None:
            raise FormatError(f"tensor {name!r} is not present in the remote file")
        return found

    def read(self, name: str) -> np.ndarray:
        """Fetch exactly one tensor's bytes and return it as a numpy array."""
        found = self.entry(name)
        payload = _http_get_range(
            self.url, found.begin, found.end - 1, timeout_seconds=self.timeout_seconds
        )
        array = np.frombuffer(payload, dtype=found.numpy_dtype)
        expected_elements = 1
        for dim in found.shape:
            expected_elements *= dim
        if array.size != expected_elements:
            raise FormatError(
                f"tensor {name} decoded to {array.size} elements, expected {expected_elements}"
            )
        return array.reshape(found.shape)
