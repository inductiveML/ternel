"""Read individual tensors out of a safetensors file, remote or local.

The official MLX distribution of Ternary Bonsai 27B is a single 8.49 GB shard.
Auditing a handful of tensors does not justify downloading it, so
:class:`RemoteSafetensors` reads the header and then only the byte ranges of the
tensors that are actually requested.

Once a shard is on disk the same header is worth reading the same way, so
:class:`LocalSafetensors` shares the parser and swaps the transport for a
``numpy.memmap``. Keeping one parser means the reader used to *check* an
artifact cannot disagree with the reader used to *audit* the baseline, and it
keeps the verification path free of MLX -- a file written by MLX is poor
evidence about itself.
"""

from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

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


class SafetensorsFile:
    """The parsed header of a safetensors file, whatever it is stored on.

    A subclass supplies ``_read_range(begin, end_inclusive)`` and a ``read``
    that turns one tensor's extent into an array. Everything about the format
    itself -- the length prefix, the JSON header, the offset arithmetic, the
    shape/dtype consistency check -- is decided once, here.
    """

    def __init__(self, source: str, read_range: Callable[[int, int], bytes]) -> None:
        self.source = source
        prefix = read_range(0, SAFETENSORS_LENGTH_BYTES - 1)
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES:
            raise FormatError(f"implausible safetensors header length {header_bytes} in {source}")
        raw = read_range(SAFETENSORS_LENGTH_BYTES, SAFETENSORS_LENGTH_BYTES + header_bytes - 1)
        try:
            header = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatError(f"safetensors header of {source} is not valid JSON") from exc
        if not isinstance(header, dict):
            raise FormatError(f"safetensors header of {source} is not a JSON object")

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
            raise FormatError(f"safetensors header of {source} declares no tensors")
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
            raise FormatError(f"tensor {name!r} is not present in {self.source}")
        return found

    def _shaped(self, entry: TensorEntry, array: np.ndarray) -> np.ndarray:
        expected_elements = 1
        for dim in entry.shape:
            expected_elements *= dim
        if array.size != expected_elements:
            raise FormatError(
                f"tensor {entry.name} decoded to {array.size} elements, "
                f"expected {expected_elements}"
            )
        return array.reshape(entry.shape)

    def read(self, name: str) -> np.ndarray:
        raise NotImplementedError


class RemoteSafetensors(SafetensorsFile):
    """A remote safetensors file addressed by HTTP range requests."""

    def __init__(self, url: str, *, timeout_seconds: float) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        super().__init__(
            url,
            lambda begin, end: _http_get_range(url, begin, end, timeout_seconds=timeout_seconds),
        )

    def read(self, name: str) -> np.ndarray:
        """Fetch exactly one tensor's bytes and return it as a numpy array."""
        found = self.entry(name)
        payload = _http_get_range(
            self.url, found.begin, found.end - 1, timeout_seconds=self.timeout_seconds
        )
        return self._shaped(found, np.frombuffer(payload, dtype=found.numpy_dtype))


class LocalSafetensors(SafetensorsFile):
    """A safetensors file on disk, read through one shared ``numpy.memmap``.

    ``read`` returns a *view* into the mapping rather than a copy, so walking a
    27B-parameter shard costs page cache rather than resident memory. Callers
    that keep a result past the next tensor should copy it.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._map = np.memmap(self.path, dtype=np.uint8, mode="r")
        super().__init__(str(self.path), lambda begin, end: bytes(self._map[begin : end + 1]))

    @property
    def file_bytes(self) -> int:
        return int(self._map.size)

    def read(self, name: str) -> np.ndarray:
        found = self.entry(name)
        if found.end > self._map.size:
            raise FormatError(
                f"tensor {name} ends at byte {found.end} but {self.path} is "
                f"{self._map.size} bytes"
            )
        window = self._map[found.begin : found.end]
        return self._shaped(found, window.view(found.numpy_dtype))

    def read_rows(self, name: str, begin: int, end: int) -> np.ndarray:
        """A row slice of one tensor, without touching the rows outside it.

        The largest tensors here are 248320 rows wide, so a whole-tensor read is
        a gigabyte the comparison does not need all at once.
        """
        found = self.entry(name)
        if not found.shape:
            raise FormatError(f"tensor {name} is a scalar and has no rows")
        rows = found.shape[0]
        if begin < 0 or end > rows or begin > end:
            raise FormatError(f"row range [{begin}, {end}) is outside {name}'s {rows} rows")
        row_elements = 1
        for dim in found.shape[1:]:
            row_elements *= dim
        row_bytes = row_elements * found.numpy_dtype.itemsize
        window = self._map[found.begin + begin * row_bytes : found.begin + end * row_bytes]
        return window.view(found.numpy_dtype).reshape((end - begin, *found.shape[1:]))
