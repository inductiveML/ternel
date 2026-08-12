"""Write safetensors shards one tensor at a time.

``mx.save_safetensors`` wants the whole checkpoint as a dict of MLX arrays. For
a 5.9 GB artifact that means holding the model in memory to write it out, and it
means the file that Ternel must later verify was produced by the same library it
is being verified against. Both are avoidable.

A safetensors header states every tensor's byte range up front, so the writer
cannot discover offsets as it goes -- but it does not need to. The packed layout
fixes each tensor's size before a byte is read, so the shards are *planned*
first and then filled, and only one tensor is resident at a time.

The header parser in :mod:`ternel_mlx.remote_safetensors` is the reader for
everything written here, and its dtype table is the one used below, so a writer
and reader disagreement is not expressible.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from bonsai_tq1.format import FormatError

from .remote_safetensors import DTYPE_MAP, SAFETENSORS_LENGTH_BYTES

# safetensors pads the header with spaces so the data section starts aligned.
HEADER_ALIGNMENT = 8
HEADER_PAD_BYTE = b" "

DTYPE_TAGS: dict[np.dtype, str] = {dtype: tag for tag, dtype in DTYPE_MAP.items()}


@dataclass(frozen=True)
class PlannedTensor:
    """One tensor's identity and size, known before its bytes exist."""

    name: str
    dtype: np.dtype
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise FormatError("a tensor needs a name")
        if self.dtype not in DTYPE_TAGS:
            raise FormatError(f"{self.name} has dtype {self.dtype}, which safetensors cannot store")
        if any(dim < 0 for dim in self.shape):
            raise FormatError(f"{self.name} has a negative dimension in {self.shape}")

    @property
    def tag(self) -> str:
        return DTYPE_TAGS[self.dtype]

    @property
    def elements(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def nbytes(self) -> int:
        return self.elements * self.dtype.itemsize


def plan_shards(
    tensors: Sequence[PlannedTensor], *, max_shard_bytes: int
) -> list[list[PlannedTensor]]:
    """Split tensors into shards of at most ``max_shard_bytes``, in the given order.

    A tensor larger than the limit gets a shard of its own rather than being
    split: safetensors has no concept of a tensor spanning files.
    """
    if max_shard_bytes <= 0:
        raise FormatError(f"shard budget must be positive, got {max_shard_bytes}")
    if not tensors:
        raise FormatError("refusing to write a checkpoint with no tensors")
    seen: set[str] = set()
    shards: list[list[PlannedTensor]] = [[]]
    used = 0
    for tensor in tensors:
        if tensor.name in seen:
            raise FormatError(f"tensor {tensor.name} was planned twice")
        seen.add(tensor.name)
        if shards[-1] and used + tensor.nbytes > max_shard_bytes:
            shards.append([])
            used = 0
        shards[-1].append(tensor)
        used += tensor.nbytes
    return shards


def shard_name(index: int, count: int) -> str:
    """``model.safetensors`` alone, or the HF ``model-00001-of-00003`` form."""
    if count < 1 or not 0 <= index < count:
        raise FormatError(f"shard {index} is outside a set of {count}")
    if count == 1:
        return "model.safetensors"
    return f"model-{index + 1:05d}-of-{count:05d}.safetensors"


def header_bytes(tensors: Iterable[PlannedTensor], metadata: dict[str, str]) -> bytes:
    """The length prefix and padded JSON header for one shard."""
    header: dict[str, object] = {}
    if metadata:
        for key, value in metadata.items():
            if not isinstance(value, str):
                raise FormatError(f"safetensors metadata value for {key!r} must be a string")
        header["__metadata__"] = dict(metadata)
    offset = 0
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.tag,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    raw = json.dumps(header, separators=(",", ":"), sort_keys=False).encode("utf-8")
    padding = -len(raw) % HEADER_ALIGNMENT
    raw += HEADER_PAD_BYTE * padding
    return struct.pack("<Q", len(raw)) + raw


def write_shard(
    path: Path,
    tensors: Sequence[PlannedTensor],
    produce: Callable[[PlannedTensor], np.ndarray],
    *,
    metadata: dict[str, str],
) -> int:
    """Write one shard, calling ``produce`` for each tensor in turn.

    Written to a temporary sibling and renamed, so a shard is either complete or
    absent -- a half-written checkpoint that loads is worse than none.
    """
    temporary = path.with_name(path.name + ".tmp")
    written = 0
    try:
        with temporary.open("wb") as handle:
            handle.write(header_bytes(tensors, metadata))
            for tensor in tensors:
                array = np.ascontiguousarray(produce(tensor))
                if array.dtype != tensor.dtype:
                    raise FormatError(
                        f"{tensor.name} was produced as {array.dtype}, planned as {tensor.dtype}"
                    )
                if tuple(array.shape) != tensor.shape:
                    raise FormatError(
                        f"{tensor.name} was produced with shape {array.shape}, "
                        f"planned as {tensor.shape}"
                    )
                handle.write(array.tobytes(order="C"))
                written += tensor.nbytes
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return written


def write_index(
    path: Path, shards: Sequence[Sequence[PlannedTensor]], *, total_bytes: int
) -> dict[str, object]:
    """The Hugging Face ``model.safetensors.index.json`` map.

    ``mlx_lm.utils.load_model`` globs ``model*.safetensors`` and ignores this,
    but every other tool in the ecosystem reads it, and writing it costs a
    kilobyte.
    """
    weight_map = {
        tensor.name: shard_name(index, len(shards))
        for index, shard in enumerate(shards)
        for tensor in shard
    }
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def expected_file_bytes(tensors: Sequence[PlannedTensor], metadata: dict[str, str]) -> int:
    """What a shard holding exactly ``tensors`` must weigh on disk."""
    return len(header_bytes(tensors, metadata)) + sum(tensor.nbytes for tensor in tensors)


SAFETENSORS_PREFIX_BYTES = SAFETENSORS_LENGTH_BYTES
