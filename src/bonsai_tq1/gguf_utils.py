from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gguf import GGUFReader

from .format import BLOCK_SIZE, SOURCE_BLOCK_BYTES


@dataclass(frozen=True)
class TensorInfo:
    index: int
    name: str
    tensor_type: str
    shape: tuple[int, ...]
    elements: int
    data_offset: int
    data_bytes: int

    @property
    def groups(self) -> int:
        if self.tensor_type != "Q2_0":
            return 0
        if self.elements % BLOCK_SIZE:
            raise ValueError(f"{self.name} element count is not divisible by {BLOCK_SIZE}")
        result = self.elements // BLOCK_SIZE
        if self.data_bytes != result * SOURCE_BLOCK_BYTES:
            raise ValueError(f"{self.name} has unexpected Q2_0 byte count")
        return result


def load_tensor_infos(path: Path) -> tuple[GGUFReader, list[TensorInfo]]:
    reader = GGUFReader(path, "r")
    infos = [
        TensorInfo(
            index=index,
            name=tensor.name,
            tensor_type=tensor.tensor_type.name,
            shape=tuple(int(value) for value in tensor.shape),
            elements=int(tensor.n_elements),
            data_offset=int(tensor.data_offset),
            data_bytes=int(tensor.n_bytes),
        )
        for index, tensor in enumerate(reader.tensors)
    ]
    return reader, infos


def summarize_types(tensors: Iterable[TensorInfo]) -> dict[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    elements: Counter[str] = Counter()
    data_bytes: Counter[str] = Counter()
    for tensor in tensors:
        counts[tensor.tensor_type] += 1
        elements[tensor.tensor_type] += tensor.elements
        data_bytes[tensor.tensor_type] += tensor.data_bytes
    return {
        name: {
            "tensor_count": counts[name],
            "elements": elements[name],
            "data_bytes": data_bytes[name],
        }
        for name in sorted(counts)
    }

