from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .format import (
    ALIGNMENT,
    BLOCK_BYTES,
    HEADER_BYTES,
    canonical_json_bytes,
    fsync_file,
    read_sidecar,
    sha256_file,
)
from .inspect_model import write_json_atomic


MAGIC = b"TQ1L23\0\0"
VERSION = 1
M_TILE = 256
CODE_SLOTS = 26
FULL_CODE_SLOTS = 25
HEADER_STRUCT = struct.Struct("<8sIIIIIIIQQQ32s")


class Lut23FormatError(ValueError):
    pass


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True)
class Lut23Header:
    tensor_count: int
    source_file_bytes: int
    manifest_offset: int
    manifest_bytes: int
    source_sha256: str

    def pack(self) -> bytes:
        digest = bytes.fromhex(self.source_sha256)
        if len(digest) != 32:
            raise Lut23FormatError("source SHA-256 must be 32 bytes")
        prefix = HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            HEADER_BYTES,
            M_TILE,
            CODE_SLOTS,
            BLOCK_BYTES,
            ALIGNMENT,
            self.tensor_count,
            self.source_file_bytes,
            self.manifest_offset,
            self.manifest_bytes,
            digest,
        )
        if len(prefix) > HEADER_BYTES:
            raise AssertionError("LUT23 header exceeds 256 bytes")
        return prefix + bytes(HEADER_BYTES - len(prefix))

    @classmethod
    def unpack(cls, raw: bytes) -> "Lut23Header":
        if len(raw) != HEADER_BYTES:
            raise Lut23FormatError("truncated LUT23 header")
        fields = HEADER_STRUCT.unpack_from(raw)
        if fields[0] != MAGIC:
            raise Lut23FormatError("invalid LUT23 magic")
        if fields[1:7] != (
            VERSION,
            HEADER_BYTES,
            M_TILE,
            CODE_SLOTS,
            BLOCK_BYTES,
            ALIGNMENT,
        ):
            raise Lut23FormatError("unsupported LUT23 geometry")
        if any(raw[HEADER_STRUCT.size:]):
            raise Lut23FormatError("nonzero LUT23 reserved header bytes")
        return cls(
            tensor_count=fields[7],
            source_file_bytes=fields[8],
            manifest_offset=fields[9],
            manifest_bytes=fields[10],
            source_sha256=fields[11].hex(),
        )


def tensor_layout(rows: int, groups: int, start: int) -> dict[str, int]:
    if rows <= 0 or groups <= 0:
        raise ValueError("tensor dimensions must be positive")
    tiles = (rows + M_TILE - 1) // M_TILE
    padded_rows = tiles * M_TILE
    codes_offset = align_up(start)
    codes_bytes = padded_rows * groups * CODE_SLOTS
    scales_offset = align_up(codes_offset + codes_bytes)
    scales_bytes = padded_rows * groups * 2
    return {
        "tiles": tiles,
        "padded_rows": padded_rows,
        "codes_offset": codes_offset,
        "codes_bytes": codes_bytes,
        "scales_offset": scales_offset,
        "scales_bytes": scales_bytes,
        "end": scales_offset + scales_bytes,
    }


def reorder_tensor_arrays(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(blocks, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != BLOCK_BYTES:
        raise ValueError("blocks must have shape (rows, groups, 28)")
    rows, groups, _ = source.shape
    tiles = (rows + M_TILE - 1) // M_TILE
    codes = np.zeros((tiles, groups, CODE_SLOTS, M_TILE), dtype=np.uint8)
    scales = np.zeros((tiles, groups, M_TILE, 2), dtype=np.uint8)
    for tile in range(tiles):
        begin = tile * M_TILE
        count = min(M_TILE, rows - begin)
        part = source[begin : begin + count]
        codes[tile, :, :, :count] = part[:, :, 2:].transpose(1, 2, 0)
        scales[tile, :, :count, :] = part[:, :, :2].transpose(1, 0, 2)
    return codes, scales


def restore_tensor_arrays(codes: np.ndarray, scales: np.ndarray, rows: int) -> np.ndarray:
    code_array = np.asarray(codes, dtype=np.uint8)
    scale_array = np.asarray(scales, dtype=np.uint8)
    if (
        code_array.ndim != 4
        or code_array.shape[2:] != (CODE_SLOTS, M_TILE)
        or scale_array.shape != (code_array.shape[0], code_array.shape[1], M_TILE, 2)
        or not 0 < rows <= code_array.shape[0] * M_TILE
    ):
        raise ValueError("invalid reordered tensor geometry")
    result = np.empty((rows, code_array.shape[1], BLOCK_BYTES), dtype=np.uint8)
    for tile in range(code_array.shape[0]):
        begin = tile * M_TILE
        count = min(M_TILE, rows - begin)
        if count <= 0:
            break
        result[begin : begin + count, :, 2:] = code_array[tile, :, :, :count].transpose(2, 0, 1)
        result[begin : begin + count, :, :2] = scale_array[tile, :, :count, :].transpose(1, 0, 2)
    return result


def _source_sha256(source: Path, packing_result: Path | None) -> str:
    if packing_result is None:
        return sha256_file(source)
    result = json.loads(packing_result.read_text(encoding="utf-8"))
    if source.stat().st_size != int(result["sidecar_file_bytes"]):
        raise Lut23FormatError("source sidecar byte count differs from packing record")
    digest = str(result["sidecar_sha256"])
    if len(digest) != 64:
        raise Lut23FormatError("invalid source sidecar hash in packing record")
    return digest


def reorder_sidecar(
    source: Path,
    destination: Path,
    *,
    packing_result: Path | None = None,
) -> dict:
    started = time.time()
    source_header, source_manifest = read_sidecar(source)
    source_hash = _source_sha256(source, packing_result)
    layouts: list[dict] = []
    cursor = HEADER_BYTES
    for tensor in source_manifest["tensors"]:
        columns, rows = (int(value) for value in tensor["shape_gguf"])
        if columns % 128:
            raise Lut23FormatError(f"non-g128 tensor {tensor['name']}")
        groups_per_row = columns // 128
        if rows * groups_per_row != int(tensor["groups"]):
            raise Lut23FormatError(f"shape/group mismatch for {tensor['name']}")
        layout = tensor_layout(rows, groups_per_row, cursor)
        layout.update(
            {
                "index": int(tensor["index"]),
                "name": str(tensor["name"]),
                "rows": rows,
                "columns": columns,
                "groups_per_row": groups_per_row,
                "source_offset": int(tensor["packed_offset"]),
                "source_bytes": int(tensor["packed_bytes"]),
                "logical_weight_sha256": str(tensor["logical_weight_sha256"]),
                "scale_bits_sha256": str(tensor["scale_bits_sha256"]),
            }
        )
        layouts.append(layout)
        cursor = layout["end"]
    manifest_offset = align_up(cursor)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("w+b") as output:
            output.truncate(manifest_offset)
        with source.open("rb") as source_handle, temporary.open("r+b") as output:
            for number, (tensor, layout) in enumerate(
                zip(source_manifest["tensors"], layouts, strict=True), start=1
            ):
                rows = layout["rows"]
                groups = layout["groups_per_row"]
                blocks = np.memmap(
                    source_handle,
                    mode="r",
                    dtype=np.uint8,
                    offset=layout["source_offset"],
                    shape=(rows, groups, BLOCK_BYTES),
                )
                codes = np.memmap(
                    output,
                    mode="r+",
                    dtype=np.uint8,
                    offset=layout["codes_offset"],
                    shape=(layout["tiles"], groups, CODE_SLOTS, M_TILE),
                )
                scales = np.memmap(
                    output,
                    mode="r+",
                    dtype=np.uint8,
                    offset=layout["scales_offset"],
                    shape=(layout["tiles"], groups, M_TILE, 2),
                )
                for tile in range(layout["tiles"]):
                    begin = tile * M_TILE
                    count = min(M_TILE, rows - begin)
                    part = np.asarray(blocks[begin : begin + count])
                    codes[tile, :, :, :count] = part[:, :, 2:].transpose(1, 2, 0)
                    scales[tile, :, :count, :] = part[:, :, :2].transpose(1, 0, 2)
                codes.flush()
                scales.flush()
                del scales, codes, blocks
                print(f"reordered {number}/{len(layouts)} {tensor['name']}", flush=True)

            manifest = {
                "format": "TQ1_LUT23_CODE_POSITION_MAJOR",
                "version": VERSION,
                "m_tile": M_TILE,
                "code_slots": CODE_SLOTS,
                "block_size": 128,
                "block_bytes": BLOCK_BYTES,
                "alignment": ALIGNMENT,
                "source": {
                    "path": str(source),
                    "file_bytes": source.stat().st_size,
                    "sha256": source_hash,
                    "tq1_source_model_sha256": source_header.source_sha256,
                },
                "padding": {
                    "codes": "zero base-3 bytes",
                    "scales": "zero FP16 bits",
                    "logical_rows_ignored_by_kernel": True,
                },
                "packed_payload_bytes": sum(
                    item["codes_bytes"] + item["scales_bytes"] for item in layouts
                ),
                "tensors": layouts,
            }
            manifest_bytes = canonical_json_bytes(manifest)
            output.seek(manifest_offset)
            output.write(manifest_bytes)
            header = Lut23Header(
                tensor_count=len(layouts),
                source_file_bytes=source.stat().st_size,
                manifest_offset=manifest_offset,
                manifest_bytes=len(manifest_bytes),
                source_sha256=source_hash,
            )
            output.seek(0)
            output.write(header.pack())
            fsync_file(output)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "schema_version": 1,
        "format": "TQ1_LUT23_CODE_POSITION_MAJOR",
        "source": str(source),
        "source_sha256": source_hash,
        "destination": str(destination),
        "destination_file_bytes": destination.stat().st_size,
        "destination_sha256": sha256_file(destination),
        "tensor_count": len(layouts),
        "groups": sum(int(item["groups"]) for item in source_manifest["tensors"]),
        "weights": sum(int(item["elements"]) for item in source_manifest["tensors"]),
        "packed_payload_bytes": manifest["packed_payload_bytes"],
        "padding_bytes": manifest["packed_payload_bytes"] - int(source_manifest["packed_payload_bytes"]),
        "manifest_offset": manifest_offset,
        "manifest_bytes": len(manifest_bytes),
        "elapsed_seconds": time.time() - started,
    }


def read_lut23_sidecar(path: Path) -> tuple[Lut23Header, dict]:
    with path.open("rb") as handle:
        header = Lut23Header.unpack(handle.read(HEADER_BYTES))
        if header.manifest_offset + header.manifest_bytes > path.stat().st_size:
            raise Lut23FormatError("manifest outside LUT23 sidecar")
        handle.seek(header.manifest_offset)
        manifest = json.loads(handle.read(header.manifest_bytes))
    if manifest.get("format") != "TQ1_LUT23_CODE_POSITION_MAJOR":
        raise Lut23FormatError("unexpected LUT23 manifest format")
    if len(manifest.get("tensors", [])) != header.tensor_count:
        raise Lut23FormatError("LUT23 tensor count mismatch")
    return header, manifest


def write_benchmark_manifest(reordered: Path, traversal_json: Path, output: Path) -> dict:
    _, manifest = read_lut23_sidecar(reordered)
    traversal = json.loads(traversal_json.read_text(encoding="utf-8"))
    reordered_by_name = {item["name"]: item for item in manifest["tensors"]}
    columns = (
        "order",
        "layer",
        "kind",
        "name",
        "m",
        "k",
        "groups",
        "q2_file_offset",
        "q2_bytes",
        "tq1_file_offset",
        "tq1_bytes",
        "codes_file_offset",
        "codes_bytes",
        "scales_file_offset",
        "scales_bytes",
        "tiles",
        "groups_per_row",
    )
    lines = ["\t".join(columns)]
    for tensor in traversal["tensors"]:
        reordered_tensor = reordered_by_name[tensor["name"]]
        values = {
            **tensor,
            "codes_file_offset": reordered_tensor["codes_offset"],
            "codes_bytes": reordered_tensor["codes_bytes"],
            "scales_file_offset": reordered_tensor["scales_offset"],
            "scales_bytes": reordered_tensor["scales_bytes"],
            "tiles": reordered_tensor["tiles"],
            "groups_per_row": reordered_tensor["groups_per_row"],
        }
        lines.append("\t".join(str(values[key]) for key in columns))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(output)
    return {
        "schema_version": 1,
        "rows": len(traversal["tensors"]),
        "output": str(output),
        "primary": "blk.0.ffn_down.weight",
        "ring": [
            item["name"]
            for item in traversal["tensors"]
            if item["kind"] == "mlp_down" and item["m"] == 5120 and item["k"] == 17408
        ][:6],
    }


def _count_trit_mismatches(source_codes: np.ndarray, reordered_codes: np.ndarray) -> int:
    mismatches = 0
    full_mask = source_codes[:, :FULL_CODE_SLOTS] != reordered_codes[:, :FULL_CODE_SLOTS]
    for left, right in zip(
        source_codes[:, :FULL_CODE_SLOTS][full_mask],
        reordered_codes[:, :FULL_CODE_SLOTS][full_mask],
    ):
        lv, rv = int(left), int(right)
        for _ in range(5):
            mismatches += lv % 3 != rv % 3
            lv //= 3
            rv //= 3
    tail_mask = source_codes[:, 25] != reordered_codes[:, 25]
    for left, right in zip(source_codes[:, 25][tail_mask], reordered_codes[:, 25][tail_mask]):
        lv, rv = int(left), int(right)
        for _ in range(3):
            mismatches += lv % 3 != rv % 3
            lv //= 3
            rv //= 3
    return int(mismatches)


def verify_reordered_sidecar(source: Path, reordered: Path) -> dict:
    started = time.time()
    _, source_manifest = read_sidecar(source)
    header, manifest = read_lut23_sidecar(reordered)
    source_by_name = {item["name"]: item for item in source_manifest["tensors"]}
    code_byte_mismatches = 0
    ternary_mismatches = 0
    scale_byte_mismatches = 0
    invalid_regular = 0
    invalid_tail = 0
    padding_code_nonzero = 0
    padding_scale_nonzero = 0
    tensor_results = []
    with source.open("rb") as source_handle, reordered.open("rb") as reordered_handle:
        for number, tensor in enumerate(manifest["tensors"], start=1):
            original = source_by_name.get(tensor["name"])
            if original is None:
                raise Lut23FormatError(f"missing source tensor {tensor['name']}")
            rows = int(tensor["rows"])
            groups = int(tensor["groups_per_row"])
            tiles = int(tensor["tiles"])
            blocks = np.memmap(
                source_handle,
                mode="r",
                dtype=np.uint8,
                offset=int(original["packed_offset"]),
                shape=(rows, groups, BLOCK_BYTES),
            )
            codes = np.memmap(
                reordered_handle,
                mode="r",
                dtype=np.uint8,
                offset=int(tensor["codes_offset"]),
                shape=(tiles, groups, CODE_SLOTS, M_TILE),
            )
            scales = np.memmap(
                reordered_handle,
                mode="r",
                dtype=np.uint8,
                offset=int(tensor["scales_offset"]),
                shape=(tiles, groups, M_TILE, 2),
            )
            tensor_code_mismatch = 0
            tensor_trit_mismatch = 0
            tensor_scale_mismatch = 0
            for tile in range(tiles):
                begin = tile * M_TILE
                count = min(M_TILE, rows - begin)
                source_part = np.asarray(blocks[begin : begin + count])
                source_codes = source_part[:, :, 2:].transpose(1, 2, 0)
                actual_codes = np.asarray(codes[tile, :, :, :count])
                differences = int(np.count_nonzero(source_codes != actual_codes))
                tensor_code_mismatch += differences
                if differences:
                    tensor_trit_mismatch += _count_trit_mismatches(
                        source_codes.transpose(0, 2, 1).reshape(-1, CODE_SLOTS),
                        actual_codes.transpose(0, 2, 1).reshape(-1, CODE_SLOTS),
                    )
                source_scales = source_part[:, :, :2].transpose(1, 0, 2)
                tensor_scale_mismatch += int(
                    np.count_nonzero(source_scales != np.asarray(scales[tile, :, :count, :]))
                )
                invalid_regular += int(np.count_nonzero(actual_codes[:, :25] > 242))
                invalid_tail += int(np.count_nonzero(actual_codes[:, 25] > 26))
                if count != M_TILE:
                    padding_code_nonzero += int(np.count_nonzero(codes[tile, :, :, count:]))
                    padding_scale_nonzero += int(np.count_nonzero(scales[tile, :, count:, :]))
            code_byte_mismatches += tensor_code_mismatch
            ternary_mismatches += tensor_trit_mismatch
            scale_byte_mismatches += tensor_scale_mismatch
            tensor_results.append(
                {
                    "name": tensor["name"],
                    "groups": int(original["groups"]),
                    "weights": int(original["elements"]),
                    "code_byte_mismatches": tensor_code_mismatch,
                    "ternary_mismatches": tensor_trit_mismatch,
                    "scale_byte_mismatches": tensor_scale_mismatch,
                }
            )
            del scales, codes, blocks
            print(f"verified {number}/{len(manifest['tensors'])} {tensor['name']}", flush=True)
    passed = not any(
        (
            code_byte_mismatches,
            ternary_mismatches,
            scale_byte_mismatches,
            invalid_regular,
            invalid_tail,
            padding_code_nonzero,
            padding_scale_nonzero,
        )
    )
    return {
        "schema_version": 1,
        "pass": passed,
        "source_sha256": header.source_sha256,
        "tensor_count": len(tensor_results),
        "groups": sum(item["groups"] for item in tensor_results),
        "weights": sum(item["weights"] for item in tensor_results),
        "code_byte_mismatches": code_byte_mismatches,
        "ternary_mismatches": ternary_mismatches,
        "scale_byte_mismatches": scale_byte_mismatches,
        "invalid_regular_codes": invalid_regular,
        "invalid_tail_codes": invalid_tail,
        "padding_code_nonzero": padding_code_nonzero,
        "padding_scale_nonzero": padding_scale_nonzero,
        "tensors": tensor_results,
        "elapsed_seconds": time.time() - started,
    }


def reorder_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reorder TQ1 into LUT23 code-position-major layout")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--packing-result", type=Path)
    args = parser.parse_args(argv)
    try:
        result = reorder_sidecar(args.source, args.output, packing_result=args.packing_result)
        write_json_atomic(args.result, result)
    except Exception as exc:
        print(f"LUT23 reorder failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exhaustively verify LUT23 reordered TQ1")
    parser.add_argument("source", type=Path)
    parser.add_argument("reordered", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_reordered_sidecar(args.source, args.reordered)
        write_json_atomic(args.result, result)
    except Exception as exc:
        print(f"LUT23 verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: value for key, value in result.items() if key != "tensors"}, indent=2))
    return 0 if result["pass"] else 3


def manifest_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the LUT23 CUDA benchmark TSV")
    parser.add_argument("reordered", type=Path)
    parser.add_argument("--traversal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = write_benchmark_manifest(args.reordered, args.traversal, args.output)
    except Exception as exc:
        print(f"LUT23 manifest failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(reorder_main())
