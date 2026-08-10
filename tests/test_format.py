from __future__ import annotations

import itertools
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from bonsai_tq1.format import (
    BLOCK_BYTES,
    HEADER_BYTES,
    HEADER_STRUCT,
    SOURCE_BLOCK_BYTES,
    FormatError,
    SidecarHeader,
    align_up,
    canonical_json_bytes,
    decode_q2_blocks,
    decode_tq1_blocks,
    encode_tq1_blocks,
    iter_memmap_blocks,
    read_sidecar,
)
from bonsai_tq1.reference import q2_g128_reference_gemv, tq1_g128_reference_gemv
from bonsai_tq1.lut23_reorder import (
    M_TILE,
    reorder_tensor_arrays,
    restore_tensor_arrays,
    tensor_layout,
)
from bonsai_tq1.lut23_reporting import paired_bootstrap_ratio


def q2_encode_for_test(scales: np.ndarray, codes: np.ndarray) -> np.ndarray:
    result = np.zeros((codes.shape[0], SOURCE_BLOCK_BYTES), dtype=np.uint8)
    result[:, :2] = scales
    fields = codes.reshape(codes.shape[0], 32, 4)
    for index, shift in enumerate((0, 2, 4, 6)):
        result[:, 2:] |= fields[:, :, index] << shift
    return result


def test_every_legal_five_trit_byte_round_trips() -> None:
    codes = np.ones((243, 128), dtype=np.uint8)
    for value in range(243):
        current = value
        for trit in range(5):
            codes[value, trit] = current % 3
            current //= 3
    scales = np.arange(243, dtype=np.uint16).view(np.uint8).reshape(243, 2)
    packed = encode_tq1_blocks(scales, codes)
    assert np.array_equal(packed[:, 2], np.arange(243, dtype=np.uint8))
    decoded_scales, decoded_codes = decode_tq1_blocks(packed)
    assert np.array_equal(decoded_scales, scales)
    assert np.array_equal(decoded_codes, codes)


@pytest.mark.parametrize("tail", range(27))
def test_every_legal_tail_byte_round_trips(tail: int) -> None:
    codes = np.ones((1, 128), dtype=np.uint8)
    current = tail
    for index in range(3):
        codes[0, 125 + index] = current % 3
        current //= 3
    packed = encode_tq1_blocks(np.asarray([[0x34, 0x12]], dtype=np.uint8), codes)
    assert int(packed[0, 27]) == tail
    _, decoded = decode_tq1_blocks(packed)
    assert np.array_equal(decoded, codes)


def test_all_ternary_patterns_and_scale_bits() -> None:
    patterns = np.asarray(list(itertools.product(range(3), repeat=5)), dtype=np.uint8)
    codes = np.resize(patterns.reshape(-1), 4 * 128).reshape(4, 128)
    scales = np.asarray([[0x00, 0x00], [0x00, 0x80], [0x00, 0x7C], [0xFF, 0x7B]], dtype=np.uint8)
    packed = encode_tq1_blocks(scales, codes)
    decoded_scales, decoded_codes = decode_tq1_blocks(packed)
    assert np.array_equal(decoded_scales, scales)
    assert np.array_equal(decoded_codes, codes)


def test_all_fp16_scale_bit_patterns_are_preserved() -> None:
    scale_words = np.arange(1 << 16, dtype=np.uint16)
    scales = scale_words.view(np.uint8).reshape(-1, 2)
    codes = np.ones((scale_words.size, 128), dtype=np.uint8)
    packed = encode_tq1_blocks(scales, codes)
    decoded_scales, decoded_codes = decode_tq1_blocks(packed)
    assert decoded_scales.tobytes() == scales.tobytes()
    assert np.array_equal(decoded_codes, codes)


def test_q2_decoder_order_and_non_ternary_rejection() -> None:
    codes = np.tile(np.asarray([0, 1, 2, 3], dtype=np.uint8), 32).reshape(1, 128)
    scales = np.asarray([[0xAA, 0x55]], dtype=np.uint8)
    source = q2_encode_for_test(scales, codes)
    decoded_scales, decoded_codes = decode_q2_blocks(source)
    assert np.array_equal(decoded_scales, scales)
    assert np.array_equal(decoded_codes, codes)
    with pytest.raises(FormatError, match="non-ternary"):
        encode_tq1_blocks(scales, codes)


def test_invalid_base3_values_are_rejected() -> None:
    packed = np.zeros((1, BLOCK_BYTES), dtype=np.uint8)
    packed[0, 2] = 243
    with pytest.raises(FormatError, match="regular=1"):
        decode_tq1_blocks(packed)
    packed[0, 2] = 0
    packed[0, 27] = 27
    with pytest.raises(FormatError, match="tail=1"):
        decode_tq1_blocks(packed)


def test_header_round_trip_and_reserved_bytes() -> None:
    header = SidecarHeader(
        version=1,
        header_bytes=256,
        block_size=128,
        block_bytes=28,
        alignment=256,
        tensor_count=2,
        source_file_bytes=1234,
        manifest_offset=4096,
        manifest_bytes=987,
        source_sha256="12" * 32,
    )
    encoded = header.pack()
    assert len(encoded) == HEADER_BYTES
    assert encoded[88:] == bytes(HEADER_BYTES - 88)
    assert SidecarHeader.unpack(encoded) == header


def test_header_rejects_nonhex_hash_and_nonzero_reserved_byte() -> None:
    header = SidecarHeader(1, 256, 128, 28, 256, 0, 0, 256, 2, "z" * 64)
    with pytest.raises(FormatError, match="hexadecimal"):
        header.pack()
    valid = SidecarHeader(1, 256, 128, 28, 256, 0, 0, 256, 2, "00" * 32)
    corrupted = bytearray(valid.pack())
    corrupted[-1] = 1
    with pytest.raises(FormatError, match="reserved"):
        SidecarHeader.unpack(corrupted)


def test_reference_gemv_matches_after_repacking() -> None:
    rng = np.random.default_rng(20260808)
    rows, columns = 3, 256
    groups = rows * columns // 128
    scales_u16 = rng.integers(1, 0x7BFF, size=groups, dtype=np.uint16)
    scales = scales_u16.view(np.uint8).reshape(groups, 2)
    codes = rng.integers(0, 3, size=(groups, 128), dtype=np.uint8)
    source = q2_encode_for_test(scales, codes)
    packed = encode_tq1_blocks(scales, codes)
    activation = rng.normal(size=columns).astype(np.float16).astype(np.float32)
    original_output = q2_g128_reference_gemv(source, activation, rows=rows, columns=columns)
    packed_output = tq1_g128_reference_gemv(packed, activation, rows=rows, columns=columns)
    assert np.array_equal(original_output, packed_output)


def test_encoding_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    scales = rng.integers(0, 256, size=(100, 2), dtype=np.uint8)
    codes = rng.integers(0, 3, size=(100, 128), dtype=np.uint8)
    assert encode_tq1_blocks(scales, codes).tobytes() == encode_tq1_blocks(scales, codes).tobytes()


def test_chunk_boundaries_match_one_shot_conversion() -> None:
    rng = np.random.default_rng(123)
    scales = rng.integers(0, 256, size=(17, 2), dtype=np.uint8)
    codes = rng.integers(0, 3, size=(17, 128), dtype=np.uint8)
    expected = encode_tq1_blocks(scales, codes)
    for chunk_size in (1, 2, 3, 5, 16, 17, 32):
        pieces = [
            encode_tq1_blocks(scales[start : start + chunk_size], codes[start : start + chunk_size])
            for start in range(0, scales.shape[0], chunk_size)
        ]
        assert np.array_equal(np.concatenate(pieces), expected)


def test_iter_memmap_blocks_honors_offset_and_final_chunk(tmp_path: Path) -> None:
    prefix = bytes(range(11))
    matrix = np.arange(7 * 6, dtype=np.uint8).reshape(7, 6)
    path = tmp_path / "blocks.bin"
    path.write_bytes(prefix + matrix.tobytes())
    chunks = list(
        iter_memmap_blocks(path, offset=len(prefix), block_count=7, block_bytes=6, chunk_groups=3)
    )
    assert [chunk.shape[0] for chunk in chunks] == [3, 3, 1]
    assert np.array_equal(np.concatenate(chunks), matrix)


def _sidecar_bytes() -> bytes:
    tensors = [
        {"name": "a", "packed_offset": 256, "packed_bytes": 56, "groups": 2},
        {"name": "b", "packed_offset": 512, "packed_bytes": 28, "groups": 1},
    ]
    manifest = canonical_json_bytes({"format": "TQ1_G128", "tensors": tensors})
    header = SidecarHeader(
        version=1,
        header_bytes=256,
        block_size=128,
        block_bytes=28,
        alignment=256,
        tensor_count=2,
        source_file_bytes=999,
        manifest_offset=540,
        manifest_bytes=len(manifest),
        source_sha256="ab" * 32,
    )
    result = bytearray(header.pack())
    result.extend(bytes(540 - len(result)))
    result.extend(manifest)
    return bytes(result)


def test_read_sidecar_accepts_valid_aligned_tensor_extents(tmp_path: Path) -> None:
    path = tmp_path / "valid.tq1"
    path.write_bytes(_sidecar_bytes())
    header, manifest = read_sidecar(path)
    assert header.tensor_count == 2
    assert [tensor["name"] for tensor in manifest["tensors"]] == ["a", "b"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("magic", "magic"),
        ("version", "version"),
        ("geometry", "geometry"),
        ("manifest_bounds", "outside"),
        ("manifest_json", "manifest"),
        ("manifest_format", "format"),
        ("tensor_count", "tensor count"),
        ("unaligned_tensor", "extent"),
        ("wrong_tensor_size", "extent"),
        ("overlap_manifest", "overlaps"),
    ],
)
def test_read_sidecar_rejects_corruption(tmp_path: Path, mutation: str, message: str) -> None:
    data = bytearray(_sidecar_bytes())
    header = SidecarHeader.unpack(data[:HEADER_BYTES])
    manifest = json.loads(data[header.manifest_offset :].decode())
    if mutation == "magic":
        data[0] ^= 0xFF
    elif mutation == "version":
        struct.pack_into("<I", data, 8, 99)
    elif mutation == "geometry":
        struct.pack_into("<I", data, 16, 127)
    elif mutation == "manifest_bounds":
        struct.pack_into("<Q", data, 48, len(data) + 1)
    elif mutation == "manifest_json":
        data[header.manifest_offset] = 0xFF
    else:
        if mutation == "manifest_format":
            manifest["format"] = "wrong"
        elif mutation == "tensor_count":
            manifest["tensors"].pop()
        elif mutation == "unaligned_tensor":
            manifest["tensors"][1]["packed_offset"] = 511
        elif mutation == "wrong_tensor_size":
            manifest["tensors"][0]["packed_bytes"] = 55
        elif mutation == "overlap_manifest":
            manifest["tensors"][1]["packed_bytes"] = 56
            manifest["tensors"][1]["groups"] = 2
        replacement = canonical_json_bytes(manifest)
        data = data[: header.manifest_offset] + replacement
        struct.pack_into("<Q", data, 48, len(replacement))
    path = tmp_path / f"{mutation}.tq1"
    path.write_bytes(data)
    with pytest.raises(FormatError, match=message):
        read_sidecar(path)


@pytest.mark.parametrize("alignment", (0, -1, 3, 6))
def test_align_up_rejects_non_power_of_two(alignment: int) -> None:
    with pytest.raises(ValueError, match="power of two"):
        align_up(10, alignment)


def test_decoders_reject_truncated_or_misshaped_inputs() -> None:
    with pytest.raises(FormatError, match="divisible by 34"):
        decode_q2_blocks(np.zeros(33, dtype=np.uint8))
    with pytest.raises(FormatError, match="divisible by 28"):
        decode_tq1_blocks(np.zeros(27, dtype=np.uint8))
    with pytest.raises(FormatError, match="expected Q2"):
        decode_q2_blocks(np.zeros((1, 2, 17), dtype=np.uint8))
    with pytest.raises(FormatError, match="expected TQ1"):
        decode_tq1_blocks(np.zeros((2, 14), dtype=np.uint8))


@pytest.mark.parametrize("rows", (1, 48, 256, 257, 513))
def test_lut23_code_position_major_round_trip(rows: int) -> None:
    generator = np.random.default_rng(20260809 + rows)
    scales = generator.integers(0, 256, size=(rows, 3, 2), dtype=np.uint8)
    logical = generator.integers(0, 3, size=(rows * 3, 128), dtype=np.uint8)
    packed = encode_tq1_blocks(scales.reshape(-1, 2), logical).reshape(rows, 3, BLOCK_BYTES)
    codes, reordered_scales = reorder_tensor_arrays(packed)
    assert codes.shape == ((rows + M_TILE - 1) // M_TILE, 3, 26, M_TILE)
    assert reordered_scales.shape == ((rows + M_TILE - 1) // M_TILE, 3, M_TILE, 2)
    np.testing.assert_array_equal(restore_tensor_arrays(codes, reordered_scales, rows), packed)
    padded = codes.shape[-1] - (rows % M_TILE or M_TILE)
    if padded:
        assert not np.any(codes[-1, :, :, -padded:])
        assert not np.any(reordered_scales[-1, :, -padded:, :])


def test_lut23_reorder_rejects_wrong_block_geometry() -> None:
    with pytest.raises(ValueError, match="shape"):
        reorder_tensor_arrays(np.zeros((2, 3, 27), dtype=np.uint8))


def test_lut23_mulshift_split_is_exact_for_all_243_codes() -> None:
    for packed in range(243):
        upper = (packed * 57) >> 9
        lower = packed - 9 * upper
        assert (upper, lower) == divmod(packed, 9)
        assert 0 <= lower < 9
        assert 0 <= upper < 27


def test_lut23_two_plus_three_tables_equal_direct_ternary_dot() -> None:
    activation = np.linspace(-3.0, 4.0, 5, dtype=np.float32)
    l2 = np.empty(9, dtype=np.float32)
    l3 = np.empty(27, dtype=np.float32)
    for code in range(9):
        l2[code] = sum(((code // (3**digit)) % 3 - 1) * activation[digit] for digit in range(2))
    for code in range(27):
        l3[code] = sum(
            ((code // (3**digit)) % 3 - 1) * activation[digit + 2] for digit in range(3)
        )
    for packed in range(243):
        upper = (packed * 57) >> 9
        lower = packed - 9 * upper
        direct = sum(
            ((packed // (3**digit)) % 3 - 1) * activation[digit] for digit in range(5)
        )
        assert float(l2[lower] + l3[upper]) == pytest.approx(float(direct), abs=1e-6)


def test_lut23_every_code_subtable_is_128_byte_aligned() -> None:
    layout = tensor_layout(rows=257, groups=17, start=256)
    assert layout["codes_offset"] % 256 == 0
    assert layout["scales_offset"] % 256 == 0
    for tile in range(layout["tiles"]):
        for group in range(17):
            for slot in range(26):
                offset = layout["codes_offset"] + (((tile * 17 + group) * 26 + slot) * M_TILE)
                assert offset % 128 == 0


def test_lut23_paired_bootstrap_is_deterministic() -> None:
    baseline = [1.0 + index / 1000 for index in range(40)]
    candidate = [value * 1.2 for value in baseline]
    first = paired_bootstrap_ratio(baseline, candidate, samples=1000, seed=7)
    second = paired_bootstrap_ratio(baseline, candidate, samples=1000, seed=7)
    assert first == second
    assert first["ci95"] == pytest.approx([1.2, 1.2])
