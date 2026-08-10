# Ternel

Ternel develops native, lossless execution formats for ternary language
models. The first completed branch losslessly repacks the released PrismML
Ternary Bonsai 27B Q2_0/g128 weights into a directly consumable base-3 CUDA
format.

The next development target is an MLX/Metal distribution for Apple Silicon.
See the self-contained [`MLX_HANDOFF_PROMPT.md`](MLX_HANDOFF_PROMPT.md) before
continuing that work.

## Patched llama.cpp distribution

The native format now has a validated single-file GGUF distribution path.
`Ternary-Bonsai-27B-TQ1_G128.gguf` is 5,904,496,192 bytes with SHA-256
`45f30340690a6cfd135f76f3d6e2d717d0cd059ff8187daa5bdd0c258e469101`.
It preserves every ternary value and FP16 scale bit, but requires the supplied
patch against the pinned Prism commit; an unpatched stock llama.cpp does not
know experimental GGML type 43.

Build the tested user-facing binary:

```bash
uv sync --extra dev --frozen
uv run python tools/bootstrap_prism.py
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build \
  --target llama-app llama-bench validate-llama-distribution -j 16
```

Run it like a normal local model:

```bash
build/bin/llama cli \
  -m artifacts/models/Ternary-Bonsai-27B-TQ1_G128.gguf \
  -ngl all -sm none --temp 0 \
  -p "The capital of France is" -n 64 --single-turn
```

The end-to-end result on RTX 6000 Ada saves 17.29% in the CUDA model buffer
and 13.65% in observed process VRAM for pp32/tg64. Decode latency was 1.0730x
Q2 (7.30% slower); the scalar-prefill fallback was 8.0187x Q2. Exhaustive file
parity, 8,194,560 full-graph logits, all 33 greedy decisions, and ordinary
`llama cli` generation passed. See
[`reports_dist/FINAL_REPORT.md`](reports_dist/FINAL_REPORT.md) and apply
[`patches/prism-tq1-g128-distribution.patch`](patches/prism-tq1-g128-distribution.patch).

Rebuild the distributable GGUF from the verified canonical sidecar:

```bash
uv run python tools/convert_tq1_gguf.py \
  artifacts/models/Ternary-Bonsai-27B-Q2_0.gguf \
  artifacts/packed/Ternary-Bonsai-27B-Q2_0.gguf.tq1g128 \
  artifacts/models/Ternary-Bonsai-27B-TQ1_G128.gguf \
  --result artifacts/results_dist/conversion.json

uv run python tools/verify_distributed_gguf.py \
  artifacts/models/Ternary-Bonsai-27B-Q2_0.gguf \
  artifacts/packed/Ternary-Bonsai-27B-Q2_0.gguf.tq1g128 \
  artifacts/models/Ternary-Bonsai-27B-TQ1_G128.gguf \
  --result artifacts/results_dist/file_parity.json
```

V1 remains frozen at **`PACKING_ONLY_KERNEL_FAIL`**. Packing passes: all
26,893,352,960 weights and 210,104,320 FP16 scale groups round-trip exactly,
and the projected file falls from 7,165,121,600 to 5,904,495,680 bytes. On an
idle RTX 6000 Ada, direct TQ1 V1 is 5.41% faster cold-ish but 72.97% slower
warm-cache; the required worse ratio is 1.729730×, above the 1.20 failure
threshold. See [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md).

The cache-realistic V2 follow-up is also closed:
**`STREAMING_TQ1_FAIL`**. A 497-GEMV real model-order traversal touches
6,805,831,680 Q2 bytes or 5,604,802,560 TQ1 bytes. Its idle-GPU medians are
8.972416 ms (Q2) and 11.273168 ms (TQ1), a 1.256425× ratio. The cache
explanation therefore failed for that frozen branch. See
[`reports_v2/FINAL_REPORT.md`](reports_v2/FINAL_REPORT.md). Those historical
reports remain hash-frozen; the later distribution work is reported separately
under `reports_dist/`.

## Reproducible environment

Python 3.12 and every Python dependency/entry point are managed by
[`uv`](https://docs.astral.sh/uv/). Native CUDA is built by CMake/NVCC under the
Python orchestrator.

```bash
uv sync --extra dev --frozen
uv run pytest
```

Pinned inputs:

- Model repository: `prism-ml/Ternary-Bonsai-27B-gguf`
- Model revision: `abbae723028d71be674e71e1a71201a6f43fab22`
- Model SHA-256: `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`
- Prism llama.cpp commit: `9ca265a57f85f2117942490f421f64a226dd9847`

## Single runner

Run every gate in order:

```bash
uv run ternel-experiment all
```

The CUDA stage refuses to run unless it sees the exact RTX 6000 Ada (`sm_89`)
and no competing CUDA processes. This prevents contaminated timings from being
promoted into a verdict. Existing successful stage artifacts are reused;
`--force` repeats them.

Individual resumable stages are:

```bash
uv run ternel-experiment download
uv run ternel-experiment audit
uv run ternel-experiment pack
uv run ternel-experiment verify
uv run ternel-experiment cuda
uv run ternel-experiment reports
```

## Manual commands

All Python commands still run through `uv`:

```bash
uv run python tools/inspect_bonsai_gguf.py \
  artifacts/models/Ternary-Bonsai-27B-Q2_0.gguf \
  --output artifacts/results/format_audit.json

uv run python tools/pack_tq1_g128.py \
  artifacts/models/Ternary-Bonsai-27B-Q2_0.gguf \
  --audit artifacts/results/format_audit.json \
  --output artifacts/packed/Ternary-Bonsai-27B-Q2_0.gguf.tq1g128 \
  --result artifacts/results/packing.json

uv run python tools/verify_tq1_g128.py \
  artifacts/models/Ternary-Bonsai-27B-Q2_0.gguf \
  artifacts/packed/Ternary-Bonsai-27B-Q2_0.gguf.tq1g128 \
  --result artifacts/results/lossless_verification.json \
  --report reports/02_lossless_verification.md

uv run python tools/bootstrap_prism.py
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target benchmark-gemv validate-prism-bridge -j 24
```

Large downloaded, packed, build, raw timing, and profiler artifacts live under
ignored `artifacts/` and `build/`. V1 source reports are under `reports/`; V2
reports are under `reports_v2/` and raw V2 results under
`artifacts/results_v2/`.

## Format summary

`TQ1_G128` sidecar v1 is little-endian. Its 256-byte header identifies the
source hash, block geometry, alignment, tensor count, and canonical JSON
manifest. Each 28-byte block contains the original two FP16 scale bytes and 26
base-3 bytes: 25 encode five trits each and the final byte encodes three. The
canonical decoder rejects values above 242, a tail above 26, and any source Q2
code 3. No training, requantization, pruning, or full-matrix decompression is
performed.
