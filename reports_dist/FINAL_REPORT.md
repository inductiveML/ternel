# BONSAI TQ1_G128 llama.cpp distribution validation

**Verdict: `DISTRIBUTION_VALIDATED_WITH_LIMITATIONS`**

The lossless TQ1 format now has a real single-file GGUF and a patched Prism
llama.cpp user path. The converted model loads with ordinary mmap or
`--no-mmap`, executes converted CUDA matrix multiplies from packed GPU storage,
and completes normal `llama cli` generation. It does not work with an
unpatched stock llama.cpp binary because GGML type 43 is experimental.

## Reproducible build and artifact

- Prism base commit: `9ca265a57f85f2117942490f421f64a226dd9847`
- Model revision: `abbae723028d71be674e71e1a71201a6f43fab22`
- Source SHA-256: `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`
- TQ1 GGUF SHA-256: `45f30340690a6cfd135f76f3d6e2d717d0cd059ff8187daa5bdd0c258e469101`
- Runtime patch: `patches/prism-tq1-g128-distribution.patch`

Exact build command used:

```bash
uv sync --extra dev --frozen
uv run python tools/bootstrap_prism.py
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build \
  --target llama-app llama-bench validate-llama-distribution -j 16
```

The patch was also applied from scratch to the pinned commit with
`git apply --check`, followed by `git diff --check`; both passed.

## Environment

- GPU: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, 49,140 MiB
- Driver: 570.133.20
- CUDA toolkit: 12.8.93
- GCC: 13.3.0
- CMake: 3.28.3
- Ninja: 1.13.0
- Python: 3.12 under uv 0.11.19
- Kernel: Linux 5.15.0-141-generic x86_64

A separate Python CUDA process retained 1,802 MiB during the paired run, as
explicitly allowed for this validation. VRAM figures below are process-specific
and timing order alternated Q2/TQ1, then TQ1/Q2.

## Correctness

| Check | Result |
|---|---:|
| Converted tensors | 498 / 498 |
| Packed bytes exhaustively verified | 5,882,920,960 |
| Logical ternary mismatches | 0 |
| FP16 scale-bit mismatches | 0 |
| Noncustom payload mismatches | 0 |
| Full-vocabulary logits compared | 8,194,560 |
| Teacher-forced positions | 33 |
| Greedy argmax matches | 33 / 33 |
| Max / mean absolute logit error | 0.121471 / 0.014254 |
| Logit RMSE | 0.018027 |
| Cosine similarity | 0.9999877864 |
| Non-finite values | 0 / 0 |

The storage transform is exact, so there is no quantization-quality loss. The
reported logit delta is floating-point drift from a different FP32 accumulation
order propagated through the recurrent model, not changed weights or scales.

The final ordinary frontend smoke test exited successfully:

```bash
build/bin/llama cli \
  -m artifacts/models/Ternary-Bonsai-27B-TQ1_G128.gguf \
  -ngl all -sm none --temp 0 -s 1234 \
  -p "The capital of France is" -n 12 \
  --single-turn --simple-io --color off --log-disable
```

It generated text and reported 27.6 prompt tok/s and 38.2 generation tok/s in
that single non-benchmark smoke run.

## Size and VRAM

| Measurement | Prism Q2 | TQ1_G128 | Change |
|---|---:|---:|---:|
| GGUF file bytes | 7,165,121,600 | 5,904,496,192 | -17.594% |
| CUDA model buffer | 6,500.64 MiB | 5,376.58 MiB | -17.292% |
| Process peak, pp32 + tg64 | 7,226 MiB | 6,240 MiB | -13.645% |

The smaller process-level saving includes identical KV/recurrent/context
buffers and TQ1 scratch. A forced pp512 diagnostic reached 8,744 MiB for TQ1
because the current scalar-prefill fallback creates an extremely large CUDA
graph; this is a runtime limitation, not an unpacked model copy. No model-sized
Q2, int8, FP16, or other expanded weight allocation is retained.

## Paired llama-bench result

Method: two alternating rounds, five warmed samples per round and model,
official llama-bench wall-clock timing around synchronized decode calls,
identical settings, mmap disabled for symmetric loading, and a 20,000-sample
bootstrap over paired latency ratios.

| Test | Q2 median | TQ1 median | TQ1/Q2 median | Bootstrap 95% CI | Ratio p5–p95 |
|---|---:|---:|---:|---:|---:|
| pp32 latency | 29.075 ms | 231.830 ms | 8.0187x | [7.8053, 9.0123] | [7.6017, 10.2144] |
| tg64 latency | 1,168.673 ms | 1,254.165 ms | 1.0730x | [1.0168, 1.1967] | [0.9663, 1.2597] |
| pp32 throughput | 1,100.60 tok/s | 138.03 tok/s | 0.1254x | — | — |
| tg64 throughput | 54.763 tok/s | 51.030 tok/s | 0.9318x | — | — |

Native decode therefore costs 7.30% median latency end to end in this run; it
does not meet the earlier 1.05x target. Prompt processing is 8.02x slower
because the distribution implementation deliberately loops the one-vector
native kernel rather than introducing an unvalidated batched packed kernel.

## How users can try it

This is a patched-fork distribution, not a GGUF that stock llama.cpp can load.
Users need both the TQ1 GGUF and a binary built with the supplied patch.

To reproduce the single-file artifact from the exact source and verified
sidecar:

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

Then build and use `build/bin/llama cli` as shown above. The runtime currently
supports one CUDA device. CPU fallback code is implemented, but full partial
offload was not benchmarked; multi-GPU split buffers are intentionally rejected.

Raw results are under `artifacts/results_dist/`, including conversion,
exhaustive parity, logits, CLI smoke output, the corrected paired benchmark,
and the pp512 diagnostic.
