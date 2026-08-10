# Ternel MLX handoff prompt

Copy the prompt below into the next implementation task. It is deliberately
self-contained so the MLX work can continue without reconstructing the CUDA
experiment from conversation history.

---

You are continuing **Ternel**, a research and distribution project for native,
lossless execution of the PrismML Ternary Bonsai 27B model. Build and validate
an MLX/Metal distribution path for the already-proven `TQ1_G128` storage
format. Work in this repository and preserve the frozen CUDA evidence; do not
repeat completed multi-billion-value scans unless a changed artifact requires
it.

## Objective

Deliver a user-installable Apple Silicon implementation that:

1. preserves every logical ternary weight and every raw FP16 g128 scale bit;
2. retains the compact 28-byte-per-128-weight representation during loading
   and execution;
3. executes packed weights directly with an MLX custom Metal operation;
4. never allocates or retains a model-sized Q2, int8-expanded, FP16, BF16, or
   other unpacked copy;
5. can be installed and run with `uv` from a clean environment; and
6. reports correctness, unified-memory use, prompt processing, and generation
   performance against honest MLX and reference baselines.

Prefer an MLX-native Hugging Face distribution using packed safetensors and a
small Python package over a fork of MLX. A package-level implementation using
`mx.fast.metal_kernel` is the first choice. Use a compiled MLX extension only
if profiling proves the package-level path cannot provide the required control
or performance. The model metadata reports the Qwen3.5 hybrid architecture,
for which current MLX-LM has a model implementation; reuse that graph rather
than recreating the architecture.

## Frozen source identities

- Model repository: `prism-ml/Ternary-Bonsai-27B-gguf`
- Model revision: `abbae723028d71be674e71e1a71201a6f43fab22`
- Source file: `Ternary-Bonsai-27B-Q2_0.gguf`
- Exact source size: `7,165,121,600` bytes
- Source SHA-256:
  `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`
- Pinned Prism llama.cpp commit:
  `9ca265a57f85f2117942490f421f64a226dd9847`
- Converted custom GGUF size: `5,904,496,192` bytes
- Converted custom GGUF SHA-256:
  `45f30340690a6cfd135f76f3d6e2d717d0cd059ff8187daa5bdd0c258e469101`
- Runtime distribution patch SHA-256:
  `a4aec826900030dbfca61a1c422957e930861b5196e5a954d7460cd480c430c2`

Large artifacts are intentionally ignored. If they are absent, download the
exact pinned source and verify both size and hash before conversion. Every
Python command must run through `uv`.

## Proven representation

The released checkpoint contains 498 Q2_0/g128 tensors, 26,893,352,960
logical values, and 210,104,320 independent FP16 scale groups. An exhaustive
scan proved that every source code is in the ternary alphabet `0/1/2`; source
code `3` never occurs.

One canonical `TQ1_G128` block represents 128 weights in 28 bytes:

- bytes 0-1: original little-endian FP16 scale, untouched bit-for-bit;
- bytes 2-26: 25 base-3 bytes, each encoding five trits with powers
  `1, 3, 9, 27, 81`;
- byte 27: one base-3 tail byte encoding the final three trits.

Codes `0/1/2` map directly to `-1/0/+1`. Canonical decoders reject regular
bytes above 242, tail bytes above 26, and source Q2 code 3. The representation
is 1.75 bits/weight including the FP16 scale, compared with 2.125 bits/weight
for the source Q2/g128 layout.

Frozen exhaustive verification established:

- zero ternary mismatches over all 26,893,352,960 weights;
- zero raw scale-bit mismatches over all 210,104,320 groups;
- 498/498 converted tensors verified;
- 5,882,920,960 packed quantized bytes verified; and
- all 10,582,016 non-custom bytes in the distributed GGUF byte-equal.

Do not substitute standard MLX affine 2-bit quantization, GGUF `TQ1_0`, sparse
storage, or any representation that changes scales or logical weights.

## MLX storage and loading design

Create an MLX-native artifact containing packed byte arrays and an explicit
canonical manifest. Safetensors is preferred because it avoids adding another
private GGUF tensor enum to MLX. Preserve scale bits as raw bytes (or another
bit-preserving integer view), not through a floating-point conversion.

The CUDA implementation transforms canonical row-major blocks at upload into:

```text
codes[m_tile][g128][code_slot][row_in_tile]
scales[m_tile][g128][row_in_tile]
```

with `M_TILE=256`. On Apple unified memory, retaining both canonical and
reordered model-sized arrays is forbidden. Either distribute the MLX artifact
already in the selected Metal-native layout or convert tensor-by-tensor while
ensuring the source mapping and temporary storage are released. Measure actual
peak resident/unified memory instead of assuming mmap pages are free.

Include enough metadata to reconstruct each logical tensor name, shape,
grouping, layout version, offsets, byte counts, source identity, logical-weight
hash, and scale-bit hash. Validate metadata and payload boundaries before a
kernel can consume them. Corruption must fail closed.

Implement packed replacements for every operation that consumes a converted
weight, at minimum linear projection and embedding/get-rows. Ordinary
`mlx_lm.generate` does not understand this private layout, so provide a small
package and CLI that construct the supported Qwen3.5 model and replace the
relevant modules without expanding weights. Aim for a clean user flow similar
to:

```bash
uvx --from ternel mlx-ternel-generate \
  --model inductiveML/Ternary-Bonsai-27B-TQ1-MLX \
  --prompt "The capital of France is"
```

The exact command may change, but installation must not require users to patch
or compile MLX itself unless the measured extension fallback proves necessary.

## Metal kernel

Port the direct LUT23 mathematics, not the CUDA source text. For each legal
five-trit byte `p`:

```text
b = (p * 57) >> 9
a = p - 9*b
```

For activation positions `x[5j:5j+5]`, construct a 9-entry two-trit table and
a 27-entry three-trit table, then consume the packed byte directly as the sum
of those two lookups. Never materialize five ternary weights. Preserve the
same logical scale application and accumulation precision used by the chosen
MLX reference path.

CUDA choices such as four warps per row, Stream-K chunk sizes, shared-memory
bank padding, and `M_TILE=256` are evidence, not immutable Metal choices.
Apple GPU threadgroup memory, simdgroups, occupancy, caches, and unified-memory
behavior require measurement. Start with a correctness-first Metal kernel,
benchmark a representative real tensor, profile, and make bounded,
evidence-driven changes. Avoid an endless tuning loop.

The frozen CUDA LUT23 branch was not a speed win and must not be described as
one. In the final llama.cpp integration on RTX 6000 Ada:

- CUDA model-buffer saving: 17.29% (`6500.64` to `5376.58` MiB);
- observed pp32/tg64 process peak saving: 13.65% (`7226` to `6240` MiB);
- token-generation latency ratio versus Q2: `1.0730x` (about 6.82% lower
  throughput);
- pp32 prompt-processing latency ratio: `8.0187x`, caused by the one-vector
  fallback for multi-vector work.

The MLX implementation must provide a true batched packed matmul path; do not
repeat the CUDA integration's scalar-prefill fallback.

## Correctness gates

Before full-model timing:

1. Unit-test all 243 legal five-trit bytes, tail boundaries, malformed bytes,
   arbitrary FP16 scale bit patterns, tensor padding, chunk boundaries, and
   deterministic conversion.
2. Exhaustively verify the distributed MLX artifact against the frozen source
   hashes: zero logical ternary and zero raw scale-bit mismatches.
3. Compare the Metal operation with two independent CPU/reference decoders on
   synthetic matrices and several real tensors, covering single-vector and
   batched inputs, awkward row counts, FP16-origin and BF16-origin activations,
   zeros, extremes, and non-finite rejection.
4. Run at least 128 deterministic activation cases plus edge cases per primary
   operation shape. Report max/mean absolute error, RMSE, relative error,
   cosine similarity, mismatch counts, and non-finite counts.
5. Validate a full MLX graph with teacher-forced tokens and ordinary generation.
   Compare greedy token decisions and logits with the most faithful available
   reference, while clearly separating exact storage parity from expected
   cross-runtime accumulation-order drift.

The earlier llama.cpp full-graph comparison checked 8,194,560 logits over 33
teacher-forced positions: max absolute error `0.1214711666`, mean absolute
error `0.0142538247`, RMSE `0.0180273960`, cosine `0.9999877864`, 33/33
argmax agreement, and zero non-finite values. Those numbers are context, not
automatic MLX acceptance thresholds.

## Performance and memory evaluation

Benchmark on identified Apple Silicon hardware with OS, MLX, MLX-LM, Python,
compiler, GPU-core count, memory size, power mode, and thermal state recorded.
Use paired alternating order, warmups, enough repetitions, synchronized MLX
evaluation, medians, p5/p95, and bootstrap 95% confidence intervals.

Measure separately:

- representative single-vector decode GEMV;
- batched prompt/prefill matmul across useful prompt lengths;
- embeddings/get-rows;
- full-model prompt processing and token generation;
- cold load and first-kernel compilation;
- warm steady state; and
- peak and steady unified-memory residency.

All layout preparation, LUT construction, synchronization, partial reductions,
and scratch traffic used during inference must be included. Report model file
bytes, packed payload bytes, resident model bytes, peak process bytes, scratch,
cache, and temporary conversion memory. Add assertions or instrumentation that
prove there is no retained model-sized unpacked copy.

Choose honest baselines available in MLX and label their representation and
quality differences. Do not claim a speed ratio against llama.cpp CUDA as an
MLX result. The primary question is whether lossless packed MLX execution is
usable and memory-material relative to the best faithful Apple-Silicon path.

## Existing repository evidence

Read these before changing the format or duplicating work:

- `README.md`
- `reports_dist/FINAL_REPORT.md`
- `reports_lut23/FINAL_REPORT.md`
- `reports_v2/FINAL_REPORT.md`
- `reports/FINAL_REPORT.md`
- `src/bonsai_tq1/format.py`
- `src/bonsai_tq1/convert_gguf.py`
- `src/bonsai_tq1/lut23_reorder.py`
- `cuda/tq1_lut23_streamk.cu`
- `patches/prism-tq1-g128-distribution.patch`

The internal historical Python module remains named `bonsai_tq1`; the project
and distribution name is now `ternel`.

## Deliverables

- reproducible `uv` project updates with pinned compatible MLX/MLX-LM versions;
- streaming converter and bit-exact verifier;
- MLX packed modules and Metal kernels, including a genuine batched path;
- user-facing generate/chat smoke command;
- tests and machine-readable raw results;
- concise Markdown report covering environment, format identity, exactness,
  numerical behavior, latency/throughput, unified memory, limitations, and
  reproduction commands; and
- a clear final verdict: viable MLX distribution, memory-only result, or
  implementation not qualified.

Do not publish model artifacts or claim compatibility/performance until the
actual target Mac completes these gates.

---
