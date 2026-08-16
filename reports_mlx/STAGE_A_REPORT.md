# Ternel MLX/Metal — Stage A report

> **Three corrections, applied after Stage B measured the model.** This report is
> left as the record of what Stage A found; all three are carried in
> `FINAL_REPORT.md`, which supersedes it.
>
> 1. **The census below counts a tensor the model does not have.** The sweep's
>    ten shapes include a 5120×12288 `self_attn.o_proj`; the model's `o_proj` is
>    5120×6144, the same shape as `linear_attn.out_proj`. So the scope is 99
>    measured points, not 110, and the real `out_proj` carries the weight the
>    phantom was given. Re-weighting changes almost nothing here — the
>    forward-pass ratios below move by at most 0.006 and the batch-1 figure not
>    at all, because the two shapes ran at similar ratios — but the dispatch-rule
>    scores restate as **43/99 exact, mean 2.04%, worst 14.52%** on the floor of
>    the three sweeps, and **40/99, mean 2.46%** on the sweep this report used.
>    `ternel_mlx.layout.MODEL_LINEAR_SHAPES` is now the one census, and it fails
>    at import unless its nine shapes account for exactly the 497 quantised
>    matmuls a decoded token issues.
> 2. **Every ratio here is a throughput ratio, not a latency one.** A5 enqueues
>    each configuration's calls with no dependency between them, so MLX overlaps
>    them. Decode does not get that: its 497 matmuls form one read-after-write
>    chain. Stage B's A6 sweep measured the same shapes with the chain forced and
>    found the unsplit kernel paying up to 8.42× its own overlapped time, which
>    is why the model was slower than this report's op-level numbers predict.
>    Splitting the K axis removes it. The kernel that ships is the split one.
> 3. **Everything here was measured in float32; the model runs bfloat16.** The
>    sweep built its activations as float32 and never said so, and the dtype is
>    not a detail: it is a kernel template parameter, so it selects which
>    compiled kernel the packed arm runs, and — through the scales
>    `build_affine_baseline` produced — which kernel MLX runs on the affine arm,
>    since bfloat16 activations against float16 scales promote to float32. Both
>    arms of every ratio below are therefore the float32 pair, not the pair a
>    user gets. The dtype is now a required argument to `ternel_mlx.bench_ops`
>    with no default, and the sweep records it.
>
>    **It has since been measured, and it reverses this report's headline.**
>    Re-swept in bfloat16 with the affine control rebuilt in the same dtype, the
>    weighted forward-pass ratio reads **0.86x at batch 1, 0.81x at 2 and 4,
>    0.86x at 8, 1.17x at 16, and 0.88x–1.03x from 32 through 512** — against the
>    1.09x/1.64x/1.77x/2.11x/1.27x below. The two arms move by very different
>    amounts, which is the signature of the defect: across the nine dispatched
>    shapes the packed arm's time changes by a median 0.93x, the affine arm's by
>    0.76x, and at batches 4 and 8 the affine kernel runs 2x to 3x faster while
>    the packed one stays within about 10% of itself. The reversal is about
>    twenty times the scatter three repeats of the float32 sweep showed, and
>    Stage B's A6 sweep corroborates it on a different harness with contention
>    held constant — the same sweep run twice changing only the control's dtype,
>    whose ledger moved from 0.90x/0.55x/0.47x/0.43x of affine to
>    0.99x/1.07x/1.08x/1.09x. One bfloat16 sweep exists so far, two repeats are
>    queued, and it ran on a quieter machine than the float32 ones (0.1% median
>    foreign GPU share against 19.0% and 31.1%), so read the direction and not
>    the third decimal; `FINAL_REPORT.md` sets out why contention does not
>    account for it.
>
>    What survives is the *family crossover*, which compares packed
>    configurations against each other and is therefore dtype-consistent within a
>    sweep. The LUT23→GEMM crossover in bfloat16 falls inside the scatter the
>    three float32 sweeps already show — GEMM is the faster family at batch 16 on
>    every tile-256 shape and the slower one at batch 8, in bfloat16 exactly as in
>    float32 — so the batch-16 threshold `select_gemm` and `select_lut23_tile`
>    carry stands as measured, and it is still what ships. Scored on its own sweep
>    the rule reads **38/99 exact, mean 2.63%, worst 29.9%**, inside the 33–40/99
>    and 2.46%–6.07% the three float32 sweeps span. What did *not* survive is the
>    other half of the rule — which row block to run at a given batch — for a
>    reason no dtype re-sweep would have caught; see the superseded banner over
>    "The dispatch rule" below. Decode is unaffected either way
>    — batch 1 takes LUT23 at every threshold — and Stage B's model-level numbers
>    do not depend on this at all, since they load both real checkpoints.

**Verdict: `STAGE_A_PASSED_WITH_A_POSITIVE_OP_LEVEL_PERFORMANCE_RESULT`** —
**withdrawn by correction 3 above.** The correctness half of that verdict stands;
the positive op-level performance half was measured in the wrong dtype and does
not survive re-measurement. `FINAL_REPORT.md` carries the verdict that supersedes
it.

Every Stage A correctness gate passes. The three Metal kernels decode packed
TQ1_G128 storage directly and are bit-exact against their own algorithm's numpy
reference in float32, with zero mismatches across **29,143,424** compared output
elements. At op level the packed path is **at or above `mx.quantized_matmul` at
every batch size measured**, weighted by the tensors the model actually holds:
1.02x to 2.06x of MLX's own 2-bit kernel across batches 1 to 512. *(That second
sentence is the one correction 3 withdraws; it is the float32 pair, and in
bfloat16 the same weighting reads 0.81x to 1.17x.)*

Two findings change the framing the plan assumed, and both are stated up front
because they decide whether Stage B is worth running.

1. **The official MLX 2-bit baseline is already lossless ternary** (A0). The
   storage claim therefore narrows to memory density exactly as the plan's
   risk 1 anticipated: 1.75 vs 2.25 bits/weight, **22.222% smaller**, with no
   accuracy advantage to claim because there is no accuracy gap to close.
2. **The op-level speed result the plan did not assume** (A5). The plan's
   memory-only branch assumed throughput would be at best neutral. At batch 2–8
   the packed kernel beats `mx.quantized_matmul` by a wide margin, peaking at
   **2.704x** on the output head at batch 8. At prefill batches it is ahead by a
   narrow one: 1.27x of the affine kernel on a whole forward pass at batch 16,
   and 1.02x to 1.07x from batch 32 up. Both halves are stated because the second
   is the one an earlier revision of this report got wrong: it read **0.44x** at
   prefill and concluded the ceiling was structural. It was not. See "how the
   prefill result changed".

### Reading the batch axis

Batch here is the number of token positions in one matmul, so the regimes mean
different things to a user and should not be collapsed:

- **batch 1** — one sequence generating one token. The ordinary single-user
  case. Wins are **modest**: 1.01–1.26x on the eight tallest tensors, against
  *losses* of 0.78x on the 1024-row `self_attn.k_proj` and 0.71x on the 48-row
  SSM gates.
- **batch 2–8** — several sequences at once, or speculative decoding. This is
  where the large numbers live: 1.52x to 2.70x on those same eight, with
  `self_attn.k_proj` climbing from 0.98x at batch 2 to 1.44x at batch 8. **The
  2.70x headline belongs here, not to single-stream generation.**
- **batch ≥ 16** — prompt processing. Ahead, but not by much: 1.27x at batch 16
  narrowing to 1.02–1.07x from batch 32 up, on a model-weighted forward pass.
  **Prefill is a parity result with a small margin, not a second headline.**

---

## Environment

Captured programmatically before and after the benchmark run, into
`results/mlx/a5_op_benchmarks.json`.

- Machine: `Mac16,5`, Apple M4 Max, 16 CPU cores (12 P + 4 E), **40 GPU cores**
- Unified memory: 51,539,607,552 bytes; MLX working-set limit 40,200,896,512
- GPU architecture reported by MLX: `applegpu_g16s`, Metal 4
- macOS 26.5.2 (25F84), Xcode 26.2 (17C52), Metal compiler 32023.864
- MLX 0.32.0, MLX-LM 0.31.3, Python 3.12.11, uv 0.9.18
- Thermal state: no thermal warning, no performance warning, no CPU power status
  recorded — before **and** after the run
- Load average at capture: 1.93; the busiest competing processes were
  `WindowServer` (15.7% CPU) and the editor session itself, none using the GPU

---

## A0 — the framing experiment

`prism-ml/Ternary-Bonsai-27B-mlx-2bit`, 2180 tensors, `{group_size: 128,
bits: 2}` affine. Six real tensors were pulled by HTTP range request and
inspected without downloading the shard.

| Tensor | rows × cols | max code | 4th code used | `bias == -scale` | distinct values/group | zero-scale groups |
|---|---:|---:|---|---|---:|---:|
| `layers.0.linear_attn.in_proj_a` | 48 × 5120 | 2 | no | yes | 3 (all 1,920) | 0 |
| `layers.0.linear_attn.in_proj_qkv` | 10240 × 5120 | 2 | no | yes | 3 (all 409,600) | 0 |
| `layers.0.linear_attn.out_proj` | 5120 × 6144 | 2 | no | yes | 3 (all 245,760) | 0 |
| `layers.0.mlp.down_proj` | 5120 × 17408 | 2 | no | yes | 3 (all 696,320) | 0 |
| `layers.3.self_attn.k_proj` | 1024 × 5120 | 2 | no | yes | 3 (all 40,960) | 0 |
| `layers.3.self_attn.q_proj` | 12288 × 5120 | 2 | no | yes | 3 (all 491,520) | 0 |

`bias/scale` was exactly `-1.0` at both extremes on every tensor, and code `3`
never occurred. The official MLX distribution is therefore *not* a min/max
affine quantization of ternary weights — it is exact ternary stored in an affine
container that costs 36 bytes per 128 weights where TQ1_G128 costs 28.

The sample deliberately spans both layer kinds — layer 0 is recurrent, layer 3
is full-attention — and both extremes of row count. It is a **sample**: six of
the model's quantized tensors, checked exhaustively over every one of their
1,886,080 groups, not a scan of all of them. The exhaustive whole-model
equivalent is the frozen earlier result that code `3` never occurs across
26,893,352,960 weights in the GGUF source, and confirming that property on the
MLX distribution itself is a Stage B step.

**Consequence for the claim.** TQ1_G128 is not more accurate than the baseline;
it is bit-identical in logical weights and 22.222% smaller. The plan's verdict
ladder calls this the memory-only branch, and that is the correct label for the
*storage* result. It does not bind the *kernel* result, which A5 measures
separately.

| | affine 2-bit baseline | TQ1_G128 | change |
|---|---:|---:|---:|
| Bytes per 128-weight group | 36 | 28 | −22.222% |
| Bits per weight | 2.25 | 1.75 | −22.222% |
| Quantized payload, 210,104,320 groups | 7,563,755,520 | 5,882,920,960 | −1,680,834,560 |

The 26,893,352,960-weight and 210,104,320-group figures are the frozen numbers
from the earlier exhaustive scan and were not recomputed. Projecting the
repository total — 8,490,506,720 bytes, of which 926,751,200 is non-quantized —
gives an artifact of about 6,809,672,160 bytes, **19.797% smaller at the file
level**. That is a projection from verified per-group arithmetic, not a
measurement; the measured number is a Stage B deliverable.

---

## What was built

| Module | Lines | Role |
|---|---:|---|
| `src/ternel_mlx/kernels.py` | 1336 | the three `mx.fast.metal_kernel` families and their templates |
| `src/ternel_mlx/kernel_gate.py` | 827 | the A4 gate: three references, 138 activation cases per shape |
| `src/ternel_mlx/bench_ops.py` | 730 | A5 paired round-robin benchmark harness |
| `src/ternel_mlx/modules.py` | 353 | `PackedLinear` / `PackedEmbedding` and the dispatch rule |
| `src/ternel_mlx/baseline_fidelity.py` | 248 | A0 |
| `src/ternel_mlx/reference.py` | 230 | the second, structurally independent numpy LUT23 decoder |
| `src/ternel_mlx/remote_safetensors.py` | 177 | range-request tensor reader, so A0 needed no download |
| `src/ternel_mlx/environment.py` | 161 | macOS environment capture |
| `src/ternel_mlx/layout.py` | 147 | per-tensor tile policy and validation |
| `src/ternel_mlx/cli.py` | 117 | `mlx-ternel-generate` |
| `src/ternel_mlx/packed_model.py` | 115 | graph surgery over the mlx-lm `qwen3_5` model |
| `src/ternel_mlx/packing.py` | 90 | tensor → packed arrays |
| Tests | 1,927 | `test_mlx_format`, `test_mlx_kernels`, `test_mlx_kernel_gate`, `test_mlx_modules` |

The four MLX test files: **644 passed in 6.43s**. The whole repository suite,
which also carries the frozen CUDA-era tests: **715 passed in 6.74s**.

### Storage layout

`TILE = 256 if rows % 256 == 0 else rows`, giving

```
codes [tile][group][slot][row_in_tile]   uint8    slot ∈ [0,27)
scales[tile][group][row_in_tile]         uint16   raw little-endian FP16 bits
```

Every row count in the model is divisible by 256 except the two 48-row SSM
gates, which take `TILE = 48`. **Zero padded rows model-wide**, so the artifact
costs exactly 28 bytes per 128 weights. `PackedTensorLayout` rejects any tensor
where `rows % tile != 0` rather than padding it.

The scale bytes are stored as `uint16` raw bit patterns and are never routed
through a float. That is what makes the "every raw FP16 scale bit preserved"
claim mechanical rather than numerical.

### Kernels

- **`tq1_matmul`** — template `<T, G, TILE, BT, PADDED_LUT, SAFE_CLAMP>`. One
  thread per output row; 256 threads cooperate to build the L2/L3 tables for a
  128-weight group, one barrier, then each thread runs the fully unrolled 25
  regular slots plus the tail and applies its scale once per group as
  `fma(as_type<half>(bits), group_sum, acc)`. `BT` accumulators live in
  registers and the 26 loaded code bytes are reused across all `BT` vectors.
- **`tq1_gemm`** — a simdgroup-tiled GEMM the plan did not anticipate, added
  once it was clear `tq1_matmul` could not reach prefill throughput. It feeds
  Apple's matrix units: weights are decoded from packed trits into
  `simdgroup_float8x8` fragments and consumed by
  `simdgroup_multiply_accumulate`. Templated over
  `(batch_block, row_block, k_block, simd_rows, simd_columns)` plus three
  booleans that select which of six forms is compiled — see "the six forms"
  below. Never materialises a dequantized weight matrix: a decoded weight lives
  for one K-block and is discarded. The four shipped tilings cost 20,480, 2,560,
  2,560 and 2,560 bytes of threadgroup memory — the first stages a K-block of
  weights and activations, and the other three hold only their trit table.
- **`tq1_get_rows`** — the embedding gather, decoding a single trit per output
  element straight from packed storage.

30 template instantiations were compiled during the benchmark. **Total JIT
compile cost 11.32 ms; total cold-call cost 21.98 ms; worst single compile
5.38 ms.** That is the whole measured sweep; replaying the shipped dispatch over
all 99 measured points reaches **seven** of them — three LUT23 batch tiles and
the four GEMM tilings. Cold start is not a concern on this platform.

`lut23_bt8_padded` is the one configuration that could not be built: its padded
tables need 52,064 threadgroup bytes against the 32,768-byte limit. It is
recorded in the results as `configurations_unavailable` rather than silently
skipped.

### Changes to the frozen CUDA modules

Eleven files under `src/bonsai_tq1/`, `tests/` and `tools/` are modified. Every
change is behaviour-preserving, because the handoff requires the CUDA evidence
stay reproducible:

- `write_json_atomic` moved from `inspect_model` to `format` verbatim, so the
  MLX modules do not import the GGUF audit tool to write a JSON file. Six call
  sites re-point at the new home.
- `quantile_summary` moved from `v2_reporting` (as `_quantile_summary`) to
  `lut23_reporting` and became public, so A5 reuses it rather than restating it.
- `environment._run` became `run_command`, public for the macOS capture module.
- The literals `242` and `26` in `decode_tq1_blocks` became `MAX_FULL_CODE_BYTE`
  and `MAX_TAIL_CODE_BYTE`, defined as `3**5 - 1` and `3**3 - 1`. Same values.
- `lut23_reorder`'s three layout functions take an explicit `tile=` keyword
  instead of always using `M_TILE`. `reorder_sidecar` passes `tile=M_TILE`, so
  the CUDA sidecar path is unchanged.

That last one had a trap worth recording. The first version of the change also
added a `"tile"` key to the dict `tensor_layout` returns — which is serialised
through `canonical_json_bytes` into the LUT23 sidecar manifest, and therefore
into the artifact's SHA-256. `reports_lut23/FINAL_REPORT.md` pins that hash as
`421ff2ee…4624`. A one-key convenience addition would have silently made a
published result irreproducible. The key was removed, the function now carries a
comment explaining why it must not grow, and a test asserts its absence.

---

## A4 — correctness gates

`results/mlx/a4_kernel_gate.json`, `passed: true`.

Three references, deliberately not sharing an implementation:

| Reference | What it is |
|---|---|
| `lut23` | numpy LUT23 — the kernel's own algorithm **and its summation order** |
| `oracle` | dense float64 matmul of fully decoded weights, capped at 2²⁷ weights |
| `restored` | tile permutation inverted, then the frozen decode-then-dot GEMV from the CUDA work |

A LUT23 kernel against the `lut23` reference sums in the same order and is held
to a 1e-6 relative tolerance — in practice it hit **exactly zero**. Every other
pairing reorders the sum and is held to 1e-5. Each error is normalised *within*
an activation case, never across cases, because one deliberate case uses 1e18
activations and would otherwise dominate every statistic in the file.

Sixteen GEMM tilings are gated. Twelve are chosen to walk the six kernel forms
and the awkward corners of each — the smallest and largest threadgroups, the
staged and register fragment builds, the staged and direct epilogues, the
table-in-threadgroup arms. The other four are the tilings `select_gemm`
actually dispatches, on the principle that a gate covering everything except
what runs proves the wrong thing. That set was three when this report was first
written and is four now; the gate was re-run against the tilings that ship, and
`results/mlx/a4_kernel_gate.json` is that run.

The same principle later added the split-K LUT23 arms, which Stage B introduced
and this section originally predates: `lut23_bt1_padded` and `lut23_bt4_packed`
are each gated at k=2, 40 and 48 as well as unsplit. A K-axis split is a
different summation order, so it is exactly the change a `lut23`-referenced arm
is there to catch, and all six land between 1.985e-07 and 4.145e-07 against all
three references — the same band as the unsplit kernels rather than a wider one.

### Shapes gated

| Tensor | rows × cols | tile | tiles | activation cases | reference rows | full shape |
|---|---:|---:|---:|---:|---:|---|
| `linear_attn.in_proj_a` | 48 × 5120 | 48 | 1 | 138 | 48 | yes |
| `self_attn.k_proj` | 1024 × 5120 | 256 | 4 | 138 | 1024 | yes |
| `linear_attn.out_proj` | 5120 × 6144 | 256 | 20 | 138 | 1024 | no |
| `mlp.gate_proj` | 17408 × 5120 | 256 | 68 | 138 | 1024 | no |
| `lm_head` | 248320 × 5120 | 256 | 970 | 138 | 1024 | no |

138 finite activation cases per shape = 128 deterministic random cases plus ten
edge cases (`zeros`, `ones`, `alternating_signs`, `single_hot_first`,
`single_hot_last`, `single_hot_tail`, `large_positive`, `large_alternating`,
`small_normal`, `mixed_magnitudes`). Five non-finite cases (`all_nan`,
`all_positive_inf`, `single_nan_first`, `single_nan_tail`, `single_inf_last`)
run as their own comparison, since a tolerance is meaningless against NaN and
what is checked there is the propagation pattern instead.

### Results

**350 matmul comparisons over 27,085,184 output elements: 0 mismatches, 0
non-finite pattern mismatches, 10 bit-exact.**

| Kernel | vs reference | activations | worst normalised error | tolerance | min cosine |
|---|---|---|---:|---:|---:|
| `lut23_single` | `lut23` | float32 | **0.000e+00** | 1e-6 | 1.0000000000 |
| `lut23_single` | `lut23` | non-finite | **0.000e+00** | 1e-6 | 1.0000000000 |
| `lut23_single` | `oracle` | float32 | 4.152e-07 | 1e-5 | 1.0000000000 |
| `lut23_single` | `restored` | float32 | 3.752e-07 | 1e-5 | 1.0000000000 |
| `lut23_batched` | `lut23` | float32 | **0.000e+00** | 1e-6 | 1.0000000000 |
| `lut23_batched` | `lut23` | non-finite | **0.000e+00** | 1e-6 | 1.0000000000 |
| `lut23_batched` | `lut23` | float16 | 4.651e-04 | 2e-3 | 0.9999999672 |
| `lut23_batched` | `lut23` | bfloat16 | 3.527e-03 | 1e-2 | 0.9999979446 |
| `lut23_batched` | `oracle` | float32 | 4.152e-07 | 1e-5 | 1.0000000000 |
| `lut23_batched` | `restored` | float32 | 3.752e-07 | 1e-5 | 1.0000000000 |
| `lut23_*` split-K (6 arms) | all three | float32 | 4.145e-07 | 1e-5 | 1.0000000000 |
| `gemm_*` (16 tilings) | `lut23` | float32 | 4.005e-06 | 1e-5 | 1.0000000000 |
| `gemm_*` (16 tilings) | `oracle` | float32 | 3.943e-06 | 1e-5 | 1.0000000000 |
| `gemm_*` (16 tilings) | `restored` | float32 | 2.861e-06 | 1e-5 | 1.0000000000 |

The non-finite rows matter: NaN and Inf propagate through the kernel to exactly
the output positions the reference puts them in, with zero pattern mismatches.
There is no clamping or flushing hidden in the fast path.

**`tq1_get_rows` on the 248320 × 5120 embedding is bit-exact in all three
dtypes** — 686,080 elements each at 134 token indices, tolerance 0.0, including
the tile boundaries 0, 1, 255, 256, 257 and 248319.

That last row was the gate's one real failure during development, and the fix is
worth recording because the naive reading was wrong. bfloat16 initially showed
400,739 mismatches. The kernel was not at fault: a purpose-built Metal probe
comparing `static_cast<bfloat16_t>` against `ml_dtypes` and MLX `astype` over all
63,488 finite fp16 values found **exact** agreement on round-to-nearest-even.
The defect was the gate's own premise — it asserted `(t−1)·scale` is exactly
representable in all three dtypes, which is false for bfloat16, since fp16
carries 10 mantissa bits and bfloat16 holds 7. Computing the reference in the
kernel's own store dtype restored exact equality. Because `t−1 ∈ {−1,0,+1}` and
RNE is sign-symmetric, `round_bf16(scale)·(t−1) == round_bf16(scale·(t−1))`,
so the kernel's order of operations is provably safe here rather than
empirically lucky.

### Scope limits of A4

- Weights are **synthetic tensors at real model shapes**. Gating against tensors
  streamed from the GGUF needs the 7 GB download that the agreed staging places
  after this gate.
- For the three tallest shapes the reference covers a **1024-row whole-tile
  prefix**, not the full row count; `reference_rows_are_full_shape` is true only
  for the 48- and 1024-row tensors. A float64 oracle over 248320 × 5120 is not
  affordable per activation case, and a partial prefix that respects tile
  boundaries tests every code path the full shape would.

---

## A5 — op-level benchmarks

Baseline: `mx.quantized_matmul` at `bits=2, group_size=128` — the representation
`prism-ml/Ternary-Bonsai-27B-mlx-2bit` actually ships and the fastest path MLX
offers for this matrix today.

Method: round-robin arm sampling (not sequential sweeps, which thermal drift
corrupts), 40 trials per arm, 5 warmup samples, `mx.eval` + `mx.synchronize`
discipline, inner-call counts chosen so every sample exceeds 1 ms, medians with
p5/p95, and a paired bootstrap 95% CI at seed 20260809. Every variant's numerical
agreement with the LUT23 reference is recorded alongside its timing, so a fast
wrong kernel cannot post a number.

Scope: **30 configurations across 10 tensor shapes at 11 batch sizes — 110
measured (shape, batch) cases**, each carrying its own interleaved affine
baseline.

### The dispatched configuration versus MLX affine 2-bit

Ratios above 1.00 mean the packed kernel is faster. `G` marks a `tq1_gemm`
dispatch; everything else is `tq1_matmul`.

| Tensor | rows × cols | b=1 | b=2 | b=4 | b=8 | b=16 | b=32 | b=40 | b=64 | b=128 | b=200 | b=512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lm_head` | 248320 × 5120 | 1.26 | 2.12 | 2.49 | **2.70** | 1.40 G | 1.09 G | 1.11 G | 1.07 G | 1.06 G | 1.07 G | 1.16 G |
| `mlp.gate_proj` | 17408 × 5120 | 1.17 | 1.65 | 1.75 | **2.11** | 1.25 G | 1.02 G | 1.04 G | 1.03 G | 1.07 G | 1.05 G | 1.07 G |
| `self_attn.q_proj` | 12288 × 5120 | 1.10 | 1.60 | 1.70 | **1.90** | 1.21 G | 1.01 G | 1.03 G | 1.02 G | 1.04 G | 1.06 G | 1.06 G |
| `linear_attn.in_proj_qkv` | 10240 × 5120 | 1.25 | 1.67 | 1.76 | **1.88** | 1.05 G | 1.01 G | 1.02 G | 1.01 G | 1.03 G | 1.05 G | 1.06 G |
| `linear_attn.in_proj_z` | 6144 × 5120 | 1.04 | 1.52 | 1.58 | **1.81** | 1.34 G | 1.11 G | 1.00 G | 0.96 G | 1.01 G | 1.04 G | 1.06 G |
| `linear_attn.out_proj` | 5120 × 6144 | 1.01 | 1.63 | 1.66 | **1.95** | 1.25 G | 1.05 G | 0.99 G | 1.00 G | 1.03 G | 1.02 G | 1.04 G |
| `self_attn.o_proj` | 5120 × 12288 | 1.05 | 1.56 | 1.78 | **2.09** | 1.32 G | 1.03 G | 1.00 G | 1.02 G | 1.03 G | 1.04 G | 1.07 G |
| `mlp.down_proj` | 5120 × 17408 | 1.17 | 1.80 | 1.77 | **2.19** | 1.43 G | 1.07 G | 1.01 G | 1.02 G | 1.03 G | 1.03 G | 1.07 G |
| `self_attn.k_proj` | 1024 × 5120 | 0.78 | 0.98 | 1.19 | 1.44 | 1.27 G | 1.13 G | 1.01 G | 1.00 G | 1.02 G | 1.01 G | 1.00 G |
| `linear_attn.in_proj_a` | 48 × 5120 | 0.71 | 0.65 | 0.66 | 0.66 | 0.89 | 0.71 | 0.72 | 0.66 G | 0.75 G | 0.81 G | 0.87 G |
| **geometric mean, 9 wide tiles** | | 1.08 | 1.58 | 1.72 | **1.98** | 1.28 | 1.06 | 1.02 | 1.01 | 1.04 | 1.04 | 1.07 |

Selected bootstrap 95% CIs, to show the intervals are tight where the claims are:

| Tensor | batch | config | speedup | 95% CI |
|---|---:|---|---:|---|
| `lm_head` | 1 | `lut23_bt1_padded` | 1.256x | [1.244, 1.265] |
| `lm_head` | 8 | `lut23_bt4_packed` | 2.704x | [2.689, 2.718] |
| `lm_head` | 16 | `gemm_16x128k64s2x4` | 1.399x | [1.385, 1.415] |
| `lm_head` | 512 | `gemm_32x64k32s1x2_frag_epi_tgt` | 1.163x | [1.074, 1.221] |
| `mlp.down_proj` | 8 | `lut23_bt4_packed` | 2.188x | [2.137, 2.230] |
| `self_attn.o_proj` | 8 | `lut23_bt4_packed` | 2.085x | [2.040, 2.107] |
| `mlp.gate_proj` | 16 | `gemm_16x128k64s2x4` | 1.248x | [1.204, 1.286] |
| `mlp.gate_proj` | 512 | `gemm_32x64k32s1x2_frag_epi_tgt` | 1.065x | [1.054, 1.092] |
| `linear_attn.in_proj_qkv` | 128 | `gemm_32x64k32s1x2_frag_epi_tgt` | 1.032x | [1.028, 1.035] |
| `linear_attn.in_proj_a` | 512 | `gemm_128x32k32s4x2_frag_epi` | 0.871x | [0.826, 0.901] |

Across the whole sweep, **all 43 dispatched LUT23 entries carry
`relative_error_vs_lut23_reference` of exactly `0.00e+00`**; the 67 dispatched
GEMM entries range from 1.62e-06 to 5.14e-06, consistent with the reordered
summation A4 measured.

### What one forward pass costs

The table above weights a 48 × 5120 tensor and a 17408 × 5120 tensor equally.
The model does not: the frozen audit in `reports/00_format_audit.md` counts 128
of the second per forward pass and 96 of the first. Multiplying each measured
shape by its count and summing gives the number that actually matters — the
matmul time of one whole forward pass, packed against affine, both dispatched:

| batch | packed | affine 2-bit | **x affine** | 48-row share of packed | `lm_head` share |
|---:|---:|---:|---:|---:|---:|
| 1 | 31.27 ms | 34.71 ms | **1.110** | 7.35% | 3.32% |
| 2 | 41.02 ms | 66.67 ms | **1.625** | 4.62% | 4.19% |
| 4 | 70.42 ms | 121.93 ms | **1.731** | 2.60% | 3.87% |
| 8 | 120.92 ms | 248.93 ms | **2.059** | 1.63% | 4.23% |
| 16 | 154.95 ms | 197.24 ms | **1.273** | 1.52% | 4.34% |
| 32 | 202.87 ms | 210.30 ms | **1.037** | 1.34% | 5.91% |
| 40 | 403.06 ms | 413.85 ms | **1.027** | 0.78% | 5.99% |
| 64 | 388.82 ms | 396.36 ms | **1.019** | 0.93% | 7.85% |
| 128 | 818.98 ms | 856.58 ms | **1.046** | 0.65% | 6.02% |
| 200 | 1425.51 ms | 1486.93 ms | **1.043** | 0.33% | 5.96% |
| 512 | 3330.67 ms | 3562.68 ms | **1.070** | 0.23% | 5.66% |

Ahead at every batch. The 48-row tensors, which lose by 0.66–0.89x individually,
are between 0.23% and 7.35% of that sum, so their loss never turns the total
around. These are *matmul* milliseconds, not end-to-end latency: attention,
normalisation, activation and the KV cache are not in this number, and neither
runtime's non-matmul work is.

One artefact to read correctly: batch 40 is **slower in absolute terms than
batch 64** on both arms — 403 ms against 389 ms packed, 414 ms against 396 ms
affine. That is the batch-block tail. A batch of 40 rows still runs whole
64-row blocks, so a quarter of the multiply is thrown away; 64 rows fill them.
MLX's kernel quantises the same way, which is why the ratio barely moves.

### How the prefill result changed

An earlier revision of this report measured prefill at **0.44–0.96x** and
concluded, from the fact that no configuration in a 14-point sweep reached
parity at any batch from 16 up, that the ceiling was structural. That conclusion
was wrong, and the way it was wrong is the useful part: the sweep varied the
tiling of a fixed kernel, so it could only measure the ceiling of *that* kernel.
The tiling was near-optimal. The kernel was not.

What the old GEMM did per K-block was stage decoded weights through threadgroup
memory and then read them back as matrix fragments. Two costs follow. Every
decoded trit makes a round trip through threadgroup memory before it is
multiplied, and the staging buffer plus the epilogue buffer claim so much of the
32 KiB budget that few threadgroups stay resident per core. The rewrite builds
`simdgroup_float8x8` fragments **directly in registers** from the packed bytes,
so a decoded weight goes to the matrix unit without touching memory, and writes
results **out of the accumulators**, so the epilogue claims nothing.

The economics are the reason this is a tiling-dependent win rather than a free
one. The register build costs one decode per *trit*; the staged fill costs one
per *byte*, and a byte holds five trits. The register form only pays for itself
when a decoded fragment is reused enough times, which is exactly
`TM = batch_block / (simd_rows × 8)` — the number of activation fragments a lane
holds. At `TM = 1` the register form **loses by up to 35%**; at `TM = 4` it wins
by 10% to 100%. That single crossover is what moved prefill from 0.44x to parity
and above, and it is why the shipped tilings are the ones with `TM = 4`.

### The six forms, and what each is worth

Three compile-time booleans select which kernel a tiling runs. None dominates,
so all six are built and the dispatch picks per shape and batch:

| Axis | What it changes | Measured worth |
|---|---|---|
| `direct_fragments` | fragments built in registers vs staged through threadgroup memory | −35% at `TM=1`, +10% to +100% at `TM=4` |
| `direct_epilogue` | results written from accumulators vs through a threadgroup buffer | ±4% at `TM=4`, but it frees the last claim on threadgroup memory, so at 128×256 the staged form needs the whole 32 KiB and the direct form measured 1.15–1.26x it |
| `threadgroup_table` | the digit-major trit table in threadgroup memory vs registers; legal only with `direct_fragments` | the one axis no mechanism settled: 0.976 with row bounds present, 1.011–1.026 with them gone, 0.999–1.025 per tiling with a quarter of rows still below parity |

`k_block` is not one of these but behaves like a fourth: it changes only the
threadgroup footprint and through it residency, and staging a whole 128-weight
group cost the entire 32 KiB budget and ran **three times slower** than staging
a quarter of one.

### What is still slower

Nine dispatched entries at batch ≥ 16 sit below 1.00x, and every one is
resolvable rather than noise:

- **`linear_attn.in_proj_a`, 48 rows** — 0.66x to 0.89x at every batch, seven of
  the nine. A 48-row tile fills 48 of 256 LUT23 threads, and for the GEMM it is
  a launch-count problem rather than a work problem: the shape is small enough
  that dispatch overhead dominates whatever tiling it takes. It is 96 tensors of
  the model and under 1.6% of forward-pass matmul time from batch 16 up.
- **`linear_attn.in_proj_z` at batch 64** — 0.961x, CI [0.944, 0.993].
- **`linear_attn.out_proj` at batch 40** — 0.991x, CI [0.989, 0.998].

Below batch 16 the losses are the same 48-row tensor plus `self_attn.k_proj` at
batch 1 (0.778x) and batch 2 (0.980x).

### The ceiling, and how much of it the rule captures

Taking the best of all 30 measured configurations at each point — the ceiling of
everything built, ignoring what the dispatch rule picks — the geometric mean
over the nine wide tiles is 1.09, 1.59, 1.77, **2.11**, 1.28, 1.06, 1.04, 1.03,
1.04, 1.05, 1.08 at batches 1 through 512. The dispatched row of the table above
reads 1.08, 1.58, 1.72, 1.98, 1.28, 1.06, 1.02, 1.01, 1.04, 1.04, 1.07.

Point by point the dispatched row trails the ceiling by 0.00% at batch 16, then
0.35%, 1.52%, 1.19%, 0.73%, 0.58% and 1.72% — so at prefill the rule is never
more than **1.72%** behind everything built, and the largest gap anywhere is
6.24% at batch 8, where it deliberately declines to dispatch the GEMM — see
below.

### Embedding gather

| Tokens | speedup | 95% CI | packed | affine |
|---:|---:|---|---:|---:|
| 1 | 0.972x | [0.928, 1.013] | 26.67 µs | 25.91 µs |
| 8 | 1.003x | [0.966, 1.078] | 28.69 µs | 28.77 µs |
| 64 | 0.903x | [0.887, 0.938] | 39.03 µs | 35.25 µs |
| 512 | 0.425x | [0.415, 0.445] | 116.24 µs | 49.37 µs |

Parity at 1 and 8 tokens — the CIs straddle 1.0, so parity is the honest word,
not a win — a small resolvable loss at 64, and a clear loss at 512. `tq1_get_rows`
was not part of the prefill work and is unchanged; this is the one op where the
packed format costs throughput at prompt-processing sizes.

### The dispatch rule

> **Superseded — this is the Stage A rule, not the shipped one.** Two changes
> landed after this report. The 128×32 bucket was chosen on float32 and loses in
> bfloat16, and the rule below never reads the row count, which cost a short
> prompt up to 9.29× the affine baseline on a 1,024-row projection inside the
> real graph. What ships is a two-ladder walk keyed on the row count, set out in
> `FINAL_REPORT.md` under "What the kernels are, and how one is chosen"; §B8
> there has the measurement that forced it. Everything below is left as measured
> at Stage A.

```
tile < 256                    → GEMM(128×32,  k32, s4×2, frag+epi)      at batch ≥ 64
tile ≥ 256 and batch ≥ 32     → GEMM(32×64,   k32, s1×2, frag+epi+tgt)
tile ≥ 256 and batch ≥ 16     → GEMM(16×128,  k64, s2×4, staged)
otherwise                     → LUT23 bt1 (batch 1) / bt2 (2–3) / bt4 (≥ 4)
```

plus one refusal: a multi-tile tensor whose row tile is not divisible by the
tiling's row block falls back to LUT23, because a row block straddling two tiles
would read the second tile's codes through the first tile's base pointer.

Three things about this rule are worth stating because they are not what the
previous one said.

**The row count is gone.** The old rule held every tensor under 5120 rows on
LUT23 forever, reasoning that 1024 rows fill only 8 threadgroups of a 128-row
tiling. The winning tilings now use 64- and 128-row blocks against a batch split
several ways, so the same 1024 rows yield 8 to 16 row blocks *times* the batch
blocks. `self_attn.k_proj` beats the fastest LUT23 by 1.14x at batch 16 and
2.08x at 512, resolvably at every batch from 16 up.

> This paragraph is wrong, and it is the reason the whole section is marked
> superseded. The row count is back, and `self_attn.k_proj` is the tensor that
> put it there. `bench_ops` times a shape by enqueueing many independent copies
> of the same matmul, so a dispatch offering 8 threadgroups is submitted 128 at a
> time and the GPU fills from the queue — the one regime in which a tiling that
> cannot fill the machine on its own looks fine. Inside a real forward pass,
> where each matmul waits on the one before it, that same 1024-row projection
> cost **9.29x** the affine baseline. What ships walks a ladder of row blocks and
> takes the first that offers three passes over the GPU's 40 cores, dropping to
> LUT23's K-axis split below two; `k_proj` at batch 16 goes back to LUT23 and
> gains 6.51x. The `select_gemm` docstring carries the rule and
> `tests/test_mlx_modules.py` replays both sweeps through it.

**The family crossover is at batch 16, not 128.** At batch 8 the two families
are a genuine wash: geometric mean 0.956 across the nine wide shapes with six of
nine unresolvable, and the three that do resolve **disagree in sign** —
`mlp.gate_proj` resolves for the GEMM at 1.12x while `self_attn.o_proj` and
`mlp.down_proj` resolve against it at 0.87x and 0.84x. No rule that sees only
the tile and the batch can satisfy all three. Dispatching the GEMM there was
scored and costs 1.116x of the per-case oracle on a forward pass against 1.069x
for staying on LUT23, so it stays on LUT23. At batch 16 the same geometric mean
is 1.380 with **nothing** unresolved.

**Two GEMM buckets, not one.** `gemm_16x128k64s2x4` was the fastest
configuration measured at **all nine** wide shapes at batch 16;
`gemm_32x64k32s1x2_frag_epi_tgt` costs a mean 2.56% against the per-case best
over the 54 wide cases at batches 32–512, worst 14.52%. Neither covers the
other's range.
Holding the narrow-tile rule fixed and forcing the second across the whole range
16–512 costs mean per-case regret of 4.38% against 2.43%, worst 39.66% against
14.52%, and **24.8% of a whole forward pass at batch 16**; forcing the first
costs 20.03% mean, 80.36% worst, and **60.8% of a forward pass at batch 32**.
A third bucket at 512 was scored and improves a forward pass by 0.1%, so it was
not added.

Scored against the per-case fastest of all 30 configurations over all 110
measured points: **exact optimum in 43/110 cases, mean regret 2.43%, worst
14.52%.** The worst case is `linear_attn.out_proj` at batch 512, where the
configuration it loses to is its own twin with the threadgroup trit table
compiled out, and their p5/p95 bands sit inside one another.

> Stage B later found that this sweep was captured while an unrelated MLX
> process held 85% of the GPU, and repeated it twice more. Re-scored against
> the floor of the three, the same rule is optimal in 46/110 cases with mean
> regret 1.98% and the same 14.52% worst case, so the conclusions below stand;
> see the Stage B report for why a steady competitor turned out to distort a
> ranking less than a bursty one.
>
> The denominator is also wrong. Eleven of those 110 points are a 5120×12288
> `self_attn.o_proj` the model has no instance of — `o_proj` is 5120×6144, the
> same shape as `linear_attn.out_proj` — so the census is 99 points, not 110.
> Recomputed on the real census, the rule as it stood here reads 43/99 exact,
> mean 2.04%, worst 14.52%. `ternel_mlx.layout.MODEL_LINEAR_SHAPES` is now the
> single census and fails at import unless its nine shapes account for exactly
> the 497 quantised matmuls a decoded token issues.

Rules were compared on **predicted whole-forward-pass matmul time**, not on
unweighted per-case regret, for the reason the forward-pass table gives. The
reordering matters, and the clearest case is the narrow tile: the two candidate
rules that differ only in which tiling the 48-row tensors take differ by **14
points of worst per-case regret** (28.54% against 14.52%) and by **0.04 points
of forward-pass geometric mean** (1.0304 against 1.0301), because those tensors
are under 1% of the sum.
Ranking on the per-case statistic would have made that the loudest decision in
the sweep; it is the quietest. Against a per-case oracle the rule chosen here
predicts **1.030x** on the geometric mean over the eleven batches; the rule it
replaces predicts 1.426x and LUT23-only predicts 1.570x.

`tests/test_mlx_modules.py` replays the measured points through `select_gemm`
and `select_lut23`, so the thresholds are checked against the file rather than
described by it, and it asserts that batch 8 is the *only* group where the
resolvable family verdicts contradict each other. The replay is now a partition
rather than a sweep over everything, which is the same correction as above seen
from the test side: of the 99 real points it scores 78, replays ten against the
in-situ sweep that chose their row block, and excludes eleven that turn on
occupancy A5's enqueue pattern supplied for free. All three sets are asserted
exactly.

### Negative results, recorded so they are not re-run

Each of these was built and measured, and none of them helped:

| Tried | Outcome |
|---|---|
| Double-buffering / pipelining the decode against the MMA | no gain; the decode is not the stall |
| Cheaper decode arithmetic in the inner loop | ~5%, inside the noise for the shapes that matter |
| Skewing or padding the staging buffer against bank conflicts | no gain |
| Half-precision operands into the matrix units | no gain |
| `direct_epilogue` without `direct_fragments` | no gain; the epilogue is not the constraint at `TM = 1` |
| A barrier-cost hypothesis for the staged form | refuted by measurement |
| `k_block = 128` (staging a whole group) | 3x slower — the residency argument above |
| Passing decoded weights directly rather than through fragments | slower at every tiling |
| `TM > 4` at any `TN`, and `TN < 4` | past the register cliff, or too little reuse |
| Register-lean fused fragment builds | slower than the straightforward build |
| Paired and vectorized inner-loop loads | no gain |
| Threadgroup trit table without removing exact-fit bounds | 0.976 — a loss until the bounds go |
| A dispatch rule keyed on thread count for the table | did not generalise across tilings |
| A third batch bucket at 512 | +0.1% of a forward pass |
| A separate 32–63 bucket using the batch-40 winner | 1.834x of the oracle at batch 32 — much worse |
| Dispatching the GEMM at batch 8 | 1.116x against 1.069x |
| Split-K for the 48-row shape | no gain; the shape is launch-bound, not work-bound |

Two occupancy facts explain the shape of the rule. GEMM threadgroup count is
`ceil(rows/row_block) × ceil(batch/batch_block)` against 40 GPU cores, so a
16×128 tiling on 5120 rows lands on 40 threadgroups exactly at batch 16. And
there is a **register cliff** between 44 and 50 simdgroup-matrix registers:
holding the output block fixed and varying only how the simdgroups are spread
across it moved runtime by **8.7x** in the earlier sweep. Tiling choice here is
dominated by register pressure, not by arithmetic intensity.

---

## A limitation in the A5 file that the reader must know about

Each configuration in `a5_op_benchmarks.json` was timed against **its own**
interleaved affine baseline run. That is the right design for reporting a
headline speedup — every ratio in the tables above is a properly paired
measurement — and it is the wrong instrument for comparing two of *our* kernels
to each other, because their packed times come from runs minutes apart.

This was not a theoretical worry. The file's own p5/p95 criterion flags
`mlp.down_proj` at batch 4 as a resolvable **11.18%** regret for batch tile 4
against batch tile 2 — bands [209.52, 228.65] µs and [234.48, 253.16] µs, with
no overlap at all. A head-to-head paired re-measurement — both tiles as the two
arms of one run, 120 trials, batches 4 and 8, all nine wide-tile shapes —
**resolved 0 of 18 points and returned 0.966x on that case, the opposite sign.**
At the smallest shape, `linear_attn.in_proj_a` at batch 64, the identical affine
workload spanned **2.34x** across the thirty runs of it in the file — 24.63 µs
to 57.72 µs for the same work.

Consequences, all of them already applied:

- Configuration ranking in this report is done on **absolute packed time**, never
  on the affine ratio.
- The `LUT23_MAX_BATCH_TILE = 4` threshold is documented as **not shown faster**
  than a tile of 2. It is 1.1% ahead on the geometric mean of the head-to-head
  run and has the lower mean regret over the sweep, and that is the whole
  justification.
- The dispatch test asserts a **mean** regret bound of 3% and a deliberately
  loose 20% max, because a tight max would be asserting on noise. The rule's
  measured worst case is a configuration losing to its own twin, on bands that
  overlap.
- The family test judges per (tile class, batch) group rather than per case,
  because that is the granularity `select_gemm` decides at, and it pins the one
  group whose resolvable verdicts disagree with each other.

Speedups against the affine baseline are unaffected. Only our-kernel-versus-
our-kernel comparisons are.

---

## Deviations from the plan

| Plan | What was built | Why |
|---|---|---|
| "Kernel 1 (gemv)" and "Kernel 2 (batched gemm)" as separate kernels | one `tq1_matmul` source, `BT=1` and `BT>1` template instantiations | the batched path is the same algorithm with more accumulators; two sources would have been two things to keep in sync |
| two matmul kernels | a **third**, `tq1_gemm` | `tq1_matmul` at any `BT` could not approach prefill throughput, and the ceiling turned out to be the algorithm rather than the tiling: one thread per output row cannot feed Apple's matrix units at all. Decoding packed trits straight into `simdgroup_float8x8` fragments and letting `simdgroup_multiply_accumulate` do the arithmetic took prefill from 0.44–0.96x to 1.02–1.27x of the affine kernel on a model-weighted forward pass |
| sequential configuration sweep | round-robin arm sampling | sequential sweeps are corrupted by thermal drift across the run |
| BT sweep decided on threadgroup budget | decided on the **register cliff** | the 8.7x spread from simdgroup spread alone dominates the threadgroup-memory argument. The same cliff, at 44 to 50 simdgroup-matrix registers, is what decides the `tq1_gemm` tilings too |
| A4 gates on synthetic **plus a handful of real** tensors | synthetic only | real tensors need the 7 GB download that the agreed staging puts after this gate |
| A4 reference over full shapes | 1024-row whole-tile prefix on the three tallest | a float64 oracle over 248320 × 5120 per activation case is not affordable |
| — | the A5 config-vs-config limitation above | discovered during the dispatch reconciliation, not anticipated |

The plan's `#pragma unroll 1` concern did not materialise: templating `G` as a
compile-time constant gave full unroll without intervention.

---

## What Stage B still needs

Nothing in Stage A has been run against the real model. Everything below is
unstarted and gated on review of this report.

1. Download and hash-verify the pinned GGUF (7,165,121,600 bytes, SHA-256
   `868c1171…31757`) and the official MLX repo.
2. Re-run the frozen audit and sidecar chain; abort if 26,893,352,960 weights /
   210,104,320 groups / 0 mismatches changes.
3. Stream-pack to safetensors shards with no model-sized array materialised.
4. Cross-check the 353 non-quantized tensors against their GGUF F32
   counterparts under both norm conventions and both conv1d layouts; a tensor
   matching neither is a hard failure.
5. Emit `config.json` (text-only, `quantization` omitted, `model_file` set),
   `ternel_manifest.json`, and safetensors metadata.
6. Exhaustive verify: zero logical-ternary mismatches over 26,893,352,960
   weights, zero scale-bit mismatches over 210,104,320 groups, 498/498 tensors.
7. Full-graph gate: teacher-forced logits and greedy-token agreement against the
   official MLX 2-bit repo through stock mlx-lm, reporting exact storage parity
   separately from cross-runtime drift.
8. Full-model benchmarks and unified-memory accounting, with the anti-cheat
   assertions that summed packed bytes equal expected payload bytes and no
   model-sized float array exists in the graph.

**A5 has already answered the plan's risk 2 at op level, and the answer is
positive.** *(Withdrawn by correction 3: in bfloat16 this answer is neutral, not
positive.)* Weighting each measured shape by how many tensors of that shape one
forward pass contains, the dispatch rule is at or above `mx.quantized_matmul` at
every batch tested — 1.11x at batch 1, 2.06x at 8, 1.27x at 16, and 1.02x to
1.07x from 32 through 512 — and within roughly 1% of the best of all thirty
configurations built. Two op-level losses remain named and unfixed: the 48-row
`ssm_alpha`/`ssm_beta` shape, where the dispatched kernel runs 0.66x to 0.89x of
affine at prefill batches, and `tq1_get_rows`, which falls to 0.43x at a
512-token gather.

The full-model consequence is still a Stage B measurement. These are matmul
ratios; a forward pass is not only matmuls, and how prompt processing and token
generation compose out of them depends on norms, attention, the recurrent state
update and the sampler — none of which per-op timings can settle.

---

## Reproduction

```bash
uv sync
uv run pytest tests/test_mlx_format.py tests/test_mlx_kernels.py \
              tests/test_mlx_kernel_gate.py tests/test_mlx_modules.py
uv run python -m ternel_mlx.baseline_fidelity   # A0  → results/mlx/a0_baseline_fidelity.json
uv run python -m ternel_mlx.kernel_gate         # A4  → results/mlx/a4_kernel_gate.json
uv run python -m ternel_mlx.bench_ops           # A5  → results/mlx/a5_op_benchmarks.json
```

Raw machine-readable results are under `results/mlx/`. No model artifact has
been produced and nothing has been published.
