# Ternel MLX/Metal — final report

**Verdict: a memory-only result.** Every gate passes and the memory win is real,
exact, and reproduced at the top of the stack — Ternary Bonsai 27B runs on Apple
Silicon in **22.19% less memory** than its own official MLX 2-bit distribution
(5.89 GB against 7.57 GB of live allocation), emitting **token-for-token
identical output** under greedy decoding. It is slower: token generation runs at
**0.61–0.72×** the baseline's rate and prompt processing at **0.80–0.90×**,
intervals excluding 1.0 throughout. That is the honest shape of the result, and
it is the middle rung of the three this phase set out to choose between — not
"viable distribution", which required throughput to be competitive, and not
"not qualified", which no gate forced.

The 26,893,352,960-weight artifact verifies bit-exactly against both the frozen
sidecar and the source GGUF, and nothing here restates the CUDA phase's numbers
as Metal ones.

---

## What this is

`TQ1_G128` stores 128 ternary weights in 28 bytes — two raw FP16 scale bytes, 25
base-3 bytes of five trits each, and a tail byte of three — for **1.75
bits/weight**. Ternel already proved that PrismML's Ternary Bonsai 27B re-encodes
into it losslessly, and shipped that through llama.cpp/CUDA, where it bought a
real 17.29% model-buffer saving and **no speed win**.

This phase moved the format to Apple Silicon: three custom Metal kernels that
execute packed storage **directly**, an mlx-lm-loadable checkpoint that never
materialises a dequantised copy of any weight, and honest numbers against the
model's own official MLX distribution.

The comparison throughout is `prism-ml/Ternary-Bonsai-27B-mlx-2bit` — the same
model, published by its authors, `{group_size: 128, bits: 2}` affine = **2.25
bits/weight** — running through stock `mlx_lm`. Stage A0 settled what that
baseline is, and the answer narrowed the claim before anything expensive was
built: **the official 2-bit checkpoint already stores exact ternary weights.**
MLX affine dequantises as `w = scale*q + bias`, and this checkpoint stores
`bias == -scale` with `q ∈ {0,1,2}`, which is exactly `{-s, 0, +s}`. So there is
no accuracy gap for TQ1_G128 to close. What remains to be claimed is memory
density — 1.75 against 2.25 bits/weight, exactly 7/9, **22.222% smaller** — and
whatever the kernels do to speed.

Both checkpoints therefore hold the **same weights**, verified elementwise in
Stage B4 rather than assumed: 497 of the 498 quantised tensors are bit-identical,
and the one exception differs in a single row of one token's embedding, at a
magnitude of `2⁻²³`. Every numerical difference reported below is kernel
accumulation order, not quantisation error. That is an unusually clean setting
for a correctness gate and the report leans on it.

---

## Environment

Captured programmatically before and after every measured run, into each
`results/mlx/*.json`.

- Machine: `Mac16,5`, Apple M4 Max, 16 CPU cores (12 P + 4 E), **40 GPU cores**
- Unified memory: 51,539,607,552 bytes; MLX recommended working set
  40,200,896,512 bytes; max buffer 30,150,672,384 bytes
- GPU architecture reported by MLX: `applegpu_g16s`, Metal 4
- macOS 26.5.2 (25F84), Xcode 26.2, Metal compiler 32023.864
- MLX 0.32.0, MLX-LM 0.31.3, Python 3.12.11, uv 0.9.18
- Thermal state: no thermal warning and no performance warning, before **and**
  after each run

---

## Sources, pinned

| What | Identity |
|---|---|
| Weights | `prism-ml/Ternary-Bonsai-27B-gguf` @ `abbae723028d71be674e71e1a71201a6f43fab22`, `Ternary-Bonsai-27B-Q2_0.gguf` |
| GGUF bytes | 7,165,121,600, SHA-256 `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757` — verified after download, not trusted |
| Non-quantised tensors and tokenizer | `prism-ml/Ternary-Bonsai-27B-mlx-2bit`, `model.safetensors` (8,490,785,104 bytes, 2,180 tensors) |
| Baseline runtime | stock `mlx_lm` 0.31.3 on that same checkpoint |

The two sources are used for disjoint halves of the model and the split is
evidence-backed, not stylistic: every quantised weight comes from the GGUF, and
every non-quantised weight comes from the official MLX checkpoint **after** being
compared numerically against its GGUF F32 counterpart (Stage B4). That keeps the
artifact traceable to the pinned GGUF hash while inheriting mlx-lm's own naming,
normalisation and conv1d conventions rather than guessing at them.

---

## The chain, stage by stage

### B1–B2 — the frozen chain, re-run and unchanged

The GGUF audit and the lossless re-encode are frozen work from the earlier phase.
They were re-run against the freshly downloaded file, and the gate was that
**nothing may have changed**:

| Quantity | Frozen | Re-run |
|---|---:|---:|
| Q2_0 tensors | 498 | 498 |
| Weights | 26,893,352,960 | 26,893,352,960 |
| 128-weight groups | 210,104,320 | 210,104,320 |
| Weight mismatches | 0 | 0 |
| Scale-bit mismatches | 0 | 0 |

The sidecar is 5,883,168,530 bytes, SHA-256
`79c1e029cf1566d40b13fc3561f7a2dd49bca84920e3bfb0d2e55cd2cde6421b`; its payload
is 5,882,920,960 bytes, which is exactly 28 bytes × 210,104,320 groups with
nothing left over. Against the Q2_0 payload of 7,143,546,880 bytes that is a
17.647% reduction — the number the CUDA phase reported, reproduced here as a
precondition rather than a result.

### B3 — the MLX pack

498 quantised tensors were streamed from the GGUF one at a time, re-encoded, and
written into the tiled layout the Metal kernels read:

```
codes [tile][group][slot][row_in_tile]   uint8    slot ∈ [0,26)
scales[tile][group][row_in_tile]         uint16   raw FP16 bits
```

with `TILE = 256` where `rows % 256 == 0` and `TILE = rows` otherwise. Every
tensor in this model has rows in {248320, 17408, 12288, 10240, 6144, 5120, 1024,
48}; only the 48-row `ssm_alpha`/`ssm_beta` pair is not a multiple of 256, and it
takes `TILE = 48`. The result is **zero row padding across the entire model** —
the artifact's quantised payload is 5,882,920,960 bytes, equal to the sidecar's
and to the canonical 28-bytes-per-128-weights figure, to the byte.

No model-sized array is materialised at any point: each tensor's source mapping
is released before the next is opened. The pack took 22.7 seconds.

The artifact is two safetensors shards totalling 5,888,388,744 bytes — 1,349
tensors: 498 × (`codes`, `scales`) plus 353 plain F16 tensors.

### B4 — the map, checked against the weights themselves

The GGUF calls a tensor `blk.7.ffn_down.weight` and mlx-lm calls it
`language_model.model.layers.7.mlp.down_proj.weight`. A wrong entry in that map
does not raise; it wires a layer's gate projection into its up projection and
produces fluent nonsense. So the map is checked three ways before any byte is
written — counts, shapes, and **values** — and the value check is what makes it
evidence.

Because the official MLX checkpoint stores exact ternary, the value check is
bit-exact rather than statistical. Every one of the 498 quantised tensors was
compared elementwise against its claimed partner in the official checkpoint:

- **497 of 498 tensors are bit-identical** — same codes, same scale bits.
- **1 tensor differs, in exactly one of its 248,320 rows.** Row 178519 of the
  embedding — one token id. In the GGUF that row is ±5.96e-08, a single F16
  *subnormal* ulp either side of zero: scale `2⁻²⁴`, codes drawn from {0, 2},
  never 0. The official MLX checkpoint stores the same row as scale `2⁻²³`, bias
  `−2⁻²⁴` and **every code 1**, which is the constant `+5.96e-08` — the signs are
  gone. Largest weight difference **1.1920928955078125e-07**, which is `2⁻²³`.
  Recorded as *within bound*, never as identical.
- **26,893,352,960 weights** compared, `max_code = 2` throughout — the ternary
  alphabet claim, re-confirmed on the MLX side.

That single row also sharpens Stage A0. A0 sampled tensors and found
`bias == -scale`, which is what makes MLX affine 2-bit exactly ternary. This pass
checked the invariant **in every one of the 210,104,320 groups**, and it holds in
all but **40** of them — the 40 groups of that one embedding row, where the
official quantiser produced `bias = −scale/2` and collapsed a row of subnormal
noise to a constant. So the framing stands with a footnote rather than a
caveat: the official checkpoint is exact ternary everywhere a weight has any
magnitude at all, and Ternel reproduces the GGUF exactly including the row it
does not.

The same pass ran a **discrimination test**: 258 deliberately wrong pairings
(swap `ffn_gate` with `ffn_up`, `ssm_alpha` with `ssm_beta`, `attn_k` with
`attn_v`, `token_embd` with `lm_head`), and **all 258 disagree**. Counts and
shapes cannot separate `ssm_alpha` from `ssm_beta` — 48 tensors of 5120 × 48
each, either way round — so without this the map would rest on a naming
convention. With it, the map rests on 26.9 billion weights.

The 353 non-quantised tensors were compared against their GGUF F32 counterparts
under both norm conventions (identity and `+1`) and both conv1d layouts: **all
353 agree, max absolute difference 0.0** over 2,645,504 elements. The conventions
that matched are recorded rather than assumed — 305 tensors under identity, and
the 48 `ssm_a` → `A_log` tensors under a log/negate transform, with the conv1d
weights reshaped rather than shifted. A tensor matching neither convention was
defined as a hard failure; none occurred.

The only bit-level differences anywhere in the plain half are **98 elements that
differ solely in the sign of zero** — `-0.0` against `+0.0`, an absolute
difference of exactly 0.0. They are counted, classified and reported rather than
rounded away, because "0 mismatches" would have been the easier sentence and the
false one.

**The finding this stage produced.** Names alone were not enough. The two
publications **order the linear-attention value heads differently**: there are 48
value heads grouped over 16 key heads (`repeat = 3`, `value_head_dim = 128`),
llama.cpp lays them out repeat-major and mlx-lm key-head-major, so mlx-lm's value
head `j` is llama.cpp's `(j % 3) * 16 + j // 3`. Eight tensor kinds carry a
value-head axis and are permuted during conversion — `attn_qkv`, `attn_gate`,
`ssm_out`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_a` — which
is 240 of the 498 quantised tensors plus their plain companions. This was
discovered by the bit-exact comparison failing, not by reading either codebase,
and it is proven against all 26,893,352,960 weights. The permutation is recorded
in `results/mlx/b4_crosscheck.json` as an explicit table.

For the column-axis case (`ssm_out`) the reorder must move whole 128-weight
groups or the packed representation could not express it; the code checks that
divisibility and raises rather than silently splitting a group.

### B5 — what the artifact declares

- `config.json` — text-only (`language_model_only: true`), `quantization`
  **omitted** so mlx-lm does not call `nn.quantize` over our packed modules, and
  `model_file: ternel_packed_model.py` so stock `mlx_lm.utils.load_model` loads
  Ternel's `Model` class by itself. No fork of MLX or MLX-LM is needed to run
  this checkpoint; `pip install ternel` and point `mlx_lm` at the directory.
- `ternel_manifest.json` — layout name and version, and for every tensor its
  logical name on both sides, rows, columns, groups per row, tile, byte counts,
  the reorder it took, and both frozen digests (`logical_weight_sha256`,
  `scale_bits_sha256`) whose definitions are reused verbatim from the frozen
  `pack_model`.
- Tokenizer, chat template, licence and notice files carried across from the
  official checkpoint unmodified.

### B6 — exhaustive verify

The artifact was streamed back off disk, the tile reordering inverted, the value-
head permutation inverted, and every weight compared against **both** the sidecar
and the source GGUF, with both frozen digests recomputed per tensor.

| Gate | Required | Measured |
|---|---|---|
| Tensors | 498/498 | 498/498 |
| Logical-ternary mismatches vs GGUF | 0 | **0** over 26,893,352,960 weights |
| Scale-bit mismatches vs GGUF | 0 | **0** over 210,104,320 groups |
| Sidecar byte mismatches | 0 | **0** |
| Frozen digest mismatches | 0 | **0** |
| Plain tensors identical | 353/353 | **353/353**, 2,645,504 elements, 0 mismatches |
| Max full code byte | ≤ 242 | **242** |
| Max tail code byte | ≤ 26 | **26** |
| Max trit code | ≤ 2 | **2** |
| Artifact structure | 1,349 tensors, index complete, nothing missing or unexpected | **complete** |

The two code-byte bounds are not cosmetic. The LUT23 decode computes
`b = (p*57)>>9` and indexes a 27-entry table with it; `p = 255` would give
`b = 28` and read past the table. Legality is enforced **twice, fail-closed** —
at conversion by `decode_tq1_blocks(validate=True)`, and here by a full streaming
scan of every code byte in the artifact — so the fast kernels carry no in-kernel
clamp. A clamped variant exists, template-gated, for fuzz tests only.

Against the official MLX checkpoint the verify reports the same single divergence
B4 found — `token_embd` row 178519, 5,120 codes, 40 scales, 40 biases, gap `2⁻²³`
— and calls it *within bound* rather than *identical*, because it is.

### B7 — the full-graph gate

Both checkpoints hold the same weights but for one row of subnormal noise, so
this stage measures very nearly one thing: whether Ternel's Metal kernels compute
the same function as `mx.quantized_matmul` through 64 layers of a 27B model.

**Thresholds were fixed from first principles before anything was measured.**
bfloat16 carries a 2⁻⁹ ≈ 1.95e-3 mantissa step; amplified over √62 layers of
independent rounding that is ≈ 1.5e-2, so the logit tolerance is **0.02** on a
drift normalised by logit range. KL divergence is gated at **0.01 nats**, greedy
agreement at **0.99**, and the free-running continuation at a **common prefix of
8 tokens** out of 48. Twelve prompts cover factual recall, long-form, reasoning,
Python, C, JSON, French, Chinese, Japanese, degenerate repetition, a long
context, and numerals.

Teacher-forced and free-running are reported separately, on principle. Feeding
both models the same context makes every difference one forward pass of
divergence, which can be gated. Feeding each its own context means that past the
first disagreement the two models are answering different questions, so that is
reported as a common prefix and gated only on its opening.

| Gate | Threshold | Measured |
|---|---|---|
| Greedy agreement, teacher-forced | ≥ 0.99 | **1.000 — 343/343 positions** |
| Unexplained greedy flips | 0 | **0** |
| Max normalised logit drift | ≤ 0.02 | **0.003635** (5.5× inside) |
| Max KL divergence | ≤ 0.01 nats | **4.83e-05** (207× inside) |
| Free-running common prefix | ≥ 8 of 48 | **48 of 48, on all 12 prompts** |
| Packed representation | no dequantised weight in the graph | **clean** |

Not gated, but recorded: max absolute logit error **0.0487**, min cosine
similarity **0.9999948**, mean KL **3.43e-06** nats, max teacher-forced
|ΔNLL| **0.00134**, top-1 overlap **1.000** on every prompt.

| Prompt | tokens | max abs Δlogit | normalised | min cosine | max KL | top-10 | prefix |
|---|---:|---:|---:|---:|---:|---:|---:|
| factual_short | 5 | 0.01816 | 0.00184 | 0.9999997 | 5.82e-06 | 1.000 | 48/48 |
| factual_long | 47 | 0.01786 | 0.00168 | 0.9999994 | 1.36e-05 | 1.000 | 48/48 |
| reasoning | 24 | 0.02421 | 0.00236 | 0.9999996 | 2.42e-05 | 0.996 | 48/48 |
| code_python | 42 | 0.02615 | 0.00157 | 0.9999992 | 1.07e-05 | 1.000 | 48/48 |
| code_c | 31 | 0.03420 | 0.00310 | 0.9999987 | 3.45e-05 | 0.997 | 48/48 |
| markup_json | 29 | 0.02594 | 0.00222 | 0.9999990 | 4.83e-05 | 1.000 | 48/48 |
| multilingual_fr | 9 | 0.02033 | 0.00227 | 0.9999990 | 7.67e-06 | 1.000 | 48/48 |
| multilingual_zh | 3 | 0.02831 | 0.00251 | 0.9999948 | 2.08e-05 | 1.000 | 48/48 |
| multilingual_ja | 13 | 0.03803 | 0.00364 | 0.9999988 | 4.08e-05 | 1.000 | 48/48 |
| repetition | 11 | 0.02206 | 0.00178 | 0.9999994 | 2.26e-05 | 1.000 | 48/48 |
| long_context | 103 | 0.03673 | 0.00213 | 0.9999991 | 1.88e-05 | 1.000 | 48/48 |
| numeric | 26 | 0.04870 | 0.00326 | 0.9999992 | 8.61e-06 | 0.996 | 48/48 |

**On the flip classifier, because getting it wrong is easy.** A greedy
disagreement is only excusable if the two candidates were closer together than
the kernels' typical numerical drift. The first version of this rule compared the
reference's top-2 margin at the flipped position against the error observed *at
that same position* — which is vacuous: reordering two logits `m` apart requires
an error above `m/2` right there, so the rule is satisfied by construction for
every flip it will ever see. It was replaced with a yardstick measured **away**
from the flip: the median, over all positions of all twelve prompts, of the
per-position maximum |Δ| — 0.009731 here — and a flip is excused only if the
reference's own margin is below twice that. The vacuous form is pinned as a
regression test (`test_the_classifier_is_not_satisfied_by_the_flip_it_is_judging`)
so it cannot come back. Nothing needed excusing: there were no flips.

**On what "packed representation" checks.** Not the config's word for itself —
the loaded graph. The census walks every parameter of the live model and requires
that the quantised bytes are `uint8` codes and `uint16` scale bits, that packed
byte totals equal the manifest's payload exactly, that no float parameter exceeds
2²⁰ elements, and that `quantization` is absent from the config. Ternel's largest
float parameter is a 40,960-element conv1d weight; the baseline's is the
79,462,400-element embedding, because MLX's affine quantiser keeps scales and
biases as float16 per group.

| | Ternel | official 2-bit |
|---|---:|---:|
| `uint8` codes | 5,462,712,320 B | — |
| `uint16` scale bits | 420,208,640 B | — |
| `uint32` packed q | — | 6,723,338,240 B |
| `float16` (norms; plus scales/biases on the baseline) | 5,291,008 B | 845,708,288 B |
| **Total parameter bytes** | **5,888,211,968** | **7,569,046,528** |

The quantised halves are 5,882,920,960 against 7,563,755,520 bytes — exactly 7/9,
**22.222%**, the density claim landing on its arithmetic value with no rounding
slack. Including the plain tensors the total is **22.21%** smaller.

### B8 — what it costs to run

Memory first, because it is the result. Each model was loaded into a process of
its own with nothing else resident, generated, and measured:

| | Ternel | official 2-bit | saving |
|---|---:|---:|---:|
| MLX active memory | 5,889,743,880 B | 7,569,472,520 B | **22.19%** |
| parameter bytes | 5,888,211,968 B | 7,569,046,528 B | **22.21%** |
| peak process RSS | 6,447,955,968 B | 8,150,990,848 B | 20.89% |
| load to first token | 1.165 s | 1.912 s | — |

The saving is the format's arithmetic, arriving intact at the top of the stack:
1.75 bits/weight against 2.25 is 7/9, and 22.19% of live MLX allocation is what
7/9 looks like once the 353 unquantised tensors are carried along unchanged. RSS
saves slightly less (20.89%) because a process is more than its weights.

Throughput is the price. Six rounds, arms alternating AB/BA, 30 samples each,
one model per process — the ratio is a paired bootstrap of medians over 100,000
resamples:

| workload | | Ternel | official 2-bit | Ternel/2-bit | 95% CI on the time ratio |
|---|---|---:|---:|---:|---|
| pp32_tg64 | decode | 20.12 tok/s | 27.88 tok/s | **0.72×** | 1.386 [1.342, 1.415] |
| | prefill | 73.27 tok/s | 91.50 tok/s | 0.80× | 1.252 [1.213, 1.307] |
| pp512_tg64 | decode | 21.46 tok/s | 34.27 tok/s | **0.63×** | 1.597 [1.563, 1.606] |
| | prefill | 137.26 tok/s | 152.28 tok/s | 0.90× | 1.108 [1.068, 1.133] |
| pp2048_tg64 | decode | 20.63 tok/s | 34.05 tok/s | **0.61×** | 1.648 [1.627, 1.671] |
| | prefill | 141.68 tok/s | 175.52 tok/s | 0.81× | 1.239 [1.219, 1.253] |

Decode costs 28–39%, prefill 10–20%. Every interval excludes 1.0, so the
direction is not in question; and the gap widens with context, which is the
signature of a decode path whose per-token cost is dominated by walking weights
rather than by the prompt.

Output is identical, not merely close. Under greedy decoding both models produced
the same 64 tokens on all three workloads — `greedy_continuations_identical`
true, common prefix 64/64 — which is what a lossless re-encoding of the same
ternary weights should produce, and the reason the throughput column can be read
as a straight cost rather than a trade.

Neither number was measured on a contended machine: across the twelve timed units
of the isolated benchmark the worst total foreign GPU share was 7.01%, eleven of
the twelve were under 0.7%, and across the paired benchmark's five units the
worst was 0.32%.

**Why fewer bytes did not become more speed.** Decoding one token reads every
quantised weight exactly once, so the arms can be compared on the rate at which
they get those bytes off memory — bytes per token from the census above, tokens
per second measured:

| workload | Ternel | official 2-bit | baseline advantage |
|---|---:|---:|---:|
| pp32_tg64 | 118.4 GB/s | 210.9 GB/s | 1.78× |
| pp512_tg64 | 126.3 GB/s | 259.2 GB/s | 2.05× |
| pp2048_tg64 | 121.4 GB/s | 257.5 GB/s | 2.12× |

That is the whole result in one line: Ternel has 22.2% less to read and reads it
at **less than half the rate**, so the saving is spent twice over. The format is
not what is slow — the storage is 7/9 the size and provably the same weights. The
decode path is what is slow, and it is slow because it is not memory-bound the
way the baseline is: at roughly 120 GB/s it is nowhere near what this machine can
stream, so something other than fetching bytes is setting the pace. The op-level
sweep does not contradict this, it is measured in a different regime — it calls
one ~20 MB tensor in a tight loop, where the weights have somewhere to live
between calls, rather than streaming 5.9 GB once per token.

This is the single number to attack. Closing the bandwidth gap is what would move
the verdict off "memory-only", and unlike the memory result, none of it is
settled by the format.

---

## What the kernels are, and how one is chosen

Stage A has the full derivation and the 110-point sweep behind it; this is the
shape of the thing, for a reader of the results above.

Three Metal kernels, all built through `mx.fast.metal_kernel` with compile-time
template constants so each shape family gets its own specialised library:

- **`tq1_matmul`** — the LUT23 decoder. 256 threads per threadgroup, one thread
  per output row, `BT` activation vectors held in registers. Per 128-weight group
  the whole threadgroup cooperates to build two small tables from the
  activations — `L2[slot][a]` over 9 entries and `L3[slot][b]` over 27, indexed
  by `b = (p*57)>>9` and `a = p − 9b` — then each thread runs 25 fully unrolled
  byte slots plus the tail and applies the group's FP16 scale once with an `fma`.
  **Five ternary weights are never materialised**; the table entry index *is* its
  base-3 digit vector. Accumulation is fp32 throughout.
- **`tq1_gemm`** — the batched path, and the thing the CUDA integration never
  had. It builds `simdgroup_float8x8` fragments **directly in registers** from
  packed bytes and lets `simdgroup_multiply_accumulate` do the arithmetic, then
  writes results straight out of the accumulators. Six compile-time forms
  (`direct_fragments`, `direct_epilogue`, `threadgroup_table`) across four
  tilings.
- **`tq1_get_rows`** — the embedding gather, decoding one trit per output element
  directly from the packed bytes.

The dispatch rule, chosen by scoring candidate rules on *predicted
whole-forward-pass matmul time* rather than on unweighted per-case regret:

```
tile < 256                    → GEMM(128×32, k32, s4×2, frag+epi)       at batch ≥ 64
tile ≥ 256 and batch ≥ 32     → GEMM(32×64,  k32, s1×2, frag+epi+tgt)
tile ≥ 256 and batch ≥ 16     → GEMM(16×128, k64, s2×4, staged)
otherwise                     → LUT23 bt1 (batch 1) / bt2 (2–3) / bt4 (≥ 4)
```

Against a per-case oracle over all 30 configurations built and all 110 measured
points, the rule is exactly optimal in 46 cases, gives up a mean of 1.98% and a
worst case of 14.52% — and the worst case is a configuration losing to its own
twin on overlapping p5/p95 bands. `tests/test_mlx_modules.py` replays all 110 points through the dispatch
functions, so the thresholds are checked against the measurement rather than
described by it.

At op level, weighting every measured shape by how many tensors of that shape one
forward pass contains, the dispatched packed path is **at or above
`mx.quantized_matmul` at every batch size measured**: 1.09x at batch 1, 1.65x at
2, 1.77x at 4, **2.11x at 8**, 1.27x at 16, and 1.02x–1.07x from 32 through 512.

**The whole sweep was run three times, on three days, and those are the medians
of the three.** The repeats exist because these ratios were measured on a
machine in use while the full-model numbers above were not, and one sweep on a
busy desktop is an assertion rather than a measurement. Independently repeating
it is what turns the caveat into evidence:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 40 | 64 | 128 | 200 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lowest of three | 1.08x | 1.63x | 1.73x | 2.06x | 1.10x | 1.03x | 1.02x | 1.02x | 1.04x | 1.03x | 1.04x |
| highest of three | 1.11x | 1.75x | 1.86x | 2.17x | 1.27x | 1.31x | 1.03x | 1.09x | 1.05x | 1.08x | 1.10x |

The headline claim holds in every sweep independently: the weighted ratio never
drops below 1.02x at any batch in any of the three. The batch-8 peak reproduces
within 5% (2.06x, 2.11x, 2.17x) and the small-batch shape is stable. The two
widest disagreements — batch 16 and batch 32 — both come from the single
dirtiest sweep, which ran at a median 31.1% foreign GPU share against 19.0% for
the primary.

Why a paired ratio survives this at all is visible directly: the same 17408x5120
shape was measured once at 5–7% foreign share and once at 69–85%, and the ratio
moved by +3.6% at batch 1, +8.6% at batch 8, and −5.2% at batch 64 — against the
7x that contention did to absolute throughput elsewhere in this report. The
alternation cancels a slowdown both arms take together. So the shape of the
op-level result is trustworthy and the third decimal is not.

---

## Limitations

**Text-only.** The GGUF has no vision weights. Shipping a vision tower from a
second source would break "every weight traces to the pinned hash", so the
artifact declares `language_model_only: true` and drops it. The official
checkpoint's `model.safetensors` is 8,490,785,104 bytes against Ternel's
5,888,388,744 on disk, but **921,460,192 of those bytes are 333 vision-tower
tensors** that stock `mlx_lm`'s `sanitize` discards at load time anyway. Its text
half is 7,569,046,528 bytes on disk — the same figure as its loaded parameters,
to the byte. **The honest on-disk comparison is text against text, and it is the
same 22.2%**; the 30.6% a raw directory listing suggests is not a storage result.

**One machine, one model.** Every number here is an M4 Max with 40 GPU cores
running Ternary Bonsai 27B. Nothing was measured on an M1/M2/M3, on a binned
part, or on any other model, and the dispatch thresholds are tuned against this
GPU's core count and register file.

**Two op-level losses remain, named and unfixed.** The 48-row
`ssm_alpha`/`ssm_beta` shape runs 0.56x–0.85x of affine across the whole sweep —
96 tensors, under 1.6% of forward-pass matmul time — and `tq1_get_rows`, which is
at parity for one token (0.98x) and ahead at eight (1.08x), falls to 0.38x on a
512-token gather. Both are visible in the full-model numbers above rather than
hidden by them.

**No single A5 sweep can rank two Ternel kernels against each other; three of
them can.** Each configuration is timed against its own interleaved affine
baseline, which is right for reporting a speedup and wrong for comparing two of
our own kernels, whose packed times come from runs minutes apart with whatever
foreign load fell between them.

That defect has a direction, not just a spread. The regret a dispatch rule
concedes is measured against the *fastest* configuration timed at each point, so
one rival catching a quiet moment lowers the denominator for everybody, and no
amount of averaging removes a bias. On the same 110 points the three sweeps
report mean regret of 2.43%, 5.16% and 5.76%, with worst cases of 14.5%, 72.0%
and 119.9%.

What corrupts the ranking turns out to be how much the foreign load *varies*,
not how large it is. The sweep captured 2026-08-10, inside the window when an
unrelated MLX process held 85% of the device, is the *cleanest* of the three: a
saturating competitor scales every configuration alike and cancels in a ratio.
The two captured against a desktop compositor bursting between 0.4% and 95% are
the unusable ones.

So the rule is judged on the floor of all three — the fastest time each
configuration was ever seen to achieve at each point. The estimator is a minimum
because the noise is one-sided: a competitor can take GPU time from a kernel and
never give it any, so the smallest of several timings is the one nearest the
uncontended speed. On that floor the shipped rule gives up **1.98% on average
and 14.5% at worst**, the worst case being ssm_out at batch 512 losing to its
own twin configuration, and no point exceeds 20%. The rule is validated; what
remains unresolved is only which of two near-identical twins wins a handful of
individual points. All three sweeps are published alongside this report so the
floor can be recomputed rather than taken on trust.

**Correctness is agreement with MLX, not with the truth.** B6 proves the stored
weights are exactly the GGUF's. B7 proves the graph computes what
`mx.quantized_matmul` computes on those weights, to a tolerance fixed in advance.
Neither is a downstream-task evaluation, and none was run: the two checkpoints
hold identical weights, so there is no quantisation-quality question left for a
benchmark suite to answer.

---

## Deviations from the plan

| Plan | What happened | Why |
|---|---|---|
| Two matmul kernels (gemv, gemm) | three: one `tq1_matmul` source templated at `BT=1` and `BT>1`, plus a genuinely different `tq1_gemm` | one thread per output row cannot feed Apple's matrix units at all; the register-fragment rewrite took prefill from 0.44–0.96x to 1.02–1.27x |
| Non-quantised tensors taken from the official repo after a norm cross-check | same, **plus a value-head permutation the plan did not anticipate** | the bit-exact comparison failed on eight tensor kinds until the repeat-major/key-head-major difference was modelled |
| `monitor_process` from `tools/benchmark_llama_distribution.py` for peak memory | a `ps -o rss=` sampler | the existing one shells out to `nvidia-smi` |
| Stage A gate on synthetic **plus real** tensors | synthetic only in A4; real weights first exercised in B4/B6/B7 | the real tensors need the 7 GB download the agreed staging puts after that gate |
| llama.cpp Metal as an optional cross-runtime check in B7 | not run | the official MLX checkpoint is a stronger reference — bit-identical weights make every difference attributable to kernels alone, which a second runtime would confound |

---

## Reproduction

```bash
uv sync --extra convert

# Stage A
uv run pytest tests/test_mlx_format.py tests/test_mlx_kernels.py \
              tests/test_mlx_kernel_gate.py tests/test_mlx_modules.py
uv run python -m ternel_mlx.baseline_fidelity --result results/mlx/a0_baseline_fidelity.json \
              --repo prism-ml/Ternary-Bonsai-27B-mlx-2bit --revision main \
              --filename model.safetensors --timeout-seconds 120 \
              --tensor language_model.model.layers.0.linear_attn.in_proj_a.weight \
              --tensor language_model.model.layers.0.linear_attn.in_proj_qkv.weight \
              --tensor language_model.model.layers.0.linear_attn.out_proj.weight \
              --tensor language_model.model.layers.0.mlp.down_proj.weight \
              --tensor language_model.model.layers.3.self_attn.k_proj.weight \
              --tensor language_model.model.layers.3.self_attn.q_proj.weight
uv run python -m ternel_mlx.kernel_gate --output results/mlx/a4_kernel_gate.json
# Run three times. Ranking one Ternel kernel against another needs the floor of
# several sweeps, and results/mlx/a5_op_benchmarks_<date>.json are the repeats.
uv run python -m ternel_mlx.bench_ops --output results/mlx/a5_op_benchmarks.json \
              --shape all --batches 1 2 4 8 16 32 40 64 128 200 512 \
              --max-foreign-gpu-share 0.25

# Stage B — GGUF is artifacts/models/Ternary-Bonsai-27B-Q2_0.gguf,
#           sidecar is artifacts/packed/Ternary-Bonsai-27B-Q2_0.gguf.tq1g128
uv run python -m ternel_mlx.convert --gguf <gguf> --sidecar <sidecar> \
              --baseline artifacts/baseline-mlx-2bit --out artifacts/ternel-mlx \
              --max-shard-bytes 4294967296 --activation-dtype bfloat16
uv run python -m ternel_mlx.crosscheck --gguf <gguf> --baseline artifacts/baseline-mlx-2bit \
              --result results/mlx/b4_crosscheck.json --weights-per-chunk 33554432 \
              --max-recorded-rows 8 --max-weight-difference 1.1920928955078125e-07
uv run python -m ternel_mlx.verify --artifact artifacts/ternel-mlx --sidecar <sidecar> \
              --gguf <gguf> --baseline artifacts/baseline-mlx-2bit \
              --result results/mlx/b6_verify.json --weights-per-chunk 33554432 \
              --max-weight-difference 1.1920928955078125e-07
uv run python -m ternel_mlx.graph_gate --artifact artifacts/ternel-mlx \
              --baseline artifacts/baseline-mlx-2bit --output results/mlx/b7_graph_gate.json \
              --continuation-tokens 48 --logit-tolerance 0.02 --max-kl-divergence 0.01 \
              --min-greedy-agreement 0.99 --min-continuation-prefix 8
uv run python -m ternel_mlx.bench_model --artifact artifacts/ternel-mlx \
              --baseline artifacts/baseline-mlx-2bit \
              --output results/mlx/b8_model_benchmarks.json --trials 30 \
              --max-foreign-gpu-share 0.25
uv run python -m ternel_mlx.bench_model --artifact artifacts/ternel-mlx \
              --baseline artifacts/baseline-mlx-2bit \
              --isolated-output results/mlx/b8_isolated_benchmarks.json \
              --rounds 6 --trials-per-round 5 --max-foreign-gpu-share 0.25

# Generation
uvx --from . mlx-ternel-generate --model artifacts/ternel-mlx \
    --prompt "The capital of France is" --max-tokens 64 --temperature 0.0 --no-chat-template
```

Every stage writes machine-readable JSON under `results/mlx/` and fails closed on
any mismatch. The `--trials 30` floor is not decorative: `paired_bootstrap_ratio`
refuses fewer, because an interval built from a handful of samples is decoration.

`--max-foreign-gpu-share` has no default, and neither timing stage will start
without it. Every Metal client on Apple silicon appears in the IO registry as an
`AGXDeviceUserClient` carrying `accumulatedGPUTime`, so sampling it around each
timed unit gives that unit's exact share of the device spent on somebody else's
work; above the ceiling, the run raises rather than reports. What is compared
against the ceiling is the share taken by every other process together, not by
the worst single one: these counters partition the device rather than overlapping
it — summed across all clients over a 20-second window on a busy desktop they
came to 99.97% — so four rivals at a fifth each would leave a fifth of the
machine while none of them tripped a 25% bar. This exists because
a benchmark was once run against a machine that was 85% occupied by an unrelated
MLX process, and every guard in the harness passed it: the arms still agreed
token for token, the bootstrap intervals were still tight, and the throughput was
wrong by 7x. Determinism and precision are not the same thing as validity.

The `0.25` is measured, not chosen. A desktop cannot reach zero: the macOS
compositor is a Metal client like any other, and on an idle machine with a live
screen it holds 17.70% of the device. Under load it holds less, because a
saturating compute process crowds it out — across a calibration sweep of five
timed units the busiest foreign client peaked at 8.82%. The ceiling sits above
both floors and 3.4x below the contamination that caused the incident, in a
region of the distribution nothing legitimate occupies. It is a tripwire rather
than the evidence: the measured foreign share of every timed unit is recorded in
`gpu_contention` whether it trips or not, so a reader can check what the machine
was actually doing instead of trusting that a threshold was not crossed. Note the
limit of the instrument — it measures time occupancy, not bandwidth, so a
compositor's short bursts and a model's long kernels are not distinguished by it.

The ceiling earns its keep. It voided an op-level sweep attempted on a desktop
that was in use, where the compositor alone held 44% of the device; a saturating
matmul loop measured on that same desktop was left 37.92% of the GPU it had asked
for. The numbers reported here were not: across the twelve timed units of the
isolated benchmark the worst total foreign share was 7.01%, eleven of the twelve
were under 0.7%, and across the paired benchmark's five units the worst was
0.32%. The one unit that saw any real competition is round 1's candidate arm —
the packed arm, so the alternation charged the interference to the side it would
count against, not the side it would flatter.
