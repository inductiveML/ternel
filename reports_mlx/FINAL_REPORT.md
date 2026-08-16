# Ternel MLX/Metal — final report

**Verdict: the memory win is real, and token generation has reached parity;
prompt processing has not.** Every gate passes. Ternary Bonsai 27B runs on Apple
Silicon in **22.19% less memory** than its own official MLX 2-bit distribution
(5.89 GB against 7.57 GB of live allocation), emitting **token-for-token
identical output** under greedy decoding.

Token generation now runs at **0.89–0.96×** the baseline's rate — at a short
prompt it is not separated from it, the interval containing 1.0 and the
co-resident cross-check landing just the other side of parity — where before the
K-axis split it ran at 0.61–0.72×. Prompt processing runs at **0.64–0.87×** and
is the weaker half: the two longer prompts are close to where they were, but
short-prompt prefill fell from 0.80× to 0.64× when the split landed. That
regression is explained and partly repaired. It was never a cost of the format or
of the split — it was the dispatch rule choosing a tiling from the tile and the
batch while never reading the row count, so a 1,024-row projection at batch 32
was handed sixteen threadgroups on a 40-core GPU and ran 9.29× the affine
baseline while a 248,320-row one ran 1.11×. The rule now reads the row count.
Measured old-rule-against-new inside the real graph, that is worth **1.076× on a
short-prompt prefill pass**, and it removes a 1.4% regression the first draft of
the rule had introduced at long prompts. It does not show at the workload: pp32
prefill reads 0.640× on the shipped build against 0.641× before the fix, because
the prefill pass is a little under half of the time-to-first-token this report
measures, and 3.7% of it sits at the edge of that measurement's resolution. Both
figures are in §B8 and neither is rounded toward the other.

That splits the criterion this phase set out to decide on. "Viable distribution"
required throughput to be competitive; generation now is and prompt processing is
not, so this is no longer the plain "memory-only" the pre-split kernel earned,
and not yet the top rung either. Nothing forced "not qualified" — every gate
passes, and the output is identical rather than merely close.

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
- Process identities in the captures are redacted at capture time to system
  software and this repository's interpreter (everything else reads
  `[redacted third-party process]`); pids, GPU shares, and every measured
  number are untouched

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
  Ternel's `Model` class by itself. The named file is **self-contained**:
  `ternel_mlx.model_file` generates it from the package sources — `layout`,
  `kernels`, `modules`, `packed_model` concatenated in dependency order, the
  package-internal imports stripped and their values inlined, with the name
  accounting failing closed — so the only packages it imports are mlx and
  mlx-lm. No fork of MLX or MLX-LM is needed to run this checkpoint, and
  neither is this repository; `pip install mlx-lm` and point `mlx_lm` at the
  directory. It is not a second definition of the format free to drift from
  the verified one: the package remains the only definition, and a test
  regenerates the file against the shipped artifact so an edit to any kernel
  without a re-emission fails the suite. A subprocess test loads the emitted
  file through the exact `spec_from_file_location` call mlx-lm makes, with
  `ternel_mlx` and `bonsai_tq1` blocked from importing, and the manifest
  records the file's sha256 plus the sha256 of each source it was emitted
  from. One consequence is load-bearing: mlx-lm executes the file without
  registering it in `sys.modules`, where `@dataclass` under PEP 563 string
  annotations crashes, so the emitted file drops
  `from __future__ import annotations` and its sections rely on dependency
  order for eager annotation evaluation.
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

This gate has been re-run against every tiling change since, most recently the
dispatch rule §B8 describes, and a different tiling is a different reduction
order — so it is a real re-measurement rather than a formality. Every figure in
this section moved in the **fourth significant figure or later**: max KL
4.8275e-05 to 4.8251e-05, min cosine 0.99999480193 to 0.99999480225, the
characteristic drift 0.009731 to 0.009742. Which tiling sums a row does not
detectably change what the model says.

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
per-position maximum |Δ| — 0.009742 here — and a flip is excused only if the
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
| peak process RSS | 6,489,423,872 B | 8,150,827,008 B | 20.38% |
| model load, weights resident | 1.563 s | 1.352 s | — |

The saving is the format's arithmetic, arriving intact at the top of the stack:
1.75 bits/weight against 2.25 is 7/9, and 22.19% of live MLX allocation is what
7/9 looks like once the 353 unquantised tensors are carried along unchanged. RSS
saves slightly less (20.38%) because a process is more than its weights.

The top two rows are counted and the bottom two are sampled, which is worth
stating because they behave differently. Across the four paired runs this model
has had — before the K-axis split, at an intermediate split threshold, after the
narrow-tile tiling correction, and at the shipped build — the counted rows are
**byte-identical every time**, as they should be: these changes decide how a
matmul is dispatched, not what is stored. Peak RSS moves 0.64% on the Ternel arm
and 0.34% on the baseline. Model load spans 1.165–1.631 s and 1.352–1.912 s,
swings of 40% and 41%, and in three of the four runs Ternel loaded *slower* than
the baseline. That looked like a cost of split-K's extra JIT'd kernel
instantiations, and it is not: each arm's own spread across the four runs, 0.47 s
and 0.56 s, is wider than the gap between the arms in three of them, 0.19 s,
0.21 s and 0.21 s, and the sign of that gap changes. The row is published as a
measurement of this run, not as a comparison.

Throughput is the price. Six rounds, arms alternating AB/BA, 30 samples each,
one model per process — the ratio is a paired bootstrap of medians over 100,000
resamples:

| workload | | Ternel | official 2-bit | Ternel/2-bit | 95% CI on the time ratio |
|---|---|---:|---:|---:|---|
| pp32_tg64 | decode | 27.33 tok/s | 28.45 tok/s | **0.96×** | 1.039 [0.997, 1.063] |
| | prefill | 60.95 tok/s | 95.27 tok/s | 0.64× | 1.557 [1.506, 1.600] |
| pp512_tg64 | decode | 32.04 tok/s | 34.15 tok/s | **0.94×** | 1.066 [1.051, 1.090] |
| | prefill | 131.28 tok/s | 150.28 tok/s | 0.87× | 1.141 [1.112, 1.170] |
| pp2048_tg64 | decode | 30.34 tok/s | 34.20 tok/s | **0.89×** | 1.127 [1.103, 1.147] |
| | prefill | 141.54 tok/s | 173.04 tok/s | 0.82× | 1.223 [1.197, 1.242] |

Decode now costs 4–11%, and at a short prompt it is still not separated from the
baseline: pp32's interval is [0.997, 1.063] and contains 1.0, barely, where the
paired run below puts the same workload on the other side of parity. The other
two exclude it, and the deficit widens with context — 0.96×, 0.94×, 0.89× —
which remains the signature of a decode path whose per-token cost is set by
walking weights rather than by the prompt.

Prefill is the worse half, and short-prompt prefill moved the wrong way when the
K-axis split landed: it was 0.80× before and 0.64× after. The two longer prompts
did not really move — the affine arm also lost a uniform ~3.7% on prefill between
those two runs, and correcting for that machine drift leaves pp2048 flat and
pp512 down about 3.5%, against a real ~20% at pp32. **That regression is
explained. It was the tiling the dispatch rule picked, and the rule was blind to
the thing that decides it.** What follows is the diagnosis, then the fix, then
what the fix is and is not worth — the last of which is the part that did not go
the way the diagnosis predicted.

The chain was measured end to end at batch 32, inside the real graph rather than
at op level, against the rule as it stood then. A bare forward pass is 1.74×
(253.5 ms against 145.6), and the
linear layers carry 105.4 of that 107.9 ms gap — 98%: with every linear replaced
by a shape-preserving stub the two arms are 16.7 ms and 14.2 ms, a difference of
2.5. Restoring one shape at a time, and differencing against the all-stubbed
reference inside each round, gives each shape's own contribution. The nine do not
share a deficit, and their ordering is not the ordering of their sizes:

| shape | rows | threadgroups | waves of 40 | packed/affine |
|---|---:|---:|---:|---:|
| `self_attn.k_proj` | 1,024 | 16 | 0.40 | **9.29×** |
| `linear_attn.in_proj_z` | 6,144 | 96 | 2.40 | 4.74× |
| `self_attn.q_proj` | 12,288 | 192 | 4.80 | 3.58× |
| `linear_attn.out_proj` | 5,120 | 80 | 2.00 | 2.17× |
| `mlp.down_proj` | 5,120 | 80 | 2.00 | 2.16× |
| `linear_attn.in_proj_qkv` | 10,240 | 160 | 4.00 | 1.36× |
| `mlp.gate_proj` | 17,408 | 272 | 6.80 | 1.21× |
| `lm_head` | 248,320 | 3,880 | 97.0 | 1.11× |

(`linear_attn.in_proj_a`, 48 rows, already dispatches split-K rather than a GEMM
and sits at 3.32×.) The nine together are 244.4 ms against 128.2, or 1.91×,
against 1.80× for the whole linear block measured directly — the residual is the
usual cost of summing nine differences instead of taking one.

A threadgroup owns one `batch_block × row_block` block of the output, so a
dispatch offers `ceil(rows/row_block) × ceil(batch/batch_block)` of them to a
40-core GPU. `select_gemm` keyed on the tile and the batch and never looked at
the row count, so at batch 32 all of these took the same 32×64 tiling — which
turns `k_proj`'s 1,024 rows into sixteen threadgroups, 40% of a single wave,
leaving 24 of the 40 cores idle for the dispatch's entire duration. The affine
baseline is not paying that, and 9.29× is what it cost. `lm_head` fills 97 waves
and is at 1.11×. This is why the regression is worst at the shortest prompt:
batch is the other factor in the threadgroup count, so a short prompt starves the
same dispatch further, and by pp512 the deficit has fallen to 1.147×.

**The fix ships, and it reads the row count.** Batch picks one of two ladders;
the walk down it stops at the widest rung that still fills the GPU, and when no
rung can, the GEMM is refused outright and LUT23's K-axis split takes the tensor.
The section below states it in full. At batch 31 that moves four of the nine
shapes: `k_proj` drops to the split, and `in_proj_z`, `out_proj` and `down_proj`
drop from a 128-row block to a 32-row one. At 511 and 2047 it moves one — the
48-row alpha projection — because at those batches both rules already choose the
same 32×64 tiling on all seven wide shapes.

Old rule against new, measured directly: both compiled into one process, every
one of the 497 packed linears pinned to what each rule chooses, ten rounds at
each batch the model actually prefills at, arms alternating within the round.
A prefill forward pass runs **1.076× faster at batch 31**, 1.0044 at 511 and
1.0014 at 2047. The first is a real effect — the two round orders agree to 0.26%
and their ranges do not touch. The last two are the result that was being looked
for: this ladder's first draft topped out at a 128-row rung and made those two
batches 0.988× and 0.986×, *slower* than the rule it replaced, and that
regression is gone.

Reading those two as "gone" rather than merely "small" needs the thing that run
turned up: **at long prompts, order dominates.** Whichever arm ran second in a
round was slower, and the ratio computed from old-first rounds differs from the
one computed from new-first rounds by 8.5% at batch 511 and 4.3% at 2047 —
against 0.26% at batch 31. A 1.4% effect fits inside a swing that size several
times over, so at those two batches only the order-balanced figure means
anything, and it is the one quoted. This also retires an earlier discrepancy: a
per-shape fit put the 128-row rung's cost at 2047 at 0.18% while a single-order
whole-pass run put it at 1.4%, and an order swing of ±4% swallows that difference
whole. The two measurements never actually disagreed.

**That 1.076× does not resolve in the table above, and the table above is what
gets published.** Short-prompt prefill reads 0.640× on the shipped build against
0.641× before the fix and 0.634× at an intermediate build — three isolated runs,
one protocol, spanning all three builds, with no systematic movement. Absolute
times did move: packed pp32 time-to-first-token went 566.5 ms → 525.0 ms between
the first and the last, −7.3%, which taken alone reads as the fix arriving. But
the affine arm moved with it, 363.1 ms → 335.9 ms, −7.5%, on code that did not
change at all. Machine drift moves both arms; only the ratio holds still. That is
why every headline number in this report is a ratio and not a rate.

The arithmetic says this is under-resolution rather than absent. The prefill
forward pass at batch 31 is 256 ms of that 525 ms time-to-first-token — a little
under half; the remainder is everything else the reported metric contains,
prompt-cache construction and the decode step that produces the first token among
it, and this report has not broken that remainder down. So 7.6% of the pass is
about 3.7% of the published number, against a bootstrap interval of ±3% on it.
An effect that size is at the edge of what this protocol can see, and this
protocol did not see it. What can be said is what was measured: the fix is worth
7.6% of a short-prompt prefill pass, it removes a 1.4% long-prompt regression,
and B8 at pp32 is not the instrument that shows either.

The ceiling on this line of work is unchanged and the rule does not reach it.
Forcing each shape's layers onto each candidate tiling in turn, the nine shapes'
then-shipped tilings totalled 259.6 ms and the best measured tiling per shape
totalled 201.6, a **1.29×**. (That run's absolute level sits above the 244.4 ms
above because its affine control was read once at the end of each round rather
than interleaved with the candidates, which is a position it pays for; the
comparison quoted is between candidates measured adjacently within one round.)
But 1.29× is an oracle — best-per-shape chosen after the fact — and a dispatch
rule has to generalise from row counts and batch alone. Where a shape would
prefer the narrower block even though the wider one has threadgroups to spare,
the occupancy criterion cannot express it, and 0.83% and 1.21% of measured
per-shape headroom is left deliberately rather than special-cased. The distance
between the 1.29× oracle and the 1.076× rule is the remaining opportunity, and
what would close it is a per-shape table keyed on measurement, not another
threshold. §B9 prices the batch-8 end of exactly that, and audits the sweep this
paragraph rests on.

The throughput table above is the six-round isolated benchmark. A separate run of
the standard paired protocol — 30 trials per arm, both models co-resident in one
process, alternating AB/BA, and the run the memory census above comes from —
reads 1.04×, 0.58×, 0.95×, 0.90×, 0.92× and 0.84× against the table's 0.96, 0.64,
0.94, 0.87, 0.89 and 0.82.

**Five of those six are more favourable to Ternel co-resident than isolated**,
and this run cannot blame contention for it: no timed unit of it exceeded 0.13%
foreign GPU share. What is left is co-residency itself, which is exactly why the
isolated protocol is the one reported — a second 27B model in the process is
cache pressure whether or not it is running, the affine arm is the larger of the
two, and a memory-bound arm has more to lose from it. That predicts the direction
of all three decode rows (0.96 → 1.04, 0.94 → 0.95, 0.89 → 0.92) and of the two
long prefills, and it is the whole reason this run is labelled evidence rather
than result.

The sixth row goes the other way and goes further: **pp32 prefill reads 0.58×
co-resident against 0.64× isolated**, 0.064 apart on intervals that do not
overlap, and the pp32 decode row above it is 0.081 apart in the opposite
direction. Something specific to the shortest prompt is not explained by a
uniform cache-pressure story, and this report does not have the measurement that
would settle it. What both protocols do agree on is the shape: short-prompt
decode sits at parity, the decode deficit widens with context, and short-prompt
prefill is the weak point. They disagree on how weak by six points of ratio, and
the isolated number is the one published.

Output is identical, not merely close. Under greedy decoding both models produced
the same 64 tokens on all three workloads — `greedy_continuations_identical`
true, common prefix 64/64, in the isolated and the paired run alike — which is
what a lossless re-encoding of the same ternary weights should produce, and the
reason the throughput column can be read as a straight cost rather than a trade.
It also holds the split-K reduction to account: summing the K-axis partials with
an ordinary `mx.sum` rather than float atomics is what makes that agreement
reproducible, since atomic addition order varies between runs.

Contention, on one metric throughout — total foreign GPU share, summed across
every other process, not the busiest single one. Across the twelve timed units
behind the numbers above, seven were under 0.2% and the worst was 11.79%. Two
attempts at round 6 were **voided by the run's own 25% bar and re-taken** —
WindowServer at 45.4% and then 30.4% — which is the mechanism doing what it is
there for rather than a caveat: a measurement taken while another process held
half the GPU is discarded, not footnoted. The paired run stayed under 0.13%
throughout.

**Where the remaining decode gap is.** Decoding one token reads every quantised
weight exactly once, so the arms can be compared on the rate at which they get
those bytes off memory — bytes per token from the census above, tokens per second
measured:

| workload | Ternel | official 2-bit | baseline advantage |
|---|---:|---:|---:|
| pp32_tg64 | 160.8 GB/s | 215.2 GB/s | 1.34× |
| pp512_tg64 | 188.5 GB/s | 258.3 GB/s | 1.37× |
| pp2048_tg64 | 178.5 GB/s | 258.7 GB/s | 1.45× |

Decode time per token is these two quantities and nothing else — bytes to read,
divided by the rate they are read at — so the ratio in the table above decomposes
exactly into the byte saving and the bandwidth deficit:

    time ratio  =  7/9  ×  baseline's bandwidth advantage
      pp32       0.778  ×  1.34  =  1.04
      pp512      0.778  ×  1.37  =  1.07
      pp2048     0.778  ×  1.45  =  1.13

This is a decomposition, not a check — bandwidth here is *derived* from the same
tokens per second, so the two sides cannot disagree. Its use is to separate what
is settled from what is not. The left factor is the format's, fixed at 7/9 and
already collected. The right factor is the kernel's, and it is the entire
remaining gap: at pp32 the two very nearly cancel, which is why decode is not
separated from parity there.

The K-axis split is what closed most of it. Before the split this table read
118.4, 126.3 and 121.4 GB/s against a baseline near 210–260, and the summary was
that Ternel had 22.2% less to read and read it at less than half the rate, so the
saving was spent twice over. It now reads at 69–75% of the baseline's rate, and
the saving very nearly covers the difference.

What has not changed is that **neither arm is close to memory-bound.** This
machine streams 546 GB/s; the baseline reaches 39–47% of that and Ternel 29–35%.
The remaining 1.34–1.45× is therefore not a bandwidth wall that the format has to
pay for — it is per-token overhead that a faster kernel can still take. An
earlier draft offered cache residency as the reason the op-level sweep does not
see it: A5 calls one ~20 MB tensor in a tight loop, where the weights have
somewhere to live between calls, rather than streaming 5.9 GB once per token.
That was a guess, and the regime difference has since been isolated to something
else — A5 dispatches its calls independently, which fills a GPU the forward pass
leaves partly idle. §A5 below has it.

---

### B9 — the two clocks, and what a bucket would buy

§B8's 1.29× comes out of a sweep the ladder above was also fitted on, so how
that sweep works, and what happened when its main objection was checked, are
worth setting out.

`ternel_mlx.bench_in_situ` forces one shape's layers onto one candidate tiling
inside the real forward pass and re-reads the pass. It does that three ways, and
what separates them is where the clock stops.

**keep-one** stubs every packed linear, restores one shape, moves it, and
differences against the all-stubbed pass inside the same round. The shape under
test is then nearly the whole pass, which is what makes a per-shape reading
resolvable at all — and is also the objection to it. A starved dispatch in a
stubbed graph has nothing beside it, but `q_proj`/`k_proj`/`v_proj` and
`gate_proj`/`up_proj` are independent branches whose dispatches could fill the
machine for each other in a real pass. That is A5's confound inverted: A5
over-fills the queue and so cannot see starvation, keep-one empties it and may
exaggerate the same effect. `GEMM_MIN_WAVES` and `GEMM_WIDE_BLOCK_WAVES` were
fitted on keep-one, so whether it exaggerates is a question about what ships.

**whole-pass** stubs nothing. Every packed linear is pinned to what the rule
chooses, one shape is moved, and the pass is timed the way mlx-lm runs it:
`generate_step` throws the model's return value away and evaluates only
`[c.state for c in cache]`, so stopping the clock on `mx.eval(out)` would charge
a prefill chunk for a 248320-row matmul it never issues.

Which is what makes the comparison checkable rather than a matter of argument.
**A prefill chunk computes no logits, so retiling `lm_head` must move a
whole-pass reading by nothing.** Every `lm_head` arm is therefore a reading whose
true value is known in advance to be zero, as is the shipped candidate, which is
its own reference. The fourteen such readings at batch 8 have a median absolute
value of 0.46 ms on a 129.0 ms pass and a 95th percentile of 1.23 ms; at batch
16 that percentile is 1.43 ms on 185.4 ms, and at batch 31, 2.89 ms on 335.7 ms.
Those are the floors every figure below has to clear, kept per batch rather than
pooled — the noise is a fraction of the pass it sits on, and a pooled floor would
charge batch 8 for batch-31 noise.

**keep-one does not inflate the occupancy penalty.** Over the 72 arms the two
modes share — `lm_head` excluded, being the null — 60 read a keep-one penalty
above their own batch's floor, and on those whole-pass reads a median **0.91×**
of keep-one's figure, quartiles 0.73 to 1.14. Summed, keep-one charges +1083 ms
against whole-pass's +1050 ms, and 63 of the 72 arms agree on whether there is a
penalty at all. So the branch-overlap objection is real in direction and small in
size: an empty graph does exaggerate, by roughly a tenth, against thresholds
spaced a whole wave apart. §B8's 1.29× and the ladder above stand as measured.

**What a bucket at batch 8 would buy.** Batch 8 is below `GEMM_MIN_BATCH`, so the
rule sends all nine shapes to LUT23 there without consulting the row count.
Moving each shape's layers in turn onto `gemm_8x32` — leaving the other eight on
the shipped choice — prices that decision one shape at a time:

| shape | rows | threadgroups | waves | whole-pass | per dispatch | keep-one |
|---|---:|---:|---:|---:|---:|---:|
| `lm_head` | 248320 | 7760 | 194.0 | −0.20 ms | −0.203 ms | −0.38 ms |
| `mlp.gate_proj`/`up_proj` | 17408 | 544 | 13.6 | **−7.74 ms** | −0.060 ms | −6.81 ms |
| `self_attn.q_proj` | 12288 | 384 | 9.6 | −1.15 ms | −0.072 ms | −0.56 ms |
| `linear_attn.in_proj_qkv` | 10240 | 320 | 8.0 | **−2.45 ms** | −0.051 ms | −1.64 ms |
| `linear_attn.in_proj_z` | 6144 | 192 | 4.8 | −0.13 ms | −0.003 ms | +2.71 ms |
| `linear_attn.out_proj`/`o_proj` | 5120 | 160 | 4.0 | +0.93 ms | +0.015 ms | +1.59 ms |
| `mlp.down_proj` | 5120 | 160 | 4.0 | −0.79 ms | −0.012 ms | +4.02 ms |
| `self_attn.k_proj`/`v_proj` | 1024 | 32 | 0.8 | +0.54 ms | +0.017 ms | +2.08 ms |
| `linear_attn.in_proj_a`/`_b` | 48 | 2 | 0.1 | +7.29 ms | +0.076 ms | +10.18 ms |

The `lm_head` row is the null control, not a result. That it reads −0.20 ms —
0.16% of the pass, well inside the 1.23 ms floor — is what licenses reading the
rest of the column, and it is also why the column is one *named* tiling rather
than each shape's best of five. Best-of-five hands `lm_head` −1.19 ms for free,
so a table built that way would credit every row with about a millisecond of
selection before any real effect started. A bucket is a rule — one tiling above
an occupancy bar — so its cost is that tiling's column.

Only one tiling is in the running at this batch, and not narrowly. Summed over
the eight shapes prefill actually issues, `gemm_8x32` costs −3.50 ms where
`gemm_16x32` costs +135.78, `gemm_16x128` +153.17, `gemm_8x128` +198.53 and
`gemm_8x256` +225.86. Whatever the batch-8 opportunity is, it belongs to the
narrowest 8-deep tiling alone.

Where the bar goes is read off the per-dispatch column, which is the delta
divided by how many of that shape a token issues and so is comparable across
rows. It changes sign between 4.8 waves (`in_proj_z`, −0.003 ms) and 8.0 waves
(`in_proj_qkv`, −0.051 ms): **320 threadgroups, eight passes over the 40 cores.**
Three prefill shapes clear that bar at batch 8 and their arms sum to −11.34 ms.
`q_proj`'s −1.15 ms is inside the 1.23 ms floor and cannot be told from zero, so
the part that resolves is `gate_proj`/`up_proj` and `in_proj_qkv`: **−10.19 ms on
a 129.0 ms pass, −7.9%**. Dropping the bar and switching everything adds the five
shapes below it, which sum to +7.84 ms and take the total to −3.50 ms — so on
these readings the bar is worth more than the switch it gates.

That −7.9% assumes the shapes' savings *add*, which a sweep that moves one shape
at a time cannot say: shapes starved of the same 40 cores may buy less together
than apart, while shapes on one dependency chain may buy the sum. So a third mode
moves every member of the bucket in one pass and each member alone in the same
rounds, the combined arm and its parts read against the same machine state.

**They add.** At batch 8 over fifteen rounds the combined arm reads **−8.29 ms on
a 99.3 ms pass, −8.3%**, against a −7.89 ms sum of parts — a ratio of 1.05, the
bucket buying slightly more together than its members bought apart — and the null
control rides inside the combined arm at +0.16 ms, 0.16% of the pass. (This run's
reference pass is 99.3 ms where the whole-pass run's was 129.0 ms — two machine
states, which is why each run is differenced only against itself — and the
fractions agree, −8.3% against −7.9%.) The batch-8 bucket is a measurement now,
not an upper bound.

**The same run kills the range the bar implied.** 320 threadgroups was read at
batch 8, and a bucket is a claim about a batch range, so the rule was also fired
at batch 12 — where `ceil(12/8)` is two batch tiles, every dispatch supplies
twice the threadgroups, and six prefill shapes clear the bar instead of three.
All six lose. The combined arm reads **+61.87 ms on a 147.1 ms pass, +42%**, the
sum of parts +64.93 ms — the parts add here too, which is no comfort — and the
batch-8 winners are underwater individually: `gate_proj`/`up_proj` +14.81 ms,
`in_proj_qkv` +2.45 ms, `q_proj` +1.85 ms, `lm_head` exactly 0.00. The bar counts
threadgroups and not what fills them: at batch 12 the second batch tile of every
dispatch carries four live lanes out of eight, and a kernel that pays for eight
to fill four loses to LUT23 on every shape it touches. What survives is not
"batch in `[8, 16)` above 320 threadgroups" but **batch 8 alone** — the one point
in the range where the batch tile is full — with 9 through 15 a measured loss at
12 and an untested one elsewhere.

What the two batches agree on is additivity itself — 1.05 and 0.95, within 5% in
both directions — which is what licenses reading whole-pass's one-shape-at-a-time
tables as bucket prices in the first place.

The batch-8 point is recorded as measured headroom rather than shipped, for the
same reasons as before, now with the additivity caveat discharged: capturing it
would change which kernel computes most of a batch-8 prefill — so the A4 kernel
gates, the B7 graph gate and the B8 benchmarks would all need re-taking — and no
workload this report publishes prefills at batch 8. What the next kernel
generation starts from is a confirmed, additive **−8.3%** at one batch, and a
demonstration that the boundary it must not cross sits at the batch tile's own
depth.

---

## What the kernels are, and how one is chosen

Stage A has the full derivation and the 99-point sweep behind it; this is the
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

The dispatch rule. The batch decides which ladder is walked and the *row count*
decides how far down it, because the thing that separates these tilings in a
real forward pass is whether the dispatch fills the machine:

```
batch < 16   → LUT23 bt1 (batch 1) / bt2 (2–3) / bt4 (≥ 4)
batch ≥ 32   → walk the large-batch ladder:  32×64, k32, s1×2, frag+epi+tgt
                                             32×32, k32, s1×2, frag+epi+tgt
otherwise    → walk the small-batch ladder:  16×128, k64, s2×4, staged
                                             16×32,  k32, s2×2, frag+epi+tgt

walking a ladder, widest row block first:
  drop the rungs whose row block does not divide the row tile — a block
    straddling two tiles would read the second tile's codes through the first
    tile's base pointer, so the pairing is refused rather than approximated
  if the narrowest survivor still offers fewer than 2×40 threadgroups, no
    tiling can fill the GPU along this tensor's rows and LUT23's K-axis split
    takes it
  otherwise take the widest survivor offering at least 3×40, else the narrowest

a dispatch offers ceil(rows/row_block) × ceil(batch/batch_block) threadgroups
```

The two ladders do not top out at the same width, and that is measured rather
than tidy. Filling the machine is a *floor* on the row block, not an argument
for width: past some width a wider block is simply worse, and at the batches
this model actually prefills at, 128 rows is over the top. So the large-batch
ladder stops at 64 where the small-batch one goes to 128. Fitted in situ at
batches 511 and 2047 over the eight shapes a prefill dispatches, a 128-row wide
rung costs 1.49% and 0.18% of the summed linear time against this one, and of
sixteen 64-against-128 readings the only two whose ranges separate both say 64.

Against a per-case oracle over all 30 configurations built, the rule gives up a
mean of 2.22% and a worst case of 11.92%, and is exactly optimal in 27 of the 78
points this sweep can still score. `tests/test_mlx_modules.py` replays all 99
through the dispatch functions and asserts the partition, so the thresholds are
checked against the measurement rather than described by it. Ten of the 21
unscored dispatch a row block A5 never swept. The other eleven are not a neutral
exclusion and are worth stating plainly: there the rule refuses on occupancy and
takes LUT23, while A5 ranks a prefill tiling first — by that file's estimator
alone, 0.2% to 178% given up. The in-situ sweep settles it the other way, and
not narrowly: the tilings A5 prefers on those two shapes cost 12.28× and 20.73×
the affine baseline inside a real forward pass, against LUT23's 1.89× and 3.91×.
Both files are right about what they measured; only one measured the regime the
model runs in.

**Those scores are now the floor of the three bfloat16 sweeps, and rescoring on
that floor found a real defect rather than confirming the old one.** The
reasoning that the dtype correction could not reach this rule — it ranks packed
configurations against each other, so both sides of every comparison move
together — holds for eight of the nine shapes and fails for the ninth. The packed
kernels take their dtype as a *template* parameter, so two dtypes are two sets of
compiled kernels, and on the 48-row narrow tile the ranking between two of them
inverts. The rule had dispatched a 128×32 tiling from batch 64 up, chosen because
it was fastest there on float32. Its batch block is 128 deep, so at batch 64 half
that block is empty, and in bfloat16 it loses to the 16×128 staged tiling by
25–30% in each of the three sweeps independently. That single point cost 29.86% —
above the 20% bound the test enforces — and the fix is the second bucket now in
the rule above, not a wider bound. It takes that point to zero, the worst case
over all 99 from 29.86% to 11.92%, and the mean from 2.43% to 2.13%.

For scale: these 96 tensors are 0.23%–0.93% of the model's matmul time at those
batches, so the correction is worth well under a tenth of a point of forward
pass. It is reported because a threshold that has drifted from its measurement is
worth fixing whether or not it is worth much, and because it is the one place
where the dtype defect reached the dispatch rule and not just the baseline.

The floor implementation was checked before anything was published from it, by
recomputing the float32 floor the same way and confirming it reproduces the
figures this section previously carried: 43/99 exact, mean 2.04%, worst 14.52% at
`linear_attn.out_proj` batch 512, for the rule as it stood before the narrow-tile
correction.

The sweep covers 99 points and not the 110 an earlier draft reported, because
the shape census it was weighted by carried a 5120×12288 `self_attn.o_proj` the
model has no instance of: `o_proj` is 5120×6144, the same shape as
`linear_attn.out_proj`. The eleven phantom points are dropped and the real
`out_proj` reweighted; `ternel_mlx.layout.MODEL_LINEAR_SHAPES` is now the single
census, and it fails at import unless its nine shapes account for exactly the
497 quantised matmuls a decoded token issues.

At op level, weighting every measured shape by how many tensors of that shape one
forward pass contains, the dispatched packed path is **at rough parity with
`mx.quantized_matmul`, not ahead of it**. Across three bfloat16 sweeps:
0.86–0.89x at batch 1, 0.80–0.82x at 2, 0.81–0.84x at 4, 0.86–0.87x at 8,
1.16–1.17x at 16, and 0.88x–1.03x from 32 through 512.

**That headline does not reproduce in the model.** The same nine shapes, the same
kernels, the same weighting, measured inside the real forward pass instead of in
a loop of independent calls, come to 1.91x at batch 32 — where this sweep reports
0.96x–0.97x. Every row of this table remains a correct measurement of the thing
it measures; the disagreement is a property of A5's regime, set out in the note
below, and it is not small. Where the two differ, the in-situ figure is the one
that describes the model, and §B8 above is where it is derived.

An earlier draft of this section reported that same weighting as *at or above
`mx.quantized_matmul` at every batch size measured* — 1.09x at batch 1, 1.64x at
2, 1.77x at 4, 2.11x at 8. That was a dtype artifact, and it is the most
consequential error this report has had to correct. The sweep ran float32
activations against an affine baseline whose scales were float16, a combination
MLX promotes to float32; the baseline was therefore executing a float32 kernel,
while the packed arm — whose dtype is a kernel template parameter it takes
directly — was executing the one it was asked for. Correcting it moves the two
arms by very different amounts, which is the signature of the defect: on the
dispatched configuration of each of the nine shapes, the packed arm's time
changes by a median 0.93× while the affine arm's drops to a median 0.76×, and at
batches 4 and 8 — where the old ratio peaked — the affine kernel runs **2× to 3×
faster** (0.35× to 0.51× of its float32 time) while the packed one stays within
about 10% of its own. The three float32 sweeps below are kept because they are
what the shipped dispatch rule was chosen against, and because
packed-against-packed comparisons within one sweep are unaffected — only the
ratio against the affine baseline moves.

Read either as a *throughput* ratio, not a latency one. A5 enqueues each
configuration's calls back to back with no dependency between them, so MLX
overlaps them and the GPU is never idle between dispatches — the wrong regime for
decode, where 497 matmuls form a single read-after-write chain and each one waits
alone. Forcing that chain is what B8 was measuring and what the next section
takes apart.

An earlier draft called that same independence *the right regime for prefill*.
It is not, and the error has the same shape as the dtype one: independence does
not merely overlap the calls, it **manufactures parallelism the forward pass does
not have.** `_synchronised` evaluates `inner` independent copies of one matmul in
a single `mx.eval`, so a shape that dispatches sixteen threadgroups in the model
is submitted here as 128, and the GPU fills. A tiling whose only defect is that
it cannot fill the GPU is therefore precisely the tiling this sweep cannot
penalise — which is how every number above reads parity while the same kernels,
on the same shapes, run up to 9.29× in the graph. A5 measures these kernels'
arithmetic, and measures it correctly. It does not measure their occupancy, and
at prefill batches occupancy is what the dispatch rule gets wrong.

**The float32 sweep was run three times, on three days.** The repeats exist
because these ratios were measured on a machine in use while the full-model
numbers above were not, and one sweep on a busy desktop is an assertion rather
than a measurement:

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 40 | 64 | 128 | 200 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lowest of three, float32 | 1.08x | 1.63x | 1.73x | 2.06x | 1.09x | 1.03x | 1.02x | 1.02x | 1.04x | 1.03x | 1.04x |
| highest of three, float32 | 1.11x | 1.75x | 1.85x | 2.17x | 1.27x | 1.31x | 1.03x | 1.09x | 1.05x | 1.08x | 1.10x |
| **lowest of three, bfloat16** | **0.86x** | **0.80x** | **0.81x** | **0.86x** | **1.16x** | **0.96x** | **0.88x** | **0.98x** | **1.01x** | **0.90x** | **1.03x** |
| **highest of three, bfloat16** | **0.89x** | **0.82x** | **0.84x** | **0.87x** | **1.17x** | **0.97x** | **0.89x** | **0.99x** | **1.02x** | **0.91x** | **1.03x** |

What the repeats establish is that the reversal is not sweep-to-sweep noise. The
float32 ratio never drops below 1.019x at any batch in any of the three, and the
batch-8 peak reproduces within 6% (2.06x, 2.11x, 2.17x) — so 2.11x falling to
0.86x is roughly twenty times the scatter the instrument shows. The two widest
disagreements among the float32 runs — batch 16 and batch 32 — both come from the
single dirtiest sweep, which ran at a median 31.1% foreign GPU share against
19.0% for the primary.

The bfloat16 rows are three sweeps run on three days, and they are far tighter
than the float32 ones: the two rows never differ by more than 0.03x at any batch,
against spans of up to 0.28x among the float32 three. The bfloat16 sweeps also
ran on a much quieter machine — median foreign GPU share 0.14%, 0.23% and 0.44%,
against the 19.0% and 31.1% quoted just above — which has to be said, because a
cleaner run makes *both* arms faster and could be mistaken for this effect, and
because it is the likeliest reason the bfloat16 rows agree so much more closely
with each other. It is not what produced the reversal. Contention lands on both
arms of a paired, alternated ratio and largely cancels, which this report
measures directly two paragraphs down: between 5–7% and 69–85% foreign share the
same shape's ratio moved 3.6–8.6%. Nothing about a quiet machine makes the affine
kernel 2–3× faster while leaving the packed one where it was.

The cleanest evidence is a different pair, where contention is not a variable at
all: A6 ran its sweep twice under the same GPU-lock discipline, changing only the
affine control's dtype, and the decode ledger moved from 0.90×/0.55×/0.47×/0.43×
of `mx.quantized_matmul` to 0.99×/1.07×/1.08×/1.09× at batches 1, 2, 4 and 8.
Read in the overlapped regime A5 uses, that same dtype-matched A6 run puts the
unsplit packed path at 1.05×, 0.95×, 0.94× and 0.97× of affine. The two
instruments disagree by 10–20% on magnitude and agree on the thing that matters:
nothing in this regime is above 1.1×.

Why a paired ratio survives this at all is visible directly: the same 17408x5120
shape was measured once at 5–7% foreign share and once at 69–85%, and the ratio
moved by +3.6% at batch 1, +8.6% at batch 8, and −5.2% at batch 64 — against the
7x that contention did to absolute throughput elsewhere in this report. The
alternation cancels a slowdown both arms take together. So the shape of the
op-level result is trustworthy and the third decimal is not.

### The K-axis split — where the decode deficit actually came from

A packed path that sits within about 15% of `mx.quantized_matmul` on every op
could not explain a model that then decoded at 0.61–0.72× of it, and A6 resolves
the difference — the split this section derives is what moved decode to the
0.89–0.96× §B8 reports. (When A6 was run the op-level gap looked wider still, because the
affine baseline was then mismeasured in the direction described above; the
contradiction shrank when that was corrected but did not close, and none of what
follows depends on its size.) The A5 sweep and the model disagree because they
are in different regimes, so A6 measured both: every legal split of every shape
in the model, timed twice
— once with calls enqueued independently, as A5 does, and once with each call's
output fed into the next so nothing can overlap, as decode does.

The unsplit kernel pays between **1.01× and 8.32× its own overlapped time** once
the chain is forced. `mx.quantized_matmul` under the identical harness sits at a
median 1.02× across the same 36 points, 31 of them under 1.10×, which is what
makes it the control: whatever part of the penalty belongs to MLX's encoder
rather than to this kernel would show up there too, and almost none does. Its
three exceptions are the diagnosis rather than a hole in it — the only points
above 1.5× (1.52×, 2.25×, 2.34×) are all `linear_attn.in_proj_a`, the 48-row
tensor that starves any kernel of threadgroups. Nor is it cache residency — the
chained arm calls the same ~20 MB tensor in the same tight loop as the
overlapped arm; only the dependency differs.

The cause is that LUT23 walks a tensor's 40–136 groups strictly in series inside
each threadgroup, so a tensor whose rows yield only a handful of row tiles runs
one short chain on a 40-core machine that is otherwise idle. Overlapping hides
it; decode cannot. The fix is to split the K axis: each threadgroup takes one
contiguous run of groups instead of all of them, multiplying the threadgroup
count by the split, and the fp32 partials are summed by `mx.sum` — not by float
atomics, whose accumulation order varies run to run and would cost the
bit-identical greedy continuations B7 and B8 both check.

Splits are restricted to **divisors** of the group count. A ragged split leaves
one threadgroup holding a full chunk while another holds none, and a dispatch
ends when its slowest threadgroup does, so the extra threadgroups would buy
nothing.

At batch 1, per shape — chained per-call time, the regime decode is in:

| shape | ×/token | tiles | groups | unsplit | split | k | `mx.quantized_matmul` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `linear_attn.in_proj_a`/`_b` | 96 | 1 | 40 | 47.98 µs | **7.01 µs** | 40 | 7.11 µs |
| `self_attn.k_proj`/`v_proj` | 32 | 4 | 40 | 56.21 µs | **10.31 µs** | 40 | 8.88 µs |
| `linear_attn.out_proj`/`o_proj` | 64 | 20 | 48 | 64.55 µs | **26.88 µs** | 24 | 26.64 µs |
| `mlp.down_proj` | 64 | 20 | 136 | 187.38 µs | **61.30 µs** | 34 | 68.43 µs |
| `linear_attn.in_proj_z` | 48 | 24 | 40 | 55.10 µs | **27.45 µs** | 20 | 27.15 µs |
| `linear_attn.in_proj_qkv` | 48 | 40 | 40 | 62.10 µs | **43.32 µs** | 20 | 41.58 µs |
| `self_attn.q_proj` | 16 | 48 | 40 | 76.96 µs | **49.11 µs** | 10 | 48.39 µs |
| `mlp.gate_proj`/`up_proj` | 128 | 68 | 40 | 78.01 µs | **63.56 µs** | 10 | 64.01 µs |
| `lm_head` | 1 | 970 | 40 | 833.96 µs | 833.96 µs | 1 | 773.75 µs |

The gain tracks how starved a shape was, exactly as the diagnosis predicts: the
48-row tensors that filled a single threadgroup gain 6.8×, and `lm_head`, whose
970 row tiles already saturate the machine, is left unsplit and gains nothing.
What the split does *not* do is open a lead. Where the unsplit kernel was 2–7×
off `mx.quantized_matmul`, the split one lands beside it — ahead on three of the
nine shapes and behind on six, by 10% at best (`mlp.down_proj`, the only 136-group
tensor) and 16% at worst (`self_attn.k_proj`).

Summed over the 497 quantised matmuls one decoded token issues:

| | chained cost of one token's matmuls |
|---|---:|
| unsplit | 40.20 ms |
| **the shipped rule** | **19.80 ms** |
| an oracle picking each shape's best split | 19.28 ms |
| `mx.quantized_matmul` | 20.09 ms |

So the packed path goes from **2.00× `mx.quantized_matmul` to 0.99×** on the
serial chain decode actually runs. That is parity, not a lead: the split recovers
the serialisation penalty and stops there. The batched regimes are recovered
about as far and land just short — 1.05× at batches 2 and 4 and 1.09× at 8,
where unsplit they were 2.15×, 2.16× and 1.52×.

An earlier draft of this section reported 0.55× at batch 2 and 0.43× at 8. Those
came from an affine control built with float16 scales against bfloat16
activations — a combination MLX promotes to float32 — so the baseline was running
a float32 kernel and giving up roughly half its speed to it. The control was
rebuilt in the dtype the model declares and every number in this section re-read
from that run. The correction turns a claimed 2× batched win into a 5–9% loss,
which is the largest single correction in this report.

The rule is one number, `SPLIT_K_THREADGROUPS = 800` — twenty threadgroups per
core on this 40-core part — and each tensor takes the largest divisor of its
group count that keeps the grid under it. How that number was picked matters more
than its value. Past the point where the serialisation factor reaches 1.0 the
curve is flat, flat enough that whichever ceiling wins on one sweep generally
loses on the next: scored against the run that did not choose it, **every ceiling
from 320 to 2560 gives up between 2.6% and 3.6%** of a decoded token against a
per-shape oracle. An earlier version of this rule shipped 320 because 320 was the
argmin of the sweep it was scored on, and the test that gated it was set just
above the 1.84% it read there — which passed, and then failed at 3.32% when the
sweep was repeated with nothing about the rule changed. That was a fit to one
run's noise, not a measurement.

800 is taken because it is the only value that beats 320 on both runs and on both
measures: 1.58% and 2.71% of the token against 1.84% and 3.32%, and 9.1% and 9.5%
on the worst single shape against 16.5% and 15.7%. It also needs 23 kernel
instantiations over the swept points where 320 needs 27, since a wider budget
lands more shapes on the same divisors. 1360 edges it on the second run's ledger
(2.19% against 2.71%) but is worse on the first (2.60% against 1.58%), worse on
the worst single shape in both, and larger. Ceilings well below remain plainly
wrong on both runs — 160 gives up 10.6% and 11.0%, and 80 gives up 22.0% and
22.9%. What keeps this capped at all rather than uncapped is the split's costs —
an extra reduction dispatch, an fp32 partial buffer of `k · batch · rows`, and
threadgroups competing with whatever else is in flight — all of which grow with
the ceiling while the gain does not.

Accuracy does not pay for it, and the argument does not rest on a tolerance.
Splitting the K axis only reassociates the sum over a row's groups, so on data
where that sum is exact the split must return *the same bits*: with every scale
1.0 and every activation ±1, each group sum is an integer in [-128, 128] and each
row total an integer far inside fp32's exact range.
`test_splitting_the_k_axis_does_not_change_exact_arithmetic` asserts bitwise
equality there for *every legal split* at all three group counts the model has —
all 23 of them, not a sample, so which splits the ceiling happens to select
cannot move what is covered. On ordinary data the arms do round differently, and
`test_a_split_stays_inside_the_fp32_summation_bound` holds each of them to the
textbook sequential-summation bound `n · eps · Σ|terms|`, with `Σ|terms|`
computed exactly as `|W| @ |x|` in float64 — a bound that belongs to the
arithmetic rather than to any arm, so it cannot be tuned to the one under test.
Every arm uses under a tenth of a percent of it; a chunk boundary that dropped or
repeated one group would land at least 154× outside it. `mx.sum` rather than a
float atomic does the reduction, so a split is also bit-identical to itself
between runs, which is what the artifact gate's identical greedy continuations
require. `tests/test_mlx_modules.py` replays `results/mlx/a6_split_k.json`
through `select_lut23`, so the ceiling is checked against the measurement rather
than described by it. Its two bounds are set from the *worse* of the two sweeps
rather than the one that chose the ceiling — that being exactly the mistake the
previous pair of bounds made.

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
amount of averaging removes a bias. On the same 99 points the three float32
sweeps report mean regret of 2.46%, 5.45% and 6.07%, with worst cases of 14.5%,
72.0% and 119.9%; the three bfloat16 sweeps, run on a much quieter machine,
report 2.32%, 4.01% and 2.32%, with worst cases of 17.2%, 80.0% and 19.2%. The
worst cases stay wild in both, which is the point: a single sweep's *worst* case
is mostly a record of what its noisiest moment was.

What corrupts the ranking turns out to be how much the foreign load *varies*,
not how large it is. The sweep captured 2026-08-10, inside the window when an
unrelated MLX process held 85% of the device, is the *cleanest* of the three: a
saturating competitor scales every configuration alike and cancels in a ratio.
The two captured against a desktop compositor bursting between 0.4% and 95% are
the unusable ones.

So the rule is judged on the floor of the three bfloat16 sweeps — the fastest
time each configuration was ever seen to achieve at each point, in the dtype the
model actually runs. The estimator is a minimum because the noise is one-sided: a
competitor can take GPU time from a kernel and never give it any, so the smallest
of several timings is the one nearest the uncontended speed. On that floor the
shipped rule gives up **2.13% on average and 11.92% at worst**, and no point
exceeds 20%. All six sweeps are published alongside this report so both floors
can be recomputed rather than taken on trust.

The worst case is no longer a twin ambiguity, and describing it as one would be
wrong. It is `linear_attn.out_proj` at batch 8, where the rule dispatches
`lut23_bt4_packed` at 142.88 µs and a prefill tiling the rule does not build,
`gemm_8x256`, runs it in 127.66 µs. The rule's family threshold sits at batch 16
because on float32 batch 8 was a wash — 0.956x on the geometric mean, six of nine
shapes with overlapping bands. In bfloat16 it is not a wash for this one tiling:
the four worst points on the whole floor are all batch 8, and all four lose to
it. Taking it per shape where it wins would be worth 1.058x on the weighted
batch-8 forward pass, reproducing at 1.049x–1.065x on the three sweeps
separately.

That is left unexploited, and deliberately. The win is not uniform — the same
tiling loses on `mlp.down_proj`, `linear_attn.in_proj_a`, `in_proj_z` and
`k_proj` — so capturing it needs a rule keyed on something beyond the tile and
the batch, plus a kernel instantiation and gate coverage that do not exist yet.
No benchmark in this report runs at batch 8: decode is batch 1 and prefill is 32
and 512, so nothing published here would move. It is recorded as measured
headroom rather than claimed.

*A rule keyed on something beyond the tile and the batch* turns out to be the
larger finding, and not at batch 8. The same blindness costs 1.29× on the linear
block at batch 32 — a batch this report does benchmark — for the reason §B8 sets
out: the missing key is the row count, and it decides how much of the GPU a
dispatch can occupy. This paragraph was written before that was known, and is
kept because the batch-8 case is still a separate, still-unexploited point.

**§B9 has since reversed which tiling and which shapes.** Measured in the forward
pass rather than in A5's loop of independent calls, `gemm_8x256` is the worst arm
at batch 8 and not the best: on `linear_attn.out_proj` — the shape A5 picks it
for — it costs +32.16 ms on a 129.0 ms pass, and over the eight shapes prefill
issues it costs +225.86 ms. The batch-8 headroom is real and it is elsewhere: on
`gemm_8x32`, on the shapes whose dispatches clear eight waves of the 40 cores,
and at batch 8 alone — measured additively at −8.3% of the pass there and a 42%
loss at batch 12. Both files are right about what they measured; the
disagreement is A5's regime again, and it is the same one §A5 records at every
other batch.

**The dispatch rule is still the one measured above.** Nothing in this report is
run with the row-keyed rule; the 1.29× is a measurement of what the tilings cost
in situ, not a result of shipping a fix. Its sweep also covers a single prefill
width so far, and a rule fitted to nine shapes at one batch is the same overfit
that put the op-level headline in this report to begin with.

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
| Two matmul kernels (gemv, gemm) | three: one `tq1_matmul` source templated at `BT=1` and `BT>1`, plus a genuinely different `tq1_gemm` | one thread per output row cannot feed Apple's matrix units at all; the register-fragment rewrite took prefill from 0.44–0.96x to 1.02–1.27x of affine — a float32 yardstick, and 0.88–1.17x once the baseline is measured in the model's own dtype, though the rewrite's gain is packed-against-packed and survives either way |
| Non-quantised tensors taken from the official repo after a norm cross-check | same, **plus a value-head permutation the plan did not anticipate** | the bit-exact comparison failed on eight tensor kinds until the repeat-major/key-head-major difference was modelled |
| `monitor_process` from `tools/benchmark_llama_distribution.py` for peak memory | a `ps -o rss=` sampler | the existing one shells out to `nvidia-smi` |
| Stage A gate on synthetic **plus real** tensors | synthetic only in A4; real weights first exercised in B4/B6/B7 | the real tensors need the 7 GB download the agreed staging puts after that gate |
| llama.cpp Metal as an optional cross-runtime check in B7 | not run | the official MLX checkpoint is a stronger reference — bit-identical weights make every difference attributable to kernels alone, which a second runtime would confound |

---

## Reproduction

```bash
uv sync --extra convert

# The whole suite. Most of it replays the JSON below rather than describing it,
# so it has to run after the stages that write those files, not before.
uv run pytest tests/

# Stage A
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
              --max-foreign-gpu-share 0.25 --activation-dtype bfloat16
uv run python -m ternel_mlx.bench_split_k --output results/mlx/a6_split_k.json \
              --batches 1 2 4 8 --enqueued 32 --trials 12 --warmup 3 \
              --max-foreign-gpu-share 0.2
uv run python -m ternel_mlx.bench_split_k --output results/mlx/a6_split_k_prefill.json \
              --batches 16 32 64 --enqueued 32 --trials 12 --warmup 3 \
              --max-foreign-gpu-share 0.2

# Stage B — GGUF is artifacts/models/Ternary-Bonsai-27B-Q2_0.gguf,
#           sidecar is artifacts/packed/Ternary-Bonsai-27B-Q2_0.gguf.tq1g128
uv run python -m ternel_mlx.convert --gguf <gguf> --sidecar <sidecar> \
              --baseline artifacts/baseline-mlx-2bit --out artifacts/ternel-mlx \
              --max-shard-bytes 4294967296 --activation-dtype bfloat16
# After any later edit to layout/kernels/modules/packed_model, re-emit the
# artifact's self-contained model_file; the drift test fails until this runs.
uv run python -m ternel_mlx.model_file --artifact artifacts/ternel-mlx
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
              --rounds 6 --trials-per-round 5 --round-attempts 8 \
              --round-retry-wait-seconds 1800 --max-foreign-gpu-share 0.25
# The three in-situ modes. keep-one takes a --baseline because it divides by the
# affine arm; the other two have no affine arm and refuse the flag.
uv run python -m ternel_mlx.bench_in_situ --mode keep-one \
              --artifact artifacts/ternel-mlx --baseline artifacts/baseline-mlx-2bit \
              --output results/mlx/b9_in_situ_keep_one.json \
              --batches 8 16 31 32 64 128 --rounds 5 --inner 3 --warmup 2 \
              --max-foreign-gpu-share 0.25
uv run python -m ternel_mlx.bench_in_situ --mode whole-pass \
              --artifact artifacts/ternel-mlx \
              --output results/mlx/b9_in_situ_whole_pass.json \
              --batches 8 16 31 --rounds 10 --inner 3 --warmup 2 \
              --max-foreign-gpu-share 0.25
uv run python -m ternel_mlx.bench_in_situ --mode bucket \
              --artifact artifacts/ternel-mlx \
              --output results/mlx/b9_in_situ_bucket.json \
              --bucket-arm gemm_8x32 --bucket-min-threadgroups 320 \
              --batches 8 12 --rounds 15 --inner 3 --warmup 2 \
              --max-foreign-gpu-share 0.25

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
