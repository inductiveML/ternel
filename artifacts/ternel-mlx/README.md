---
license: apache-2.0
pipeline_tag: text-generation
library_name: mlx
base_model: prism-ml/Ternary-Bonsai-27B-gguf
tags:
  - mlx
  - apple-silicon
  - ternary
  - lossless
  - qwen3_5
---

# Ternary Bonsai 27B — MLX, lossless, 1.75 bits/weight

The smallest exact representation of [Ternary Bonsai 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
on Apple Silicon: the same ternary weights as the official
[MLX 2-bit distribution](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-mlx-2bit),
stored at **1.75 bits/weight** instead of 2.25 — exactly 7/9 the size — and executed
directly from packed storage by custom Metal kernels. No weight is ever dequantised
into a model-sized buffer. Under greedy decoding it emits **token-for-token identical
output** to the official checkpoint while holding **22.19% less live memory**
(5.89 GB against 7.57 GB during generation).

This is a memory-density result, not an accuracy or speed win. The official 2-bit
checkpoint already stores exact ternary weights (its affine parameters satisfy
`bias == -scale` with codes in `{0, 1, 2}`), so there was no accuracy gap to close —
both checkpoints hold the same weights, verified elementwise. Token generation runs at
0.89–0.96× the official checkpoint's rate and prompt processing at 0.64–0.87×,
measured on an M4 Max; if you want maximum speed and have the memory, use the
official 2-bit repo. If you want the smallest exact Bonsai, this is it.

## Run it

Requires an Apple Silicon Mac with **16 GB+ unified memory** (the weights are 5.5 GB
on disk and generation holds ~5.9 GB live) and Python with
[mlx-lm](https://github.com/ml-explore/mlx-lm) `>= 0.31.3`:

```bash
pip install mlx-lm
mlx_lm.generate --model inductiveML/Ternary-Bonsai-27B-mlx-lossless-1.75bpw \
  --prompt "Explain what a ternary weight is." --max-tokens 256
```

The first run downloads the checkpoint (~5.5 GB) into the Hugging Face cache;
later runs load from cache. From Python:

```python
from mlx_lm import load, generate

model, tokenizer = load("inductiveML/Ternary-Bonsai-27B-mlx-lossless-1.75bpw")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain what a ternary weight is."}],
    add_generation_prompt=True,
    tokenize=False,
)
text = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=True)
```

**mlx-lm newer than 0.31.3:** upstream has merged a `trust_remote_code` gate around
checkpoints that carry their own architecture code, as this one does. On releases
that include it, add `--trust-remote-code` to the CLI (or `trust_remote_code=True`
to `load(...)`). On 0.31.3, the current release, no flag is needed.

## The numbers

| | official mlx-2bit | this repo |
|---|---|---|
| Bits per weight (quantised tensors) | 2.25 | **1.75** |
| Weight storage | 8,490,785,104 B | **5,888,388,744 B** |
| Live memory during generation | 7.57 GB | **5.89 GB** (−22.19%) |
| Greedy output | — | **identical, 576/576 tokens** |
| Token generation rate | 1× | 0.89–0.96× |
| Prompt processing rate | 1× | 0.64–0.87× |

Measured on a MacBook Pro M4 Max (40 GPU cores, 48 GB), macOS 26.5.2, MLX 0.32.0,
mlx-lm 0.31.3, against `prism-ml/Ternary-Bonsai-27B-mlx-2bit` running through stock
mlx-lm. Full methodology, paired bootstrap confidence intervals, and every gate's
machine-readable result are in the
[Ternel repository](https://github.com/inductiveML/ternel).

## Intelligence density

The [Bonsai model card](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) measures
capability per deployed byte as intelligence density — `D = -log2(1 - score/100) / size_GB`
— and quotes **0.400 /GB** for Ternary Bonsai 27B from a **5.9 GB** size. That 5.9 GB is
the format's *ideal* size, which the same card notes no shipped build reaches — "the
deployed footprint sits above the representation's information-theoretic minimum until
native ternary kernels close the gap": 7.17 GB deployed in llama.cpp (2.125 bits/weight
slots), 8.49 GB in the official MLX distribution (2.25 bits/weight).

This repo is that gap closed, on Apple Silicon. The packed artifact deploys at 5.89 GB —
the ideal size, as a runnable checkpoint rather than a limit:

| Deployed build | Weights on disk | Benchmark avg* | Density (1/GB) |
|---|---|---|---|
| Qwen3.6-27B FP16 | 54 GB | 85.07 | 0.051 |
| Qwen3.6-27B IQ2_XXS ("2-bit") | 9.4 GB | 72.73 | 0.199 |
| Bonsai, official mlx-2bit | 8.49 GB | 80.49 | 0.278 |
| Bonsai, official llama.cpp Q2_0 | 7.17 GB | 80.49 | 0.329 |
| **Bonsai, this repo** | **5.89 GB** | **80.49** | **0.400** |

\* Scores are the official Bonsai card's 15-benchmark thinking-mode averages, not re-run
here. For this repo that inheritance is exact rather than approximate: the language-model
weights are bit-identical to the official checkpoint (verified over all 26,893,352,960 —
see Fidelity) and greedy output is token-identical, so the official score *is* this
artifact's score; the density gain is entirely the denominator. Two of the 15 benchmarks
exercise the vision tower, which no Bonsai language-model artifact ships (theirs load a
separate mmproj pack; this repo is text-only).

## Fidelity

`TQ1_G128` stores each group of 128 ternary weights in 28 bytes: two raw FP16 scale
bytes, 25 base-3 bytes of five trits each, and a tail byte of three. The conversion
is a re-encoding, not a re-quantisation:

- All **26,893,352,960** ternary weights across 498 quantised tensors verify
  **bit-exactly** against the pinned source GGUF — zero mismatches, and zero raw
  scale-bit mismatches across all 210,104,320 groups.
- Teacher-forced logits agree with the official checkpoint at every one of 343
  positions tested (argmax-identical; max KL 4.8×10⁻⁵ nats, residual differences
  are kernel accumulation order, not representation).
- Greedy generation is token-identical across every test prompt, 576/576 tokens.

## This checkpoint contains code

`config.json` names a `model_file`, so mlx-lm builds the model from
[`ternel_packed_model.py`](./ternel_packed_model.py) shipped in this repo — that is
how the packed weights run without installing anything beyond mlx-lm. You are
trusting that file, so it is built to be audited:

- It is a single self-contained Python file that imports only `math`, `mlx`, and
  `mlx_lm`, generated mechanically from the sources at
  [inductiveML/ternel](https://github.com/inductiveML/ternel) — never hand-edited,
  and a test in that repo fails if the shipped file drifts from what the sources emit.
- [`ternel_manifest.json`](./ternel_manifest.json) records its SHA-256 and the
  SHA-256 of each source module it was generated from, alongside per-tensor content
  hashes for every weight in the checkpoint.
- The model class subclasses `mlx_lm.models.qwen3_5.Model`; the file adds the packed
  storage layout, three Metal kernels (single-vector, batched, and embedding
  lookup), and the module replacement — nothing else.

## Compatibility

- **Supported:** mlx-lm ≥ 0.31.3 on Apple Silicon (CLI, Python API, and anything
  that calls `mlx_lm.load` with checkpoint code enabled). Verified against mlx
  0.32.0 / mlx-lm 0.31.3.
- **Not supported: LM Studio.** Its MLX engine deliberately does not execute code
  shipped in checkpoints, so it refuses any repo that declares a `model_file`.
  Native support would require the packed format landing in mlx-lm itself.
- **Text-only.** The source GGUF carries no vision tower, and `config.json` sets
  `language_model_only: true`. Bonsai's vision path is not in this checkpoint.

## Sources, pinned

Every weight traces to a pinned upstream artifact; both hashes were verified after
download, not trusted:

| What | Identity |
|---|---|
| Quantised weights | `prism-ml/Ternary-Bonsai-27B-gguf` @ `abbae723028d71be674e71e1a71201a6f43fab22`, `Ternary-Bonsai-27B-Q2_0.gguf` (SHA-256 `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`) |
| Non-quantised tensors, tokenizer, chat template | `prism-ml/Ternary-Bonsai-27B-mlx-2bit`, cross-checked numerically against their GGUF F32 counterparts |

## License and attribution

Apache 2.0, inherited from the sources; `LICENSE.txt` and `NOTICE.txt` ship in this
repo. Created using **Bonsai by Prism ML**. Bonsai is built from **Qwen3.6-27B**,
Copyright 2026 Alibaba Cloud, also Apache 2.0. This repository is an independent
re-packing and is not affiliated with Prism ML or Alibaba Cloud.
