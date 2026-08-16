# Ternel

Ternel builds native, lossless execution formats for ternary language models.
`TQ1_G128` stores each group of 128 ternary weights in 28 bytes — two raw FP16
scale bytes plus 26 base-3 code bytes — for **1.75 bits/weight**, and custom
kernels execute the packed bytes directly: no weight is ever dequantized into a
model-sized buffer.

The shipped result is
[**Ternary Bonsai 27B — MLX, lossless, 1.75 bits/weight**](https://huggingface.co/inductiveML/Ternary-Bonsai-27B-mlx-lossless-1.75bpw):
the same ternary weights as PrismML's official distribution, verified bit-exact
over all 26,893,352,960 of them, emitting token-for-token identical greedy
output while holding 22.19% less live memory (5.89 GB against 7.57 GB during
generation, measured on an M4 Max).

## Run it

Any Apple Silicon Mac with 16 GB+ unified memory:

```bash
pip install mlx-lm
mlx_lm.generate --model inductiveML/Ternary-Bonsai-27B-mlx-lossless-1.75bpw \
  --prompt "Explain what a ternary weight is." --max-tokens 256
```

The checkpoint carries its own model code (`ternel_packed_model.py`, generated
from this repository); the model card documents the audit trail. From this
repository the equivalent entry point is:

```bash
uv run mlx-ternel-generate \
  --model inductiveML/Ternary-Bonsai-27B-mlx-lossless-1.75bpw \
  --prompt "Explain what a ternary weight is." \
  --max-tokens 256 --temperature 0.0 --chat-template
```

## The MLX result

Full narrative in [`reports_mlx/FINAL_REPORT.md`](reports_mlx/FINAL_REPORT.md);
every gate's machine-readable record is under [`results/mlx/`](results/mlx/).

- **Storage**: 5,888,388,744 bytes against the official mlx-2bit's
  8,490,785,104 (1.75 vs 2.25 bits/weight on quantised tensors).
- **Fidelity**: zero mismatches across 26,893,352,960 ternary weights and
  210,104,320 raw FP16 scale groups against the pinned source GGUF;
  teacher-forced logits argmax-identical at all 343 tested positions
  (max KL 4.8×10⁻⁵ nats); greedy generation token-identical, 576/576.
- **Memory**: 22.19% less live memory during generation.
- **Speed**: decode at 0.89–0.96× the official 2-bit checkpoint, prefill at
  0.64–0.87× — a memory-density result, not a speed win, stated as such.
- **Kernels**: three Metal kernels via `mx.fast.metal_kernel` (single-vector,
  batched with split-K, embedding lookup) executing packed base-3 bytes through
  a threadgroup LUT; the artifact ships a self-contained `model_file` so stock
  `mlx-lm >= 0.31.3` loads it with nothing else installed.

## The format

Each 28-byte `TQ1_G128` block holds the original two FP16 scale bytes and 26
base-3 bytes: 25 encode five trits each, the final byte encodes three. Decoding
rejects code bytes above 242, a tail byte above 26, and any non-ternary source
code, fail-closed. Conversion is a re-encoding of released weights — no
training, requantization, pruning, or full-matrix decompression anywhere.

## Reproduce and audit

Python 3.12 and every dependency are managed by
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra convert --frozen
uv run pytest
```

Conversion, exhaustive verification, and the benchmark gates live under
[`src/ternel_mlx/`](src/ternel_mlx/) (`convert.py`, `verify.py`,
`graph_gate.py`, `bench_model.py`); every argument is explicit and every stage
fails closed, capturing its environment — hardware, thermals, and GPU
contention — into its result document.

Pinned inputs:

- Model repository: `prism-ml/Ternary-Bonsai-27B-gguf`
- Model revision: `abbae723028d71be674e71e1a71201a6f43fab22`
- Model SHA-256: `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`
- Non-quantised tensors: `prism-ml/Ternary-Bonsai-27B-mlx-2bit`, cross-checked
  numerically against their GGUF F32 counterparts

## Earlier CUDA branch (frozen)

The format was first proven on CUDA against a patched llama.cpp. That branch
shipped a validated single-file GGUF (`Ternary-Bonsai-27B-TQ1_G128.gguf`,
5,904,496,192 bytes) saving 17.29% of the CUDA model buffer with full file
parity and greedy agreement, but no speed win — and two earlier kernel
hypotheses failed their own gates (`PACKING_ONLY_KERNEL_FAIL`,
`STREAMING_TQ1_FAIL`). Those reports are hash-frozen under
[`reports/`](reports/), [`reports_v2/`](reports_v2/), and
[`reports_dist/`](reports_dist/); the CUDA runner is
`uv run ternel-experiment all` and requires the exact pinned GPU.
