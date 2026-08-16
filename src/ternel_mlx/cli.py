"""``mlx-ternel-generate`` -- run a packed TQ1_G128 model from the command line.

The artifact carries a self-contained ``model_file`` emitted by
:mod:`ternel_mlx.model_file`, so ``mlx_lm.load`` builds the packed graph on its
own without this package installed and everything downstream
of it -- the tokenizer, the chat template, the sampler, the KV cache -- is stock
mlx-lm. This module is therefore deliberately thin: its job is to be the entry
point ``uvx --from ternel`` exposes, not to reimplement generation.

No argument has a default. Which model, which prompt and how many tokens are all
decisions this tool refuses to make on the caller's behalf.
"""

from __future__ import annotations

import argparse
import sys

import mlx.core as mx
import mlx.nn as nn

from bonsai_tq1.format import FormatError


def count_packed_modules(model: nn.Module) -> tuple[int, int]:
    """Count packed leaves and the bytes they hold.

    Reported before generation because it is the claim under test: if a
    dequantised copy had been made anywhere, the byte count here would not be
    the artifact's payload size.

    The match is on class name and buffer dtypes rather than on
    ``ternel_mlx.modules``'s class objects: the artifact's ``model_file`` is
    self-contained, so the classes in a graph it loaded are its own.
    """
    modules = 0
    payload = 0
    for name, module in model.named_modules():
        if type(module).__name__ not in ("PackedLinear", "PackedEmbedding"):
            continue
        codes = getattr(module, "codes", None)
        scales = getattr(module, "scales", None)
        if not isinstance(codes, mx.array) or codes.dtype != mx.uint8:
            raise FormatError(f"{name} has no uint8 ternary codes")
        if not isinstance(scales, mx.array) or scales.dtype != mx.uint16:
            raise FormatError(f"{name} has no uint16 scale bits")
        modules += 1
        payload += codes.nbytes + scales.nbytes
    return modules, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlx-ternel-generate",
        description="Generate text from a packed TQ1_G128 model with MLX",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face repo id or local path of the packed artifact",
    )
    parser.add_argument("--prompt", required=True, help="the prompt to generate from")
    parser.add_argument(
        "--max-tokens", type=int, required=True, help="how many tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="sampling temperature; 0.0 for greedy decoding",
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="wrap the prompt in the tokenizer's chat template",
    )
    args = parser.parse_args(argv)

    if args.max_tokens <= 0:
        raise FormatError(f"--max-tokens must be positive, got {args.max_tokens}")
    if args.temperature < 0.0:
        raise FormatError(f"--temperature cannot be negative, got {args.temperature}")

    # Imported here rather than at module scope: mlx_lm pulls in transformers,
    # which is slow enough that an --help should not pay for it.
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)

    modules, payload = count_packed_modules(model)
    print(
        f"packed modules: {modules}, weight bytes resident: {payload:,} "
        f"({payload / 2**30:.2f} GiB)",
        file=sys.stderr,
    )

    prompt = args.prompt
    if args.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )

    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=args.temperature),
        verbose=True,
    )

    peak = mx.get_peak_memory()
    print(f"peak MLX memory: {peak:,} bytes ({peak / 2**30:.2f} GiB)", file=sys.stderr)

    # ``generate`` returns None when the model emitted nothing at all. That is a
    # legitimate outcome of the library call, but it is a failed smoke test: the
    # point of this command is to demonstrate that packed weights generate text.
    if not text:
        print("no text was generated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
