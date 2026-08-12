"""Gate B7: the whole packed model against the official checkpoint it copies.

Stage B6 proved the artifact *stores* the same weights as
``prism-ml/Ternary-Bonsai-27B-mlx-2bit`` -- bit for bit, over 26,893,352,960 of
them. That is a claim about bytes at rest, and it says nothing about whether the
Metal kernels reading those bytes compute the same function. This module asks
the second question by running both checkpoints through stock ``mlx_lm`` and
comparing what comes out.

The comparison is unusual in one respect that makes it much stronger than a
typical quantisation evaluation: **the two models hold identical weights**. There
is no approximation to budget for. Every difference measured here is
accumulation order and nothing else -- Ternel's LUT23 kernels decode trits and
sum them in slot order, MLX's ``quantized_matmul`` unpacks affine words and sums
through its own tiling -- so a divergence larger than fp32-over-5120-terms drift
is a defect, not a tradeoff. The one exception is stated where it arises: row
178519 of the embedding table, whose two encodings differ by 1.19e-07 in weight
space (see :mod:`ternel_mlx.crosscheck`).

Three things are measured, and they fail for different reasons:

**Teacher-forced logits.** One forward pass per prompt over the whole sequence.
Reported as raw logit distance *and* as KL divergence between the two next-token
distributions, because a logit gap matters only through the softmax and a model
whose logits drift uniformly is not wrong at all.

**Greedy agreement.** The token each model would actually emit at every position.
This is the user-visible claim, and with identical weights it should be perfect.
Where it is not, the disagreement is classified rather than merely counted: if
the reference's own top two logits at that position are within twice the drift
this kernel pair *routinely* produces, the flip is arithmetic noise landing on a
coin-toss the reference was already indifferent about, and it says nothing about
whether the kernels are right. Anything wider is a defect.

"Routinely" is the whole content of that rule, and it has to be measured away
from the flip itself. The obvious formulation -- compare the margin against the
drift at the flipped position -- is worthless: flipping an argmax across a
margin ``m`` *requires* an error above ``m/2`` at that position, so the test
passes by construction and would explain away every flip including the ones that
matter. The drift is therefore taken as the median, over every position of every
prompt, of the largest logit gap at that position. A median cannot be moved by
the handful of positions under suspicion, so a genuinely broken kernel produces
anomalous drift that its own yardstick refuses to excuse.

**Free-running continuation.** What each model generates when it is fed its own
output rather than a shared context. This is reported as a *common prefix* and
gated only on its first few tokens, and the asymmetry is deliberate: past the
first divergence the two models are answering different questions -- each is
conditioned on a context the other never saw -- so an agreement rate measured
beyond that point is not a statement about the kernels. The teacher-forced
numbers are where the arithmetic is actually gated; the prefix is where the
behaviour is.

**Representation.** That the packed model really is packed -- that the graph
holds ``uint8`` codes and ``uint16`` scales and no model-sized float array
anywhere, and that its parameters weigh what the manifest says. A correctness
gate on a model that silently dequantised itself would pass for the wrong reason.

Every threshold is supplied by the caller. The llama.cpp CUDA figures from the
earlier phase (max abs 0.1214711666, cosine 0.9999877864, 33/33 argmax) are a
different runtime on different hardware and are context, not acceptance limits.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.utils import load_model, load_tokenizer

from bonsai_tq1.format import FormatError, write_json_atomic

from . import LAYOUT_NAME, LAYOUT_VERSION
from .convert import MANIFEST_NAME
from .environment import capture_environment
from .kernel_gate import compare
from .modules import PackedEmbedding, PackedLinear

SCHEMA_VERSION = 1

# Fixed prompts, spanning what changes which parts of the graph: short and long
# contexts (the recurrent layers' state has to be built either way), factual
# recall (dense embedding lookups), code and markup (a different token
# distribution), non-English text (the far end of a 248320-token vocabulary),
# and repetition, where a small logit gap is most likely to flip an argmax
# because the top candidates are nearly tied.
PROMPTS: tuple[tuple[str, str], ...] = (
    ("factual_short", "The capital of France is"),
    (
        "factual_long",
        "The Apollo program was the third United States human spaceflight program "
        "carried out by NASA, which accomplished landing the first humans on the "
        "Moon. Conceived during the presidency of Dwight D. Eisenhower, the program "
        "was dedicated to the national goal of",
    ),
    (
        "reasoning",
        "A farmer has 17 sheep. All but 9 run away. The number of sheep the farmer "
        "has left is",
    ),
    (
        "code_python",
        "def binary_search(values, target):\n"
        "    low, high = 0, len(values) - 1\n"
        "    while low <= high:\n"
        "        middle = (low + high) // 2\n",
    ),
    (
        "code_c",
        "#include <stdio.h>\n\nint main(void) {\n    for (int i = 0; i < 10; i++) {\n",
    ),
    ("markup_json", '{"model": "Ternary Bonsai", "parameters": 27000000000, "bits":'),
    ("multilingual_fr", "La ville de Paris est la capitale de la"),
    ("multilingual_zh", "中华人民共和国的首都是"),
    ("multilingual_ja", "日本の首都は東京です。日本で二番目に大きい都市は"),
    ("repetition", "one two three one two three one two three one two"),
    (
        "long_context",
        "In the beginning the Universe was created. This has made a lot of people "
        "very angry and been widely regarded as a bad move. Many were increasingly "
        "of the opinion that they had all made a big mistake in coming down from "
        "the trees in the first place. And some said that even the trees had been a "
        "bad move, and that no one should ever have left the oceans. The story so "
        "far: in the beginning the Universe was created. This has made a lot of "
        "people very angry and been widely regarded as a",
    ),
    ("numeric", "2 4 8 16 32 64 128 256 512"),
)

# What is compared per prompt, so a report can name the position that diverged.
POSITION_LABEL = "position_{index}"

# Top-k depths the overlap is reported at. 1 is the greedy decision; 5 and 10 are
# what a sampler would actually draw from, so they say whether the two models
# would behave the same under temperature as well as under argmax.
TOP_K = (1, 5, 10)

# Parameter dtypes the packed graph is allowed to hold, and what each is for. A
# dtype outside this set means something was dequantised on the way in.
PACKED_DTYPES = {"uint8": "ternary codes", "uint16": "raw FP16 scale bits"}

# The largest float parameter a correctly packed graph can hold: the norms and
# the SSM parameters, none of which is a matmul weight. ``ssm_conv1d`` is the
# widest at 2 x 12288 x 4, so anything past a million elements is a matmul weight
# that failed to pack.
MAX_FLOAT_PARAMETER_ELEMENTS = 1 << 20


def argmax_sampler(logprobs: mx.array) -> mx.array:
    """Greedy sampling, stated rather than left to ``generate_step``'s default."""
    return mx.argmax(logprobs, axis=-1)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Stable log-softmax in float64, over the last axis."""
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def kl_divergence(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> np.ndarray:
    """``KL(reference || candidate)`` in nats, per position.

    The direction is chosen deliberately: the official checkpoint is the
    reference distribution, so this measures the information lost by sampling
    from Ternel's distribution when the official one is what was meant. Mass the
    reference assigns to a token is what gets weighted, which is the right way
    round for a question about behaviour.
    """
    reference = log_softmax(reference_logits)
    candidate = log_softmax(candidate_logits)
    return np.sum(np.exp(reference) * (reference - candidate), axis=-1)


def top_k_overlap(reference_logits: np.ndarray, candidate_logits: np.ndarray, k: int) -> np.ndarray:
    """The fraction of each position's top ``k`` tokens the two models share."""
    if k < 1:
        raise FormatError(f"top-k depth must be positive, got {k}")
    reference = np.argpartition(-reference_logits, kth=k - 1, axis=-1)[:, :k]
    candidate = np.argpartition(-candidate_logits, kth=k - 1, axis=-1)[:, :k]
    return np.array(
        [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(reference, candidate)]
    )


@dataclass(frozen=True)
class PromptRun:
    """One model's output for one prompt: every logit, and the greedy tokens."""

    label: str
    tokens: tuple[int, ...]
    logits: np.ndarray
    continuation: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.logits.shape[0] != len(self.tokens):
            raise FormatError(
                f"{self.label}: {self.logits.shape[0]} logit rows for {len(self.tokens)} tokens"
            )


def teacher_forced_logits(model, tokens: tuple[int, ...]) -> np.ndarray:
    """Logits at every position of ``tokens``, from a single forward pass.

    No cache: the whole sequence goes through in one call, which is the prefill
    path rather than the decode path, and is what a batched kernel gets wrong if
    it gets anything wrong.
    """
    out = model(mx.array(tokens, dtype=mx.int32)[None])
    mx.eval(out)
    if out.shape[0] != 1 or out.shape[1] != len(tokens):
        raise FormatError(f"expected (1, {len(tokens)}, vocab) logits, got {tuple(out.shape)}")
    return np.array(out[0].astype(mx.float32), copy=False).copy()


def greedy_continuation(model, tokens: tuple[int, ...], count: int) -> tuple[int, ...]:
    """``count`` tokens of argmax decoding, with no early stop.

    Stopping at an end-of-sequence token would make the two models comparable
    only up to whichever stopped first, so nothing stops: the comparison is over
    a fixed number of steps either way.
    """
    if count < 1:
        raise FormatError(f"continuation length must be positive, got {count}")
    produced: list[int] = []
    for token, _ in generate_step(
        mx.array(tokens, dtype=mx.int32), model, max_tokens=count, sampler=argmax_sampler
    ):
        produced.append(int(token))
    if len(produced) != count:
        raise FormatError(f"asked for {count} tokens, generation yielded {len(produced)}")
    return tuple(produced)


def parameter_census(model) -> dict[str, object]:
    """What the loaded graph actually holds, by dtype and by module kind.

    This is the anti-cheat assertion the plan asks for, in the form that can
    still be checked after ``load_model`` has done whatever it was going to do.
    A packed graph that had been dequantised would still answer every logit
    question correctly and would fail here.
    """
    from mlx.utils import tree_flatten

    by_dtype: dict[str, dict[str, int]] = {}
    largest_float = {"name": "", "elements": 0}
    for name, array in tree_flatten(model.parameters()):
        tag = str(array.dtype).removeprefix("mlx.core.")
        entry = by_dtype.setdefault(tag, {"tensors": 0, "elements": 0, "bytes": 0})
        entry["tensors"] += 1
        entry["elements"] += int(array.size)
        entry["bytes"] += int(array.nbytes)
        if tag not in PACKED_DTYPES and int(array.size) > largest_float["elements"]:
            largest_float = {"name": name, "elements": int(array.size)}

    modules = {"PackedLinear": 0, "PackedEmbedding": 0}
    for _, module in model.named_modules():
        if isinstance(module, PackedLinear):
            modules["PackedLinear"] += 1
        elif isinstance(module, PackedEmbedding):
            modules["PackedEmbedding"] += 1

    return {
        "by_dtype": by_dtype,
        "packed_dtypes": sorted(PACKED_DTYPES),
        "packed_bytes": sum(
            entry["bytes"] for tag, entry in by_dtype.items() if tag in PACKED_DTYPES
        ),
        "unpacked_bytes": sum(
            entry["bytes"] for tag, entry in by_dtype.items() if tag not in PACKED_DTYPES
        ),
        "total_bytes": sum(entry["bytes"] for entry in by_dtype.values()),
        "packed_modules": modules,
        "largest_unpacked_parameter": largest_float,
        "max_unpacked_parameter_elements": MAX_FLOAT_PARAMETER_ELEMENTS,
    }


def run_model(
    model_path: Path, *, prompts: tuple[tuple[str, str], ...], continuation_tokens: int
) -> tuple[list[PromptRun], dict[str, object]]:
    """Load one checkpoint, answer every prompt, and report what it weighed."""
    started = time.perf_counter()
    mx.reset_peak_memory()
    model, config = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    mx.eval(model.parameters())
    loaded = time.perf_counter() - started

    runs: list[PromptRun] = []
    for label, text in prompts:
        tokens = tuple(int(token) for token in tokenizer.encode(text))
        if not tokens:
            raise FormatError(f"prompt {label!r} tokenised to nothing")
        runs.append(
            PromptRun(
                label=label,
                tokens=tokens,
                logits=teacher_forced_logits(model, tokens),
                continuation=greedy_continuation(model, tokens, continuation_tokens),
            )
        )

    census = parameter_census(model)
    report: dict[str, object] = {
        "path": str(model_path),
        "load_seconds": loaded,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "quantization_declared": bool(
            "quantization" in config
            or "quantization_config" in config
            or "quantization_config" in config.get("text_config", {})
        ),
        "census": census,
        "vocabulary": int(runs[0].logits.shape[1]),
    }
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return runs, report


@dataclass(frozen=True)
class PromptComparison:
    """Ternel against the official checkpoint, for one prompt."""

    label: str
    tokens: int
    logits: dict[str, object]
    greedy_agreements: int
    greedy_positions: int
    drift_explained_flips: int
    unexplained_flips: int
    smallest_flipped_margin: float | None
    max_kl_divergence: float
    mean_kl_divergence: float
    top_k_overlap: dict[str, float]
    reference_nll: float | None
    candidate_nll: float | None
    continuation_common_prefix: int
    continuation_agreements: int
    continuation_length: int
    first_greedy_disagreement: int | None

    @property
    def nll_difference(self) -> float | None:
        """How much likelihood the packed model loses on the prompt itself."""
        if self.candidate_nll is None or self.reference_nll is None:
            return None
        return self.candidate_nll - self.reference_nll

    @property
    def greedy_agreement(self) -> float:
        return self.greedy_agreements / self.greedy_positions

    @property
    def continuation_agreement(self) -> float:
        return self.continuation_agreements / self.continuation_length

    def to_json(self) -> dict[str, object]:
        return {
            "label": self.label,
            "tokens": self.tokens,
            "logits": self.logits,
            "greedy_agreements": self.greedy_agreements,
            "greedy_positions": self.greedy_positions,
            "greedy_agreement": self.greedy_agreement,
            "first_greedy_disagreement": self.first_greedy_disagreement,
            "drift_explained_flips": self.drift_explained_flips,
            "unexplained_flips": self.unexplained_flips,
            "smallest_flipped_margin": self.smallest_flipped_margin,
            "max_kl_divergence": self.max_kl_divergence,
            "mean_kl_divergence": self.mean_kl_divergence,
            "top_k_overlap": self.top_k_overlap,
            "reference_nll": self.reference_nll,
            "candidate_nll": self.candidate_nll,
            "nll_difference": self.nll_difference,
            "continuation_length": self.continuation_length,
            "continuation_common_prefix": self.continuation_common_prefix,
            "continuation_agreements": self.continuation_agreements,
            "continuation_agreement": self.continuation_agreement,
        }


def position_drift(candidate: PromptRun, reference: PromptRun) -> np.ndarray:
    """The largest logit gap at each position of one prompt."""
    return np.abs(
        candidate.logits.astype(np.float64) - reference.logits.astype(np.float64)
    ).max(axis=1)


def characteristic_drift(candidates: list[PromptRun], references: list[PromptRun]) -> float:
    """The drift this pair of kernels routinely produces, over every position.

    A median rather than a mean or a maximum: the statistic exists to judge a
    small number of suspect positions, so it must be one those positions cannot
    move.
    """
    drifts = np.concatenate(
        [position_drift(candidate, reference) for candidate, reference in zip(candidates, references)]
    )
    return float(np.median(drifts))


def compare_prompt(
    candidate: PromptRun,
    reference: PromptRun,
    *,
    logit_tolerance: float,
    typical_drift: float,
) -> PromptComparison:
    if candidate.label != reference.label or candidate.tokens != reference.tokens:
        raise FormatError(
            f"{candidate.label}: the two models were not asked the same question; "
            "the artifact's tokenizer must be the checkpoint's"
        )
    got, want = candidate.logits, reference.logits
    labels = tuple(POSITION_LABEL.format(index=index) for index in range(got.shape[0]))
    statistics = compare(
        got, want, name="mlx_2bit", tolerance=logit_tolerance, labels=labels
    ).to_json()

    greedy_candidate = np.argmax(got, axis=-1)
    greedy_reference = np.argmax(want, axis=-1)
    same = greedy_candidate == greedy_reference
    disagreements = np.flatnonzero(~same)

    # A flip is explained by drift when the reference's own margin between its
    # top two tokens is within twice the drift this kernel pair routinely
    # produces elsewhere. See the module docstring for why the yardstick cannot
    # be the drift at the flipped position itself.
    margins = np.array(
        [float(np.diff(np.sort(want[position])[-2:])[0]) for position in disagreements]
    )
    explained = margins <= 2.0 * typical_drift

    divergences = kl_divergence(want, got)

    # Teacher-forced negative log likelihood of the token that actually follows.
    # The final position predicts nothing inside the prompt, so it is excluded
    # rather than scored against a token that does not exist -- which leaves a
    # single-token prompt with nothing to score at all, reported as absent
    # rather than as the NaN that averaging no numbers produces.
    targets = np.array(candidate.tokens[1:], dtype=np.int64)
    positions = np.arange(targets.size)
    candidate_nll = (
        float(-log_softmax(got[:-1])[positions, targets].mean()) if targets.size else None
    )
    reference_nll = (
        float(-log_softmax(want[:-1])[positions, targets].mean()) if targets.size else None
    )

    shared = 0
    for a, b in zip(candidate.continuation, reference.continuation):
        if a != b:
            break
        shared += 1

    return PromptComparison(
        label=candidate.label,
        tokens=len(candidate.tokens),
        logits=statistics,
        greedy_agreements=int(np.count_nonzero(same)),
        greedy_positions=int(same.size),
        drift_explained_flips=int(np.count_nonzero(explained)),
        unexplained_flips=int(np.count_nonzero(~explained)),
        smallest_flipped_margin=float(margins.min()) if margins.size else None,
        max_kl_divergence=float(divergences.max()),
        mean_kl_divergence=float(divergences.mean()),
        top_k_overlap={
            f"top_{k}": float(top_k_overlap(want, got, k).mean()) for k in TOP_K
        },
        reference_nll=reference_nll,
        candidate_nll=candidate_nll,
        continuation_common_prefix=shared,
        continuation_agreements=int(
            sum(a == b for a, b in zip(candidate.continuation, reference.continuation))
        ),
        continuation_length=len(candidate.continuation),
        first_greedy_disagreement=int(disagreements[0]) if disagreements.size else None,
    )


def check_representation(report: dict[str, object], artifact_dir: Path) -> list[str]:
    """The anti-cheat assertions, as a list of what failed."""
    census = report["census"]
    problems: list[str] = []
    if report["quantization_declared"]:
        problems.append(
            "the artifact config declares a quantisation, so mlx-lm would have called "
            "nn.quantize and built an affine copy of every weight"
        )
    if census["packed_modules"]["PackedEmbedding"] != 1:
        problems.append(
            f"expected one packed embedding, the graph holds "
            f"{census['packed_modules']['PackedEmbedding']}"
        )
    if census["packed_modules"]["PackedLinear"] < 1:
        problems.append("the graph holds no packed linear layers")
    if census["largest_unpacked_parameter"]["elements"] > MAX_FLOAT_PARAMETER_ELEMENTS:
        problems.append(
            f"{census['largest_unpacked_parameter']['name']} is an unpacked parameter of "
            f"{census['largest_unpacked_parameter']['elements']} elements, which is a "
            "matmul weight that did not pack"
        )

    manifest = json.loads((artifact_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    expected = int(manifest["quantised"]["payload_bytes"])
    if census["packed_bytes"] != expected:
        problems.append(
            f"the graph holds {census['packed_bytes']} packed bytes, the manifest "
            f"records {expected}"
        )
    return problems


def gate(
    artifact_dir: Path,
    baseline_dir: Path,
    *,
    continuation_tokens: int,
    logit_tolerance: float,
    max_kl_divergence: float,
    min_greedy_agreement: float,
    min_continuation_prefix: int,
) -> dict[str, object]:
    environment = capture_environment()

    # Sequentially, never together: two 27B checkpoints resident at once would
    # measure this machine's memory pressure rather than either model.
    print("  loading the packed artifact", file=sys.stderr, flush=True)
    candidate_runs, candidate_report = run_model(
        artifact_dir, prompts=PROMPTS, continuation_tokens=continuation_tokens
    )
    print("  loading the official 2-bit checkpoint", file=sys.stderr, flush=True)
    reference_runs, reference_report = run_model(
        baseline_dir, prompts=PROMPTS, continuation_tokens=continuation_tokens
    )

    typical_drift = characteristic_drift(candidate_runs, reference_runs)
    comparisons = [
        compare_prompt(
            candidate,
            reference,
            logit_tolerance=logit_tolerance,
            typical_drift=typical_drift,
        )
        for candidate, reference in zip(candidate_runs, reference_runs)
    ]

    problems = check_representation(candidate_report, artifact_dir)
    positions = sum(item.greedy_positions for item in comparisons)
    agreements = sum(item.greedy_agreements for item in comparisons)
    continuation_positions = sum(item.continuation_length for item in comparisons)
    continuation_agreements = sum(item.continuation_agreements for item in comparisons)
    shortest_prefix = min(item.continuation_common_prefix for item in comparisons)
    worst_kl = max(item.max_kl_divergence for item in comparisons)
    worst_drift = max(float(item.logits["normalised_max_abs_error"]) for item in comparisons)
    greedy_agreement = agreements / positions
    unexplained = sum(item.unexplained_flips for item in comparisons)
    logits_within_tolerance = all(bool(item.logits["passed"]) for item in comparisons)
    verdict = {
        "representation_packed": not problems,
        "no_unexplained_greedy_flips": unexplained == 0,
        "greedy_agreement_met": greedy_agreement >= min_greedy_agreement,
        "continuation_prefix_met": shortest_prefix >= min(
            min_continuation_prefix, continuation_tokens
        ),
        "continuation_identical": continuation_agreements == continuation_positions,
        "kl_within_bound": worst_kl <= max_kl_divergence,
        "logit_drift_within_tolerance": logits_within_tolerance,
    }
    # ``continuation_identical`` is recorded but not gated -- see the module
    # docstring: past the first divergence the two models are conditioned on
    # different contexts, so equality there is not a claim about the kernels.
    verdict["pass"] = all(
        value for key, value in verdict.items() if key != "continuation_identical"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "layout_name": LAYOUT_NAME,
        "layout_version": LAYOUT_VERSION,
        "what_this_measures": (
            "the two checkpoints hold bit-identical weights (Stage B6), so every "
            "difference below is kernel accumulation order, not quantisation error"
        ),
        "candidate": candidate_report,
        "reference": reference_report,
        "prompts": [label for label, _ in PROMPTS],
        "continuation_tokens": continuation_tokens,
        "thresholds": {
            "logit_tolerance": logit_tolerance,
            "max_kl_divergence": max_kl_divergence,
            "min_greedy_agreement": min_greedy_agreement,
            "min_continuation_prefix": min_continuation_prefix,
        },
        "totals": {
            "prompts": len(comparisons),
            "teacher_forced_positions": positions,
            "greedy_agreements": agreements,
            "greedy_agreement": greedy_agreement,
            "characteristic_drift": typical_drift,
            "drift_explained_flips": sum(item.drift_explained_flips for item in comparisons),
            "unexplained_flips": unexplained,
            "smallest_flipped_margin": min(
                (
                    item.smallest_flipped_margin
                    for item in comparisons
                    if item.smallest_flipped_margin is not None
                ),
                default=None,
            ),
            "continuation_positions": continuation_positions,
            "continuation_agreements": continuation_agreements,
            "continuation_agreement": continuation_agreements / continuation_positions,
            "shortest_continuation_prefix": shortest_prefix,
            "max_kl_divergence": worst_kl,
            "mean_kl_divergence": float(
                np.average(
                    [item.mean_kl_divergence for item in comparisons],
                    weights=[item.greedy_positions for item in comparisons],
                )
            ),
            "max_normalised_logit_drift": worst_drift,
            "max_abs_logit_error": max(
                float(item.logits["max_abs_error"]) for item in comparisons
            ),
            "min_cosine_similarity": min(
                float(item.logits["min_cosine_similarity"]) for item in comparisons
            ),
            "max_nll_difference": max(
                (
                    abs(item.nll_difference)
                    for item in comparisons
                    if item.nll_difference is not None
                ),
                default=None,
            ),
        },
        "per_prompt": [item.to_json() for item in comparisons],
        "representation_problems": problems,
        "environment": environment,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate the packed model end to end (B7)")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuation-tokens", type=int, required=True)
    parser.add_argument(
        "--logit-tolerance",
        type=float,
        required=True,
        help="largest logit gap, normalised by the position's largest reference logit",
    )
    parser.add_argument("--max-kl-divergence", type=float, required=True)
    parser.add_argument("--min-greedy-agreement", type=float, required=True)
    parser.add_argument(
        "--min-continuation-prefix",
        type=int,
        required=True,
        help="generated tokens every prompt must agree on before free-running divergence",
    )
    args = parser.parse_args(argv)

    payload = gate(
        args.artifact,
        args.baseline,
        continuation_tokens=args.continuation_tokens,
        logit_tolerance=args.logit_tolerance,
        max_kl_divergence=args.max_kl_divergence,
        min_greedy_agreement=args.min_greedy_agreement,
        min_continuation_prefix=args.min_continuation_prefix,
    )
    write_json_atomic(args.output, payload)

    totals = payload["totals"]
    print(
        f"greedy:  {totals['greedy_agreements']}/{totals['teacher_forced_positions']} "
        f"teacher-forced positions, {totals['continuation_agreements']}/"
        f"{totals['continuation_positions']} generated tokens, shortest common "
        f"prefix {totals['shortest_continuation_prefix']}"
    )
    print(
        f"flips:   {totals['unexplained_flips']} unexplained, "
        f"{totals['drift_explained_flips']} explained by drift, smallest flipped "
        f"reference margin {totals['smallest_flipped_margin']}"
    )
    print(
        f"logits:  max abs {totals['max_abs_logit_error']:.6g}, normalised "
        f"{totals['max_normalised_logit_drift']:.6g}, min cosine "
        f"{totals['min_cosine_similarity']:.10f}"
    )
    print(
        f"kl:      max {totals['max_kl_divergence']:.6g} nats, mean "
        f"{totals['mean_kl_divergence']:.6g}, max |dNLL| {totals['max_nll_difference']}"
    )
    print(
        f"memory:  packed {payload['candidate']['census']['packed_bytes']} bytes, "
        f"unpacked {payload['candidate']['census']['unpacked_bytes']}, "
        f"active {payload['candidate']['active_memory_bytes']} vs baseline "
        f"{payload['reference']['active_memory_bytes']}"
    )
    if payload["representation_problems"]:
        print(f"PROBLEMS: {payload['representation_problems']}")
    print(f"pass = {payload['verdict']['pass']}")
    return 0 if payload["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
