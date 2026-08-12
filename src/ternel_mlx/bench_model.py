"""Stage B8: what the packed model costs, end to end, against the checkpoint it copies.

Stage A5 measured single operators. This measures the thing a user actually
waits for -- a prompt going in and tokens coming out of a 27B model -- and the
memory it takes to do it, against ``prism-ml/Ternary-Bonsai-27B-mlx-2bit``
running through stock ``mlx_lm``.

The two questions need opposite setups, so they get separate phases rather than
one convenient run:

**Memory** is measured with one model resident and nothing else, in its own
subprocess. Two 27B checkpoints in one process would report the allocator's high
water mark for the pair, which is not a number anybody needs. The subprocess also
makes process footprint meaningful and gives cold start -- load, first Metal JIT,
first token -- honestly, since a warm process has already paid all three.

**Throughput** is measured twice, because the obvious design is suspect. Holding
both models in one process and alternating AB/BA defeats thermal drift, and 14 GB
against a 40.2 GB recommended working set is not an allocation problem -- but
decode is memory-bound, and a co-resident 27B model is cache pressure whether or
not it is running. If that costs the two arms differently -- and the affine model
is the larger of the two, so it has more to lose -- the paired ratio flatters the
packed arm and is not the ratio a user would see.

So the reported throughput comes from ``--isolated-output``: one model per
process, arms alternating round by round, nothing else resident. Thermal drift is
still handled -- by the alternation of rounds rather than of calls -- and the
number means what it says. The paired run is kept under ``--output`` as the
evidence for the comparison, labelled as co-resident rather than deleted.

Both modes refuse to report a number measured while another process was on the
GPU: see :func:`exclusive_gpu`, and ``--max-foreign-gpu-share``, which has no
default because a timing run has no business guessing what it tolerates.

Method inherited from A5 and the CUDA phase before it: warm before timing,
medians with p5/p95 rather than means, a paired bootstrap interval on every
ratio so one that straddles 1.0 is visibly not a result, and an anti-cheat pass
that fails the run rather than footnoting it. The strongest of those checks here
is free: greedy decoding is deterministic, so every trial of an arm must produce
the *same tokens*, and both arms must produce the same tokens as each other.
A kernel that got fast by skipping work cannot also keep saying the same thing.

Prompt processing is reported as time-to-first-token, which includes the single
decode step that produces that token. That slightly understates the pure prefill
rate, identically for both arms, so the ratio is unaffected and the absolute
number is the one a user experiences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.utils import load_model, load_tokenizer

from bonsai_tq1.format import FormatError, write_json_atomic
from bonsai_tq1.lut23_reporting import paired_bootstrap_ratio, quantile_summary

from . import LAYOUT_NAME, LAYOUT_VERSION
from .environment import capture_environment, exclusive_gpu
from .graph_gate import PROMPTS, argmax_sampler, check_representation, parameter_census

SCHEMA_VERSION = 1

# Fixed so a rerun resamples the same bootstrap, as in A5.
SEED = 20260811

# Enough paired trials for ``paired_bootstrap_ratio``, which refuses fewer than
# 30 -- an interval built from a handful of samples is decoration.
TRIALS = 30

# Two full passes over every workload before the first timed one, so neither
# Metal JIT nor a cold buffer cache lands inside a measurement.
WARMUP_ROUNDS = 2

# The isolated run pays its warmup once per subprocess rather than once per
# measurement, so it takes one pass instead of two: a fresh process has to
# compile its Metal libraries and fault its weights in, and one full pass over
# every workload does both.
SOLO_WARMUP_ROUNDS = 1

# Warm runs the probe takes after its cold one. More than one, because the
# difference between a single cold sample and a single warm sample on a
# three-second workload is dominated by run-to-run noise -- the first
# measurement of this returned a *negative* cold-start cost. A5 measured
# Metal JIT in isolation; what is wanted here is only whether cold start is
# visible above that noise at all.
PROBE_WARM_RUNS = 3

# How often the parent samples a probe subprocess's resident size. macOS exposes
# no per-process GPU accounting, so RSS is the honest proxy and is labelled as
# one in the emitted document.
RSS_POLL_SECONDS = 0.25


@dataclass(frozen=True)
class Workload:
    """One prompt length and one generation length, timed as a single run."""

    label: str
    prompt_tokens: int
    generate_tokens: int

    def __post_init__(self) -> None:
        if self.prompt_tokens < 1:
            raise FormatError(f"{self.label}: a prompt of {self.prompt_tokens} tokens is not one")
        if self.generate_tokens < 2:
            raise FormatError(
                f"{self.label}: the decode rate is measured after the first token, so "
                f"{self.generate_tokens} tokens leaves nothing to time"
            )


# Short, medium and long prompts against a fixed generation length. The three
# prompt lengths are what move the batched path: 32 tokens is one small GEMM per
# layer, 2048 is the prefill regime the CUDA integration never had.
WORKLOADS: tuple[Workload, ...] = (
    Workload(label="pp32_tg64", prompt_tokens=32, generate_tokens=64),
    Workload(label="pp512_tg64", prompt_tokens=512, generate_tokens=64),
    Workload(label="pp2048_tg64", prompt_tokens=2048, generate_tokens=64),
)


def corpus_tokens(tokenizer, *, at_least: int) -> tuple[int, ...]:
    """A token sequence at least ``at_least`` long, built from the B7 prompts.

    Reusing them keeps the benchmark reading the same text the correctness gate
    did, and prefill cost is a function of shape rather than content, so the
    repetition needed to reach 2048 tokens costs nothing in representativeness.
    """
    passage = "\n\n".join(text for _, text in PROMPTS)
    tokens: list[int] = []
    while len(tokens) < at_least:
        tokens.extend(int(token) for token in tokenizer.encode(passage))
    return tuple(tokens[:at_least])


def common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """How far two greedy continuations agree before diverging."""
    shared = 0
    for a, b in zip(left, right):
        if a != b:
            break
        shared += 1
    return shared


def timed_generation(model, tokens: tuple[int, ...], *, generate_tokens: int):
    """Prefill seconds, decode seconds, and every token produced.

    The clock is read only after a host-side read of a token has forced the
    queued work to complete, so neither segment can borrow time from the other.
    """
    mx.synchronize()
    start = time.perf_counter()
    first: float | None = None
    produced: list[int] = []
    for token, _ in generate_step(
        mx.array(tokens, dtype=mx.int32),
        model,
        max_tokens=generate_tokens,
        sampler=argmax_sampler,
    ):
        produced.append(int(token))
        if first is None:
            first = time.perf_counter()
    mx.synchronize()
    end = time.perf_counter()
    if first is None or len(produced) != generate_tokens:
        raise FormatError(f"asked for {generate_tokens} tokens, generation yielded {len(produced)}")
    return first - start, end - first, tuple(produced)


def rates(workload: Workload, prefill: float, decode: float) -> dict[str, float]:
    return {
        "time_to_first_token_seconds": prefill,
        "decode_seconds": decode,
        "prompt_tokens_per_second": workload.prompt_tokens / prefill,
        # The first token was produced inside the prefill segment, so the decode
        # segment covers the remaining ones and is divided by that count.
        "generated_tokens_per_second": (workload.generate_tokens - 1) / decode,
    }


def measure_workload(
    workload: Workload,
    *,
    candidate,
    reference,
    prompt: tuple[int, ...],
    trials: int,
) -> dict[str, object]:
    """Both arms, alternating, over ``trials`` paired trials."""
    for _ in range(WARMUP_ROUNDS):
        timed_generation(candidate, prompt, generate_tokens=workload.generate_tokens)
        timed_generation(reference, prompt, generate_tokens=workload.generate_tokens)

    arms: dict[str, list[tuple[float, float]]] = {"candidate": [], "reference": []}
    produced: dict[str, set[tuple[int, ...]]] = {"candidate": set(), "reference": set()}
    for trial in range(trials):
        order = ("candidate", "reference") if trial % 2 == 0 else ("reference", "candidate")
        for arm in order:
            model = candidate if arm == "candidate" else reference
            prefill, decode, tokens = timed_generation(
                model, prompt, generate_tokens=workload.generate_tokens
            )
            arms[arm].append((prefill, decode))
            produced[arm].add(tokens)

    # Greedy decoding is a function, so an arm that produced two different
    # answers for one prompt was not measured -- something else was running, or
    # the kernel is not deterministic. Either way the timings mean nothing.
    for arm, answers in produced.items():
        if len(answers) != 1:
            raise FormatError(
                f"{workload.label}: the {arm} arm produced {len(answers)} distinct greedy "
                "continuations across identical trials"
            )
    candidate_tokens = next(iter(produced["candidate"]))
    reference_tokens = next(iter(produced["reference"]))

    measured: dict[str, object] = {
        "label": workload.label,
        "prompt_tokens": workload.prompt_tokens,
        "generate_tokens": workload.generate_tokens,
        "trials": trials,
        "greedy_continuations_identical": candidate_tokens == reference_tokens,
        "greedy_common_prefix": common_prefix_length(candidate_tokens, reference_tokens),
    }
    for segment, index in (("prefill", 0), ("decode", 1)):
        base = [entry[index] for entry in arms["reference"]]
        cand = [entry[index] for entry in arms["candidate"]]
        measured[segment] = {
            "packed_seconds": quantile_summary(np.asarray(cand, dtype=np.float64)),
            "affine_2bit_seconds": quantile_summary(np.asarray(base, dtype=np.float64)),
            "speedup_vs_affine_2bit": float(np.median(base) / np.median(cand)),
            # The bootstrap is over candidate/reference, so its median is the
            # reciprocal of the speedup above; both are reported because a
            # reader checking whether an interval straddles 1.0 should not have
            # to invert an interval in their head.
            "bootstrap_candidate_over_reference": paired_bootstrap_ratio(
                base, cand, samples=100_000, seed=SEED
            ),
        }
    measured["packed_rates"] = rates(
        workload,
        float(np.median([entry[0] for entry in arms["candidate"]])),
        float(np.median([entry[1] for entry in arms["candidate"]])),
    )
    measured["affine_2bit_rates"] = rates(
        workload,
        float(np.median([entry[0] for entry in arms["reference"]])),
        float(np.median([entry[1] for entry in arms["reference"]])),
    )
    return measured


def probe(model_path: Path, *, workload: Workload) -> dict[str, object]:
    """Everything measurable with one model resident and nothing else.

    Run as its own process by :func:`run_probe`, which is what makes the memory
    numbers and the cold-start numbers mean what they say.
    """
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, config = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    mx.eval(model.parameters())
    mx.synchronize()
    load_seconds = time.perf_counter() - started
    resident = {
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
    }

    prompt = corpus_tokens(tokenizer, at_least=workload.prompt_tokens)
    cold_prefill, cold_decode, _ = timed_generation(
        model, prompt, generate_tokens=workload.generate_tokens
    )
    warm = [
        timed_generation(model, prompt, generate_tokens=workload.generate_tokens)[:2]
        for _ in range(PROBE_WARM_RUNS)
    ]
    warm_prefill = float(np.median([entry[0] for entry in warm]))
    warm_decode = float(np.median([entry[1] for entry in warm]))
    return {
        "path": str(model_path),
        "load_seconds": load_seconds,
        "quantization_declared": bool(
            "quantization" in config
            or "quantization_config" in config
            or "quantization_config" in config.get("text_config", {})
        ),
        "resident": resident,
        "after_generation": {
            "active_memory_bytes": int(mx.get_active_memory()),
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "cache_memory_bytes": int(mx.get_cache_memory()),
        },
        "census": parameter_census(model),
        "cold": rates(workload, cold_prefill, cold_decode),
        "warm": rates(workload, warm_prefill, warm_decode) | {"runs": PROBE_WARM_RUNS},
        # What the first run pays and later runs do not: Metal library JIT for
        # the packed kernels, the first buffer allocations, the first touch of
        # every weight page. One cold sample against a warm median, so it is a
        # bound on the cold-start cost rather than a precise one; A5 measured
        # JIT compilation on its own.
        "cold_minus_warm_seconds": (cold_prefill + cold_decode)
        - (warm_prefill + warm_decode),
        "device_info": {key: value for key, value in mx.device_info().items()},
        "tokens_prompted": len(prompt),
    }


def sample_resident_bytes(pid: int, stop: threading.Event) -> list[int]:
    """Poll a child's resident size until it exits.

    ``ps`` reports kibibytes; the caller wants bytes. A failed sample is
    dropped rather than raised -- the process exiting mid-poll is the normal
    way this loop ends.
    """
    samples: list[int] = []
    while not stop.wait(RSS_POLL_SECONDS):
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True, capture_output=True, check=False
        )
        line = completed.stdout.strip()
        if completed.returncode == 0 and line.isdigit():
            samples.append(int(line) * 1024)
    return samples


def run_probe(model_path: Path, *, workload: Workload) -> dict[str, object]:
    """Run :func:`probe` in a fresh process, sampling its footprint from outside."""
    command = [
        sys.executable,
        "-m",
        "ternel_mlx.bench_model",
        "--probe",
        str(model_path),
        "--probe-workload",
        workload.label,
    ]
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stop = threading.Event()
    samples: list[int] = []

    def poll() -> None:
        samples.extend(sample_resident_bytes(process.pid, stop))

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    stdout, stderr = process.communicate()
    stop.set()
    watcher.join()
    if process.returncode != 0:
        raise FormatError(f"memory probe of {model_path} failed:\n{stderr}")
    document = json.loads(stdout)
    document["process"] = {
        "peak_resident_bytes": max(samples) if samples else None,
        "resident_samples": len(samples),
        "poll_seconds": RSS_POLL_SECONDS,
        "note": (
            "on unified memory there is no separate VRAM figure to read, so the "
            "resident size of the process is the closest honest proxy for what the "
            "model costs the machine; the IO registry accounts GPU time per "
            "process but not GPU memory per process"
        ),
    }
    return document


def workload_by_label(label: str) -> Workload:
    for workload in WORKLOADS:
        if workload.label == label:
            return workload
    raise FormatError(f"unknown workload {label!r}")


def prompt_digest(tokens: tuple[int, ...]) -> str:
    """Identify a token sequence across process boundaries.

    Each arm builds its own corpus from its own tokenizer, so the parent has to
    check they built the same one rather than assume it.
    """
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


def solo(model_path: Path, *, trials: int) -> dict[str, object]:
    """Every workload, timed ``trials`` times, with only this model resident.

    Run as its own process by :func:`run_solo`. Returns raw per-trial seconds
    rather than summaries: the parent pairs the arms round by round, and a
    median taken here would throw away the pairing.
    """
    model, _ = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    mx.eval(model.parameters())
    mx.synchronize()

    longest = max(workload.prompt_tokens for workload in WORKLOADS)
    corpus = corpus_tokens(tokenizer, at_least=longest)

    for _ in range(SOLO_WARMUP_ROUNDS):
        for workload in WORKLOADS:
            timed_generation(
                model, corpus[: workload.prompt_tokens], generate_tokens=workload.generate_tokens
            )

    measured: dict[str, object] = {}
    for workload in WORKLOADS:
        prompt = corpus[: workload.prompt_tokens]
        prefill_seconds: list[float] = []
        decode_seconds: list[float] = []
        produced: set[tuple[int, ...]] = set()
        for _ in range(trials):
            prefill, decode, tokens = timed_generation(
                model, prompt, generate_tokens=workload.generate_tokens
            )
            prefill_seconds.append(prefill)
            decode_seconds.append(decode)
            produced.add(tokens)
        if len(produced) != 1:
            raise FormatError(
                f"{workload.label}: {model_path} produced {len(produced)} distinct greedy "
                "continuations across identical trials"
            )
        measured[workload.label] = {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "prompt_digest": prompt_digest(prompt),
            "tokens": list(next(iter(produced))),
        }
    return {"path": str(model_path), "trials": trials, "workloads": measured}


def run_solo(model_path: Path, *, trials: int) -> dict[str, object]:
    """Run :func:`solo` in a fresh process, so nothing else is resident."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ternel_mlx.bench_model",
            "--solo",
            str(model_path),
            "--solo-trials",
            str(trials),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise FormatError(f"isolated run of {model_path} failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


def measure_isolated(
    artifact_dir: Path,
    baseline_dir: Path,
    *,
    rounds: int,
    trials_per_round: int,
    max_foreign_gpu_share: float,
) -> dict[str, object]:
    """The reported throughput: one model per process, arms alternating by round.

    Rounds alternate which arm runs first so a thermal ramp cannot land on one of
    them, and every round is a fresh process for both arms, so neither ever sees
    the other's weights in memory.
    """
    if rounds < 2:
        raise FormatError(f"{rounds} rounds cannot alternate which arm goes first")
    samples = rounds * trials_per_round
    if samples < TRIALS:
        raise FormatError(
            f"{rounds} rounds of {trials_per_round} is {samples} samples; "
            f"paired_bootstrap_ratio refuses fewer than {TRIALS}"
        )

    before = capture_environment()
    contention: list[dict[str, object]] = []
    collected: dict[str, list[dict[str, object]]] = {"candidate": [], "reference": []}
    for index in range(rounds):
        order = ("candidate", "reference") if index % 2 == 0 else ("reference", "candidate")
        for arm in order:
            path = artifact_dir if arm == "candidate" else baseline_dir
            print(f"  round {index + 1}/{rounds}, {arm}", file=sys.stderr, flush=True)
            label = f"round {index + 1} {arm}"
            with exclusive_gpu(
                label, max_foreign_share=max_foreign_gpu_share, record=contention
            ):
                collected[arm].append(run_solo(path, trials=trials_per_round))

    measurements = []
    for workload in WORKLOADS:
        series: dict[str, dict[str, list[float]]] = {}
        answers: dict[str, set[tuple[int, ...]]] = {}
        digests: set[str] = set()
        for arm, runs in collected.items():
            prefill: list[float] = []
            decode: list[float] = []
            answers[arm] = set()
            for run in runs:
                entry = run["workloads"][workload.label]
                prefill.extend(entry["prefill_seconds"])
                decode.extend(entry["decode_seconds"])
                answers[arm].add(tuple(entry["tokens"]))
                digests.add(entry["prompt_digest"])
            series[arm] = {"prefill": prefill, "decode": decode}
        if len(digests) != 1:
            raise FormatError(
                f"{workload.label}: the arms were not timed on the same prompt "
                f"({len(digests)} distinct token sequences)"
            )
        for arm, produced in answers.items():
            if len(produced) != 1:
                raise FormatError(
                    f"{workload.label}: the {arm} arm produced {len(produced)} distinct greedy "
                    "continuations across rounds"
                )
        candidate_tokens = next(iter(answers["candidate"]))
        reference_tokens = next(iter(answers["reference"]))

        measured: dict[str, object] = {
            "label": workload.label,
            "prompt_tokens": workload.prompt_tokens,
            "generate_tokens": workload.generate_tokens,
            "samples": samples,
            "greedy_continuations_identical": candidate_tokens == reference_tokens,
            "greedy_common_prefix": common_prefix_length(candidate_tokens, reference_tokens),
        }
        for segment in ("prefill", "decode"):
            base = series["reference"][segment]
            cand = series["candidate"][segment]
            measured[segment] = {
                "packed_seconds": quantile_summary(np.asarray(cand, dtype=np.float64)),
                "affine_2bit_seconds": quantile_summary(np.asarray(base, dtype=np.float64)),
                "speedup_vs_affine_2bit": float(np.median(base) / np.median(cand)),
                "bootstrap_candidate_over_reference": paired_bootstrap_ratio(
                    base, cand, samples=100_000, seed=SEED
                ),
            }
        measured["packed_rates"] = rates(
            workload,
            float(np.median(series["candidate"]["prefill"])),
            float(np.median(series["candidate"]["decode"])),
        )
        measured["affine_2bit_rates"] = rates(
            workload,
            float(np.median(series["reference"]["prefill"])),
            float(np.median(series["reference"]["decode"])),
        )
        measurements.append(measured)

    return {
        "schema_version": SCHEMA_VERSION,
        "layout_name": LAYOUT_NAME,
        "layout_version": LAYOUT_VERSION,
        "seed": SEED,
        "measured": "one model per process, arms alternating by round",
        "rounds": rounds,
        "trials_per_round": trials_per_round,
        "samples_per_arm": samples,
        "solo_warmup_rounds": SOLO_WARMUP_ROUNDS,
        "max_foreign_gpu_share": max_foreign_gpu_share,
        "gpu_contention": contention,
        "baseline": {
            "name": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
            "path": str(baseline_dir),
            "mode": "affine",
            "bits": 2,
            "group_size": 128,
            "bits_per_weight": 2.25,
            "runtime": "stock mlx-lm",
        },
        "workloads": measurements,
        "environment_before": before,
        "environment_after": capture_environment(),
    }


def benchmark(
    artifact_dir: Path,
    baseline_dir: Path,
    *,
    workloads: tuple[Workload, ...],
    trials: int,
    max_foreign_gpu_share: float,
) -> dict[str, object]:
    before = capture_environment()
    contention: list[dict[str, object]] = []

    # Memory first, one model at a time, before anything else is resident. The
    # probes are guarded too: they are where cold load and the warm reference
    # rates come from, and both are timings.
    print("  probing the packed artifact alone", file=sys.stderr, flush=True)
    with exclusive_gpu("probe candidate", max_foreign_share=max_foreign_gpu_share, record=contention):
        candidate_probe = run_probe(artifact_dir, workload=workloads[0])
    print("  probing the official 2-bit checkpoint alone", file=sys.stderr, flush=True)
    with exclusive_gpu("probe reference", max_foreign_share=max_foreign_gpu_share, record=contention):
        reference_probe = run_probe(baseline_dir, workload=workloads[0])

    # The probe reported what the config declared and what the graph became,
    # so the B7 representation gate re-runs here against the process that was
    # actually measured rather than against a fresh, unmeasured load.
    problems = check_representation(candidate_probe, artifact_dir)
    if reference_probe["census"]["packed_modules"]["PackedLinear"] != 0:
        problems.append("the official checkpoint loaded Ternel modules, so it is not a baseline")

    print("  loading both models for the paired throughput run", file=sys.stderr, flush=True)
    candidate, _ = load_model(artifact_dir)
    candidate_tokenizer = load_tokenizer(artifact_dir)
    reference, _ = load_model(baseline_dir)
    reference_tokenizer = load_tokenizer(baseline_dir)
    mx.eval(candidate.parameters(), reference.parameters())

    longest = max(workload.prompt_tokens for workload in workloads)
    prompt = corpus_tokens(candidate_tokenizer, at_least=longest)
    if prompt != corpus_tokens(reference_tokenizer, at_least=longest):
        raise FormatError(
            "the two tokenizers disagree, so the arms would not be timed on the same work"
        )

    measurements = []
    for workload in workloads:
        print(f"  timing {workload.label}", file=sys.stderr, flush=True)
        with exclusive_gpu(
            workload.label, max_foreign_share=max_foreign_gpu_share, record=contention
        ):
            measurements.append(
                measure_workload(
                    workload,
                    candidate=candidate,
                    reference=reference,
                    prompt=prompt[: workload.prompt_tokens],
                    trials=trials,
                )
            )

    resident = {
        "packed_bytes": candidate_probe["resident"]["active_memory_bytes"],
        "affine_2bit_bytes": reference_probe["resident"]["active_memory_bytes"],
    }
    parameter_bytes = {
        "packed": candidate_probe["census"]["total_bytes"],
        "affine_2bit": reference_probe["census"]["total_bytes"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "layout_name": LAYOUT_NAME,
        "layout_version": LAYOUT_VERSION,
        "seed": SEED,
        "trials_per_arm": trials,
        "warmup_rounds": WARMUP_ROUNDS,
        "max_foreign_gpu_share": max_foreign_gpu_share,
        "gpu_contention": contention,
        "baseline": {
            "name": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
            "path": str(baseline_dir),
            "mode": "affine",
            "bits": 2,
            "group_size": 128,
            "bits_per_weight": 2.25,
            "runtime": "stock mlx-lm",
        },
        "memory": {
            "measured": "one model per process, nothing else resident",
            "candidate": candidate_probe,
            "reference": reference_probe,
            "active_memory_saving": 1.0 - resident["packed_bytes"] / resident["affine_2bit_bytes"],
            "parameter_bytes": parameter_bytes,
            "parameter_saving": 1.0
            - parameter_bytes["packed"] / parameter_bytes["affine_2bit"],
        },
        "workloads": measurements,
        "representation_problems": problems,
        "environment_before": before,
        "environment_after": capture_environment(),
    }


def print_workloads(payload: dict[str, object]) -> None:
    """The per-workload lines, identical in shape for both throughput modes."""
    for entry in payload["workloads"]:
        prefill = entry["prefill"]["bootstrap_candidate_over_reference"]["ci95"]
        decode = entry["decode"]["bootstrap_candidate_over_reference"]["ci95"]
        print(
            f"{entry['label']:<14} prompt {entry['packed_rates']['prompt_tokens_per_second']:8.2f} "
            f"vs {entry['affine_2bit_rates']['prompt_tokens_per_second']:8.2f} tok/s "
            f"(x{entry['prefill']['speedup_vs_affine_2bit']:.3f}, ci "
            f"{prefill[0]:.3f}-{prefill[1]:.3f}) | generate "
            f"{entry['packed_rates']['generated_tokens_per_second']:6.2f} vs "
            f"{entry['affine_2bit_rates']['generated_tokens_per_second']:6.2f} tok/s "
            f"(x{entry['decode']['speedup_vs_affine_2bit']:.3f}, ci "
            f"{decode[0]:.3f}-{decode[1]:.3f}) | same tokens "
            f"{entry['greedy_continuations_identical']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the full packed model (B8)")
    parser.add_argument("--probe", type=Path, help="internal: measure one model in this process")
    parser.add_argument("--probe-workload", type=str, help="internal: the probe's workload label")
    parser.add_argument("--solo", type=Path, help="internal: time one model in this process")
    parser.add_argument("--solo-trials", type=int, help="internal: the solo run's trial count")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, help="paired run: both models in one process")
    parser.add_argument("--trials", type=int, help="paired trials, requires --output")
    parser.add_argument(
        "--isolated-output", type=Path, help="isolated run: one model per process, by round"
    )
    parser.add_argument("--rounds", type=int, help="isolated rounds, requires --isolated-output")
    parser.add_argument(
        "--trials-per-round", type=int, help="isolated trials per round, requires --isolated-output"
    )
    parser.add_argument(
        "--max-foreign-gpu-share",
        type=float,
        help="void the run if another process exceeds this share of the GPU while timing",
    )
    args = parser.parse_args(argv)

    if args.probe is not None:
        if args.probe_workload is None:
            parser.error("--probe requires --probe-workload")
        json.dump(probe(args.probe, workload=workload_by_label(args.probe_workload)), sys.stdout)
        return 0

    if args.solo is not None:
        if args.solo_trials is None:
            parser.error("--solo requires --solo-trials")
        json.dump(solo(args.solo, trials=args.solo_trials), sys.stdout)
        return 0

    for name in ("artifact", "baseline"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required")
    if args.output is None and args.isolated_output is None:
        parser.error("one of --output or --isolated-output is required")
    if args.max_foreign_gpu_share is None:
        parser.error("--max-foreign-gpu-share is required; a timing run has to know what it allows")

    problems: list[str] = []
    if args.output is not None:
        if args.trials is None:
            parser.error("--output requires --trials")
        payload = benchmark(
            args.artifact,
            args.baseline,
            workloads=WORKLOADS,
            trials=args.trials,
            max_foreign_gpu_share=args.max_foreign_gpu_share,
        )
        write_json_atomic(args.output, payload)

        memory = payload["memory"]
        print(
            f"memory:  packed {memory['candidate']['resident']['active_memory_bytes']} vs affine "
            f"{memory['reference']['resident']['active_memory_bytes']} bytes active, "
            f"{memory['active_memory_saving'] * 100:.2f}% less"
        )
        print(
            f"         peak RSS {memory['candidate']['process']['peak_resident_bytes']} vs "
            f"{memory['reference']['process']['peak_resident_bytes']}, cold load "
            f"{memory['candidate']['load_seconds']:.2f}s vs "
            f"{memory['reference']['load_seconds']:.2f}s, cold-warm "
            f"{memory['candidate']['cold_minus_warm_seconds']:.2f}s vs "
            f"{memory['reference']['cold_minus_warm_seconds']:.2f}s"
        )
        print("paired, both models resident -- kept as evidence, not as the result:")
        print_workloads(payload)
        problems.extend(payload["representation_problems"])

    if args.isolated_output is not None:
        for name in ("rounds", "trials_per_round"):
            if getattr(args, name) is None:
                parser.error(f"--isolated-output requires --{name.replace('_', '-')}")
        isolated = measure_isolated(
            args.artifact,
            args.baseline,
            rounds=args.rounds,
            trials_per_round=args.trials_per_round,
            max_foreign_gpu_share=args.max_foreign_gpu_share,
        )
        write_json_atomic(args.isolated_output, isolated)
        print("isolated, one model per process -- the reported result:")
        print_workloads(isolated)

    if problems:
        print(f"PROBLEMS: {problems}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
