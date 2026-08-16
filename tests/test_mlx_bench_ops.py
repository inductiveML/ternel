"""Gates on the op sweep's contract with itself.

The sweep needs a GPU and an hour, so what is gated here is the part that
decides *what* gets measured. One argument does that almost entirely: the
activation dtype is a kernel template parameter, so it selects which compiled
kernel the packed arm runs, and through the scales it builds it selects which
kernel MLX runs on the affine arm too. A sweep that picked it silently would
answer a question nobody asked -- which is what happened, since the committed
``results/mlx/a5_op_benchmarks*.json`` were measured in float32 while the model
declares bfloat16.
"""

from __future__ import annotations

import pytest

from bonsai_tq1.format import FormatError
from ternel_mlx.bench_ops import ACTIVATION_DTYPES, main, run
from ternel_mlx.kernel_gate import NARROWED_RELATIVE_TOLERANCE

COMPLETE = [
    "--output", "out.json",
    "--shape", "all",
    "--batches", "1",
    "--max-foreign-gpu-share", "0.25",
    "--activation-dtype", "bfloat16",
]


def without(flag: str) -> list[str]:
    """``COMPLETE`` with one flag and its values removed."""
    argv = list(COMPLETE)
    start = argv.index(flag)
    end = start + 1
    while end < len(argv) and not argv[end].startswith("--"):
        end += 1
    return argv[:start] + argv[end:]


@pytest.mark.parametrize(
    "flag",
    ["--output", "--shape", "--batches", "--max-foreign-gpu-share", "--activation-dtype"],
)
def test_an_incompletely_specified_sweep_is_refused_rather_than_defaulted(flag):
    """Every knob is explicit, so a missing one is an error and never a default."""
    with pytest.raises(SystemExit) as exit_info:
        main(without(flag))
    assert exit_info.value.code == 2


def test_a_dtype_the_kernels_do_not_template_on_is_refused():
    """Caught at the flag, before a GPU is touched."""
    with pytest.raises(SystemExit) as exit_info:
        main([*without("--activation-dtype"), "--activation-dtype", "float64"])
    assert exit_info.value.code == 2


def test_the_dtype_is_checked_again_by_the_function_the_flag_calls():
    """``run`` is importable, so the flag's choices are not the only guard.

    A caller that reaches past ``main`` -- a notebook, a future harness -- would
    otherwise get an arbitrary kernel rather than a refusal.
    """
    with pytest.raises(FormatError, match="unknown activation dtype"):
        run(
            shapes=(("probe", 256, 5120),),
            batches=(1,),
            max_foreign_gpu_share=0.25,
            activation_dtype_name="float64",
        )


def test_every_sweepable_dtype_has_a_correctness_bound():
    """The sweep gates each arm before timing it, by dtype name.

    Without this the two tables drift and adding a dtype to the sweep raises a
    ``KeyError`` mid-run, an hour in, rather than at the flag.
    """
    assert set(ACTIVATION_DTYPES) <= set(NARROWED_RELATIVE_TOLERANCE)


def test_the_dtype_names_are_the_dtypes_they_name():
    """The table is keyed by name because MLX dtypes are not singletons.

    That makes the name the identity, so nothing else checks that the mapping is
    honest -- a table saying ``"bfloat16": mx.float32`` would sweep float32 and
    label the document bfloat16.
    """
    for name, dtype in ACTIVATION_DTYPES.items():
        assert str(dtype) == f"mlx.core.{name}"
