"""The model class the packed artifact points ``mlx_lm`` at.

``mlx_lm.utils.load_model`` honours a ``model_file`` key in ``config.json``: it
imports that file from the model directory and takes ``Model`` and ``ModelArgs``
from it. That is the entire injection point. No fork of MLX or MLX-LM is needed,
and none is made here.

The graph itself is mlx-lm's ``qwen3_5``, untouched. This module builds it and
then swaps every ``nn.Linear`` and ``nn.Embedding`` for a packed equivalent,
which matters for three reasons:

* The shape arithmetic of a hybrid Qwen3.5 -- which layers are recurrent, how
  the QKV fan-out is sized, where the SSM gates are only 48 rows wide -- is
  mlx-lm's business and is read off the modules it built rather than restated
  here, where it could drift.
* Walking the built graph cannot miss a layer kind. A hand-written list of
  tensor names can, and this architecture has two different layer kinds
  interleaved on a period of four.
* The replacement happens inside ``__init__``, before ``load_weights`` and long
  before ``mx.eval``. MLX arrays are lazy, so the ``nn.Linear`` weights that are
  discarded here are never evaluated and never occupy a byte.

What is emphatically *not* done: ``config.json`` carries no ``quantization``
key, so ``load_model`` never calls ``nn.quantize`` and no affine-quantised copy
of any weight is ever constructed.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path
from mlx_lm.models.qwen3_5 import Model as Qwen35Model
from mlx_lm.models.qwen3_5 import ModelArgs as Qwen35ModelArgs

from bonsai_tq1.format import FormatError

from .modules import PackedEmbedding, PackedLinear

__all__ = ["Model", "ModelArgs", "replace_with_packed"]

ModelArgs = Qwen35ModelArgs

# The key our converter writes into ``text_config`` to state the dtype the graph
# runs in. A packed embedding has no float weight to read a dtype from and is
# handed indices rather than activations, so nothing else in the graph can tell
# it what to emit.
ACTIVATION_DTYPE_KEY = "ternel_activation_dtype"

_DTYPES: dict[str, mx.Dtype] = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}


def resolve_activation_dtype(text_config: dict[str, object]) -> mx.Dtype:
    """Read the graph's dtype out of the artifact config, or refuse to guess."""
    name = text_config.get(ACTIVATION_DTYPE_KEY)
    if not isinstance(name, str):
        raise FormatError(
            f"artifact config is missing a string '{ACTIVATION_DTYPE_KEY}'; the packed "
            "embedding has no float weight to infer an output dtype from"
        )
    if name not in _DTYPES:
        raise FormatError(
            f"unsupported activation dtype {name!r}, expected one of {sorted(_DTYPES)}"
        )
    return _DTYPES[name]


def replace_with_packed(model: nn.Module, *, dtype: mx.Dtype) -> dict[str, int]:
    """Swap every linear and embedding leaf for its packed equivalent, in place.

    Returns a count per replaced kind so a caller can assert the graph it got
    is the graph it expected, rather than trusting that the walk found
    everything.
    """
    counts = {"linear": 0, "embedding": 0}

    def replace(path: str, module: nn.Module) -> nn.Module:
        if isinstance(module, nn.Linear):
            counts["linear"] += 1
            return PackedLinear.from_linear(module)
        if isinstance(module, nn.Embedding):
            counts["embedding"] += 1
            return PackedEmbedding.from_embedding(module, dtype=dtype)
        return module

    leaves = tree_map_with_path(replace, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return counts


class Model(Qwen35Model):
    """mlx-lm's Qwen3.5 graph with every matmul weight left packed.

    ``sanitize`` is inherited deliberately. Our artifact is written already in
    mlx-lm's convention, so the parent's key rewriting is a no-op on it, and the
    conv1d transpose and norm shift it performs are guarded on conditions our
    artifact does not meet. Inheriting keeps a single definition of what those
    conventions are.
    """

    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        dtype = resolve_activation_dtype(args.text_config)
        counts = replace_with_packed(self, dtype=dtype)
        if counts["embedding"] != 1:
            raise FormatError(
                f"expected exactly one embedding table, replaced {counts['embedding']}"
            )
        if counts["linear"] < 1:
            raise FormatError("no linear layers were replaced; the graph is not what was expected")
        self.packed_module_counts = counts
