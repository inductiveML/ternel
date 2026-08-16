"""Gates on the emitted self-contained ``model_file``.

``mlx_lm.utils.load_model`` executes ``config["model_file"]`` by path under the
module name ``custom_model``: one file, no package, the model directory never
on ``sys.path``, and never registered in ``sys.modules`` -- which is why the
emitted file may not rely on PEP 563 string annotations, and why these tests
import it exactly that way rather than as a module. One of them does so in a
fresh interpreter with the Ternel packages deliberately unimportable, because
"loads on a machine with nothing but mlx-lm installed" is the entire claim the
file makes.

The drift gate at the bottom regenerates the file against the shipped
artifact: editing ``layout``, ``kernels``, ``modules`` or ``packed_model``
without re-emitting the artifact's model_file is a test failure here, not a
silent fork of the format.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bonsai_tq1.format import FormatError
from ternel_mlx import LAYOUT_NAME, model_file
from ternel_mlx.convert import CONFIG_NAME, MANIFEST_NAME
from ternel_mlx.model_file import (
    MODEL_FILE_NAME,
    SECTION_MODULES,
    VERIFIED_VERSIONS,
    emit_model_file_source,
    write_model_file,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "ternel-mlx"

# Runs in a fresh interpreter: the blocker sits at the front of ``sys.meta_path``
# before anything Ternel could have been imported, so a single surviving import
# of either package fails the exec rather than resolving from this checkout.
BLOCKED_LOADER = """
import importlib.util
import sys

BLOCKED = {"ternel_mlx", "bonsai_tq1"}


class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(f"the standalone model_file imported {fullname}")
        return None


sys.meta_path.insert(0, Blocker())
spec = importlib.util.spec_from_file_location("custom_model", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from mlx_lm.models.qwen3_5 import Model as Qwen35Model
from mlx_lm.models.qwen3_5 import ModelArgs as Qwen35ModelArgs

assert issubclass(module.Model, Qwen35Model)
assert module.ModelArgs is Qwen35ModelArgs
assert not [name for name in sys.modules if name.split(".")[0] in BLOCKED]
print("standalone-load-ok")
"""


def test_the_model_file_imports_the_way_mlx_lm_imports_it(tmp_path: Path) -> None:
    """The failure this catches: an artifact whose ``model_file`` cannot import.

    ``load_model`` loads the file by path under the name ``custom_model``, with
    no package and without the model directory on ``sys.path``, so sibling
    modules copied beside it would not resolve. Executing it exactly that way is
    the only check that the emitted checkpoint is loadable at all.
    """
    pytest.importorskip("mlx.core")
    record = write_model_file(tmp_path)
    path = tmp_path / MODEL_FILE_NAME
    text = path.read_text(encoding="utf-8")
    assert record["name"] == MODEL_FILE_NAME
    assert record["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert record["requires"] == f"mlx-lm>={dict(VERIFIED_VERSIONS)['mlx-lm']}"
    generated_from = record["generated_from"]
    assert isinstance(generated_from, dict)
    assert set(generated_from) == {f"{name}.py" for name in SECTION_MODULES}
    assert all(len(digest) == 64 for digest in generated_from.values())

    spec = importlib.util.spec_from_file_location("custom_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from mlx_lm.models.qwen3_5 import Model as Qwen35Model
    from mlx_lm.models.qwen3_5 import ModelArgs as Qwen35ModelArgs

    assert issubclass(module.Model, Qwen35Model)
    assert module.ModelArgs is Qwen35ModelArgs
    assert callable(module.replace_with_packed)
    assert LAYOUT_NAME in (module.__doc__ or "")


def test_the_model_file_loads_where_ternel_is_not_installed(tmp_path: Path) -> None:
    """The claim on the box: a user with only mlx-lm installed can load this."""
    pytest.importorskip("mlx.core")
    write_model_file(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", BLOCKED_LOADER, str(tmp_path / MODEL_FILE_NAME)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "standalone-load-ok" in completed.stdout


def test_the_emitted_text_contains_no_path_back_to_the_packages() -> None:
    """An independent read of the text the emitter's own self-check approved."""
    tree = ast.parse(emit_model_file_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the emitted file contains a relative import"
            assert node.module != "__future__", "the emitted file relies on PEP 563"
            roots = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not roots & {"ternel_mlx", "bonsai_tq1"}, ast.dump(node)


def test_the_emitter_refuses_an_import_it_cannot_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A function import cannot be inlined as a constant, so it must refuse."""
    real = model_file._section_source

    def poisoned(name: str) -> str:
        source = real(name)
        if name == "packed_model":
            return source.replace(
                "from bonsai_tq1.format import FormatError",
                "from bonsai_tq1.format import FormatError\n"
                "from ternel_mlx.naming import build_name_map",
            )
        return source

    monkeypatch.setattr(model_file, "_section_source", poisoned)
    with pytest.raises(FormatError):
        emit_model_file_source()


def test_the_emitter_refuses_a_renamed_internal_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = model_file._section_source

    def poisoned(name: str) -> str:
        source = real(name)
        if name == "packed_model":
            return source.replace(
                "from bonsai_tq1.format import FormatError",
                "from bonsai_tq1.format import FormatError as TernelFormatError",
            )
        return source

    monkeypatch.setattr(model_file, "_section_source", poisoned)
    with pytest.raises(FormatError):
        emit_model_file_source()


def test_the_emitter_refuses_an_unverified_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The emitted file claims the gates' environment; another one must not emit."""
    monkeypatch.setattr(model_file, "VERIFIED_VERSIONS", (("mlx", "0.0.1"),))
    with pytest.raises(FormatError):
        emit_model_file_source()


@pytest.mark.skipif(
    not ARTIFACT_DIR.is_dir(), reason="the converted artifact is not on this machine"
)
def test_the_shipped_artifact_carries_exactly_what_the_package_emits() -> None:
    """Editing a section module without re-emitting the artifact fails here."""
    shipped = (ARTIFACT_DIR / MODEL_FILE_NAME).read_text(encoding="utf-8")
    assert shipped == emit_model_file_source()

    manifest = json.loads((ARTIFACT_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    record = manifest["files"]["model_file"]
    assert record["name"] == MODEL_FILE_NAME
    assert record["sha256"] == hashlib.sha256(shipped.encode("utf-8")).hexdigest()
    assert record["requires"] == f"mlx-lm>={dict(VERIFIED_VERSIONS)['mlx-lm']}"

    config = json.loads((ARTIFACT_DIR / CONFIG_NAME).read_text(encoding="utf-8"))
    assert config["model_file"] == MODEL_FILE_NAME
