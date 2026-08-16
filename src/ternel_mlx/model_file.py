"""Emit the self-contained ``model_file`` the artifact ships.

``mlx_lm.utils.load_model`` imports ``Model`` and ``ModelArgs`` from the file
``config["model_file"]`` names. It loads that file by path under the module
name ``custom_model``, with no package and without the model directory on
``sys.path``, so the file can only import from installed packages. A file that
imports ``ternel_mlx`` therefore turns "download the checkpoint" into "install
this package first"; a file with no such import makes ``pip install mlx-lm``
the whole requirement.

The file is generated, never written by hand: ``layout``, ``kernels``,
``modules`` and ``packed_model`` are concatenated in dependency order, their
package-internal imports stripped, and every stripped name inlined with the
value the package holds at emission time. The accounting fails closed -- an
import the emitter cannot classify, a renamed internal name, a cross-section
name the earlier section does not define, a redefinition between sections: each
is an error, never a gap in the output. The emitted text is then compiled and
re-parsed to prove no internal or relative import survived.

This is not a second definition of the format, which was the argument against
vendoring: the package stays the only definition, this module derives the file
from it, and ``tests/test_mlx_model_file.py`` regenerates the file against the
shipped artifact so the two cannot drift apart silently.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

from bonsai_tq1.format import FormatError, write_json_atomic

from . import LAYOUT_NAME, LAYOUT_VERSION

MODEL_FILE_NAME = "ternel_packed_model.py"

# The sections of the emitted file, in dependency order: each may import only
# from the ones before it. This module itself is deliberately absent.
SECTION_MODULES = ("layout", "kernels", "modules", "packed_model")

# Package roots whose imports must not survive into the emitted file.
INTERNAL_ROOTS = ("ternel_mlx", "bonsai_tq1")

# The environment the artifact's gates ran against. Emission from any other
# environment is refused rather than silently producing a file whose floor
# nobody verified; re-run the gates first, then move these pins.
VERIFIED_VERSIONS = (("mlx", "0.32.0"), ("mlx-lm", "0.31.3"))


@dataclass(frozen=True)
class _Inlined:
    """One name stripped from an internal import, with the text that recreates it."""

    name: str
    module: str
    source: str


@dataclass(frozen=True)
class _Section:
    name: str
    sha256: str
    body: str
    external_imports: tuple[str, ...]
    inlined: tuple[_Inlined, ...]
    cross_needed: tuple[tuple[str, str], ...]
    defined: frozenset[str]
    exports: tuple[str, ...] | None


def _section_source(name: str) -> str:
    path = Path(__file__).resolve().with_name(f"{name}.py")
    if not path.is_file():
        raise FormatError(f"section source {path} is missing")
    return path.read_text(encoding="utf-8")


def _line_span(node: ast.stmt) -> range:
    if node.end_lineno is None:
        raise FormatError(f"statement at line {node.lineno} has no end line")
    return range(node.lineno - 1, node.end_lineno)


def _statement_text(lines: list[str], node: ast.stmt) -> str:
    return "\n".join(lines[index] for index in _line_span(node))


def _plain_names(section: str, node: ast.ImportFrom) -> list[str]:
    names: list[str] = []
    for alias in node.names:
        if alias.name == "*":
            raise FormatError(f"{section}.py uses a star import; the emitter inlines names")
        if alias.asname is not None:
            raise FormatError(
                f"{section}.py renames {alias.name} on an internal import; "
                "the emitter inlines names verbatim"
            )
        names.append(alias.name)
    return names


def _inline_source(module_name: str, attr: str) -> str:
    module = importlib.import_module(module_name)
    if not hasattr(module, attr):
        raise FormatError(f"{module_name} does not define {attr}")
    value = getattr(module, attr)
    if isinstance(value, type):
        return inspect.getsource(value).rstrip("\n")
    if isinstance(value, (int, str)):
        return f"{attr} = {value!r}"
    raise FormatError(
        f"{module_name}.{attr} is {type(value).__name__}; "
        "the emitter inlines only ints, strings and classes"
    )


def _exported_names(section: str, node: ast.Assign) -> tuple[str, ...]:
    exported = ast.literal_eval(node.value)
    if not isinstance(exported, list) or not all(isinstance(item, str) for item in exported):
        raise FormatError(f"{section}.py binds __all__ to something other than a list of names")
    return tuple(exported)


def _is_dunder_all(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__all__"
    )


def _defined_names(nodes: list[ast.stmt]) -> frozenset[str]:
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.update(
                        element.id for element in target.elts if isinstance(element, ast.Name)
                    )
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return frozenset(names)


def _strip_section(name: str, source: str) -> _Section:
    tree = ast.parse(source, filename=f"{name}.py")
    lines = source.splitlines()
    dropped: set[int] = set()
    external: list[str] = []
    inlined: list[_Inlined] = []
    cross: list[tuple[str, str]] = []
    exports: tuple[str, ...] | None = None

    for node in tree.body:
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
            internal = roots & set(INTERNAL_ROOTS)
            if internal:
                raise FormatError(f"{name}.py imports {sorted(internal)} as plain modules")
            external.append(_statement_text(lines, node))
            dropped.update(_line_span(node))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and module == "__future__":
                if _plain_names(name, node) != ["annotations"]:
                    raise FormatError(f"{name}.py imports more than annotations from __future__")
            elif node.level == 1 and module == "":
                inlined.extend(
                    _Inlined(item, "ternel_mlx", _inline_source("ternel_mlx", item))
                    for item in _plain_names(name, node)
                )
            elif node.level == 1 and module in SECTION_MODULES:
                cross.extend((module, item) for item in _plain_names(name, node))
            elif node.level != 0:
                raise FormatError(f"{name}.py has an unaccounted relative import of {module!r}")
            elif module.split(".")[0] in INTERNAL_ROOTS:
                inlined.extend(
                    _Inlined(item, module, _inline_source(module, item))
                    for item in _plain_names(name, node)
                )
            else:
                external.append(_statement_text(lines, node))
            dropped.update(_line_span(node))
        elif _is_dunder_all(node):
            exports = _exported_names(name, node)
            dropped.update(_line_span(node))

    body = "\n".join(line for index, line in enumerate(lines) if index not in dropped)
    return _Section(
        name=name,
        sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        body=body.strip("\n"),
        external_imports=tuple(external),
        inlined=tuple(inlined),
        cross_needed=tuple(cross),
        defined=_defined_names(tree.body),
        exports=exports,
    )


def _bound_names(text: str) -> list[str]:
    node = ast.parse(text).body[0]
    if isinstance(node, ast.Import):
        return [alias.asname or alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [alias.asname or alias.name for alias in node.names]
    raise FormatError(f"{text!r} is not an import statement")


def _merge_external_imports(sections: list[_Section]) -> tuple[list[str], set[str]]:
    """Dedupe the surviving imports and refuse two spellings of one bound name."""
    by_bound: dict[str, str] = {}
    texts: list[str] = []
    for section in sections:
        for text in section.external_imports:
            for bound in _bound_names(text):
                if bound in by_bound and by_bound[bound] != text:
                    raise FormatError(
                        f"{section.name}.py binds {bound} via {text!r} but an earlier "
                        f"section binds it via {by_bound[bound]!r}"
                    )
                by_bound[bound] = text
            if text not in texts:
                texts.append(text)
    plain = sorted(text for text in texts if text.startswith("import "))
    froms = sorted(text for text in texts if not text.startswith("import "))
    return plain + froms, set(by_bound)


def _merge_inlined(sections: list[_Section]) -> list[_Inlined]:
    seen: dict[str, _Inlined] = {}
    merged: list[_Inlined] = []
    for section in sections:
        for item in section.inlined:
            if item.name in seen:
                if seen[item.name] != item:
                    raise FormatError(f"{item.name} is inlined twice with different definitions")
                continue
            seen[item.name] = item
            merged.append(item)
    return merged


def _prelude(inlined: list[_Inlined]) -> str:
    order: list[str] = []
    groups: dict[str, list[_Inlined]] = {}
    for item in inlined:
        if item.module not in groups:
            groups[item.module] = []
            order.append(item.module)
        groups[item.module].append(item)
    chunks = ["# Inlined by ternel_mlx.model_file, values read from the package at emission."]
    for module in order:
        chunks.append(f"\n# from {module}:")
        for item in groups[module]:
            chunks.append(f"{item.source}\n" if "\n" in item.source else item.source)
    return "\n".join(chunks)


def _account_names(sections: list[_Section], available: set[str]) -> None:
    """Prove every stripped internal name resolves, and nothing is shadowed."""
    defined_by: dict[str, frozenset[str]] = {}
    for section in sections:
        for provider, needed in section.cross_needed:
            if provider not in defined_by:
                raise FormatError(
                    f"{section.name}.py imports from {provider}, which is emitted later"
                )
            if needed not in defined_by[provider]:
                raise FormatError(
                    f"{section.name}.py imports {needed} from {provider}, "
                    "which does not define it"
                )
        collisions = section.defined & available
        if collisions:
            raise FormatError(f"{section.name}.py redefines {sorted(collisions)}")
        available.update(section.defined)
        defined_by[section.name] = section.defined


def _exports(sections: list[_Section]) -> str:
    owners = [section for section in sections if section.exports is not None]
    if len(owners) != 1:
        raise FormatError(f"{len(owners)} sections define __all__; exactly one must")
    exports = owners[0].exports
    if exports is None:
        raise FormatError(f"{owners[0].name}.py binds __all__ to nothing")
    return "__all__ = [" + ", ".join(f"{name!r}" for name in exports) + "]"


def _header(sections: list[_Section]) -> str:
    hashes = "\n".join(
        f"    {section.name + '.py':<16} {section.sha256[:12]}" for section in sections
    )
    versions = ", ".join(f"{package}=={pinned}" for package, pinned in VERIFIED_VERSIONS)
    return f'''"""The classes ``mlx_lm.utils.load_model`` imports for this checkpoint.

``config.json`` names this file in ``model_file``. mlx-lm loads it by path
under the module name ``custom_model``, with no package and without the model
directory on ``sys.path``, so it can only import from installed packages --
and the only packages it imports are mlx and mlx-lm:

    pip install mlx-lm
    mlx_lm.generate --model <this directory> --prompt "..."

The weights are {LAYOUT_NAME} v{LAYOUT_VERSION} blocks in a tiled layout only
the Metal kernels below can read; there is no dequantised copy to fall back to.

Generated by ``ternel_mlx.model_file`` -- do not edit. The canonical
definition of everything here is the Ternel package, whose test suite
regenerates this file and fails if the two differ. Emitted from sources:

{hashes}

against {versions}, the environment the artifact's gates ran on.
"""'''


def _check_emitted(text: str) -> None:
    compile(text, MODEL_FILE_NAME, "exec")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise FormatError("the emitted file contains a relative import")
            if node.module == "__future__":
                raise FormatError("the emitted file must not rely on PEP 563 annotations")
            if (node.module or "").split(".")[0] in INTERNAL_ROOTS:
                raise FormatError(f"the emitted file still imports {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in INTERNAL_ROOTS:
                    raise FormatError(f"the emitted file still imports {alias.name}")
    defined = _defined_names(tree.body)
    for required in ("Model", "ModelArgs"):
        if required not in defined:
            raise FormatError(f"the emitted file does not define {required}")


def _require_verified_environment() -> None:
    for package, pinned in VERIFIED_VERSIONS:
        installed = importlib.metadata.version(package)
        if installed != pinned:
            raise FormatError(
                f"{package} {installed} is installed but the artifact's gates ran against "
                f"{pinned}; re-run the gates before emitting a model_file from this environment"
            )


def emit_model_file_source() -> str:
    _require_verified_environment()
    sections = [_strip_section(name, _section_source(name)) for name in SECTION_MODULES]
    imports, bound = _merge_external_imports(sections)
    inlined = _merge_inlined(sections)
    _account_names(sections, {item.name for item in inlined} | bound)

    # The sections' ``from __future__ import annotations`` is dropped, not
    # carried over: mlx-lm executes this file without registering it in
    # ``sys.modules``, and ``@dataclass`` under PEP 563 resolves its string
    # annotations through ``sys.modules[cls.__module__]`` -- an AttributeError
    # at load time. Annotations therefore evaluate eagerly here, which is why
    # the sections are emitted in dependency order.
    blocks = [
        _header(sections),
        "\n".join(imports),
        _prelude(inlined),
    ]
    for section in sections:
        stamp = f"ternel_mlx.{section.name} (source sha256 {section.sha256[:12]})"
        blocks.append(f"# {'=' * 8} {stamp} {'=' * 8}\n\n{section.body}")
    blocks.append(_exports(sections))

    text = "\n\n\n".join(blocks) + "\n"
    _check_emitted(text)
    return text


def write_model_file(out_dir: Path) -> dict[str, str | dict[str, str]]:
    """Write the ``model_file`` mlx-lm imports, and record what produced it."""
    text = emit_model_file_source()
    (out_dir / MODEL_FILE_NAME).write_text(text, encoding="utf-8")
    return {
        "name": MODEL_FILE_NAME,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "requires": f"mlx-lm>={dict(VERIFIED_VERSIONS)['mlx-lm']}",
        "generated_from": {
            f"{name}.py": hashlib.sha256(_section_source(name).encode("utf-8")).hexdigest()
            for name in SECTION_MODULES
        },
    }


def regenerate(artifact_dir: Path) -> dict[str, str | dict[str, str]]:
    """Re-emit an existing artifact's model_file and update its manifest record."""
    # Imported here, not at module level: convert imports this module.
    from ternel_mlx.convert import CONFIG_NAME, MANIFEST_NAME

    config_path = artifact_dir / CONFIG_NAME
    manifest_path = artifact_dir / MANIFEST_NAME
    if not config_path.is_file() or not manifest_path.is_file():
        raise FormatError(
            f"{artifact_dir} is not a Ternel artifact; it lacks a config or manifest"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_file") != MODEL_FILE_NAME:
        raise FormatError(
            f"the artifact's config names model_file {config.get('model_file')!r}, "
            f"not {MODEL_FILE_NAME!r}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("model_file"), dict):
        raise FormatError("the manifest carries no files.model_file record to update")
    record = write_model_file(artifact_dir)
    files["model_file"] = record
    write_json_atomic(manifest_path, manifest)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    record = regenerate(args.artifact)
    print(f"{record['name']}: sha256 {record['sha256']}")
    print(f"requires {record['requires']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
