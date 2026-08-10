#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from bonsai_tq1.constants import PRISM_COMMIT, PRISM_DIR, PRISM_REPO, ROOT


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and patch the pinned Prism baseline")
    parser.add_argument("--destination", type=Path, default=PRISM_DIR)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--no-checkout", PRISM_REPO, str(destination))
    run("git", "fetch", "--depth", "1", "origin", PRISM_COMMIT, cwd=destination)
    run("git", "checkout", "--detach", PRISM_COMMIT, cwd=destination)
    patches = (
        ROOT / "patches" / "prism-mmvq-benchmark.patch",
        ROOT / "patches" / "prism-tq1-g128-distribution.patch",
    )
    for patch in patches:
        if not patch.is_file():
            raise RuntimeError(f"required Prism patch is missing: {patch}")
        reverse = subprocess.run(
            ("git", "apply", "--reverse", "--check", str(patch)),
            cwd=destination,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if reverse.returncode != 0:
            run("git", "apply", "--check", str(patch), cwd=destination)
            run("git", "apply", str(patch), cwd=destination)
    actual = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=destination, text=True).strip()
    if actual != PRISM_COMMIT:
        raise RuntimeError(f"unexpected Prism commit {actual}")
    print(f"Prism baseline ready at {destination} ({actual})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
