"""Judge a candidate against its base instead of against zero.

Ruff, Pyright and the file-size check each report hundreds of pre-existing
violations, so none of them can be a pass/fail gate on its own. This wraps them:
it counts violations per (file, rule) on both the base tree and the candidate,
and fails only when a count rises. Existing debt does not block work; adding to
it does.

There is deliberately no baseline file. A baseline keyed by path goes stale the
moment a file is renamed, which is exactly when a large refactor needs the gate
to stay out of the way, and it becomes a second artifact with its own ownership
question. Git already knows what changed.

Both sides are measured with the CANDIDATE's configuration: the base tree gets
the candidate's pyproject.toml copied in before it is measured. Otherwise
enabling a new rule would make every one of its findings look new, and the
ratchet would block the very change that turns a check on.
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import _support

_DEFAULT_BASE_BRANCH: Final = "main"
_TOOLS_DIR: Final = Path(__file__).resolve().parent
# The base tree's own pyproject.toml, kept beside the candidate's copy so that
# configuration-level suppressions can still be compared.
_BASE_CONFIG: Final = "pyproject.base.toml"


class RatchetError(RuntimeError):
    """Raised when the base tree or a detector cannot produce a comparison."""


@dataclass(frozen=True)
class Regression:
    """One (path, rule) whose violation count rose between base and candidate."""

    path: str
    rule: str
    before: int
    after: int

    @property
    def increase(self) -> int:
        return self.after - self.before


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args), cwd=root, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RatchetError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def resolve_base(root: Path, base: str | None) -> str:
    """Return the commit the candidate is judged against."""
    if base is not None:
        return _git(root, "rev-parse", base).strip()
    return _git(root, "merge-base", "HEAD", _DEFAULT_BASE_BRANCH).strip()


def changed_paths(root: Path, base: str) -> frozenset[str]:
    """Return every path the candidate touches, working tree included.

    The comparison includes uncommitted and untracked work, not only committed
    history, so the ratchet gives the same verdict before and after a commit.
    """
    names = set(_git(root, "diff", "--name-only", base).splitlines())
    names.update(_git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    return frozenset(name for name in names if name)


def changed_python_files(root: Path, base: str) -> tuple[str, ...]:
    """Return candidate-touched Python files under the checked roots."""
    return tuple(
        sorted(
            name
            for name in changed_paths(root, base)
            if name.endswith(".py") and name.split("/")[0] in _support.CHECKED_ROOTS
        )
    )


def extract_base_tree(root: Path, base: str, destination: Path) -> None:
    """Materialise `base` as a plain directory, with the candidate's config."""
    archive = destination / "base.tar"
    with archive.open("wb") as stream:
        completed = subprocess.run(
            ("git", "archive", "--format=tar", base),
            cwd=root,
            check=False,
            stdout=stream,
            stderr=subprocess.PIPE,
            text=False,
        )
    if completed.returncode != 0:
        raise RatchetError(f"git archive {base} failed")
    tree = destination / "tree"
    tree.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(tree, filter="data")
    archive.unlink()
    extracted = tree / "pyproject.toml"
    if extracted.is_file():
        extracted.replace(tree / _BASE_CONFIG)
    shutil.copy2(root / "pyproject.toml", extracted)


def ruff_counts(tree: Path, files: Iterable[str]) -> Mapping[tuple[str, str], int]:
    """Return {(path, rule): count} from ruff over `files` that exist in `tree`."""
    present = [name for name in files if (tree / name).exists()]
    if not present:
        return {}
    completed = subprocess.run(
        ("ruff", "check", "--output-format", "json", "--no-cache", *present),
        cwd=tree,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RatchetError(f"ruff failed in {tree}: {completed.stderr.strip()}")
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for item in json.loads(completed.stdout or "[]"):
        path = Path(item["filename"])
        relative = path.relative_to(tree) if path.is_absolute() else path
        counts[(relative.as_posix(), item["code"] or "?")] += 1
    return counts


def pyright_counts(
    tree: Path, files: Iterable[str] | None
) -> Mapping[tuple[str, str], int]:
    """Count errors in existing files, or the configured whole tree for None.

    Both trees use the invoking worktree's interpreter and installed dependencies.
    The extracted base has no environment of its own.
    """
    present = (
        [name for name in files if (tree / name).exists()] if files is not None else []
    )
    if files is not None and not present:
        return {}
    completed = subprocess.run(
        ("pyright", "--pythonpath", sys.executable, "--outputjson", *present),
        cwd=tree,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RatchetError(f"pyright failed in {tree}: {completed.stderr.strip()}")
    try:
        diagnostics = json.loads(completed.stdout)["generalDiagnostics"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RatchetError(f"pyright produced an invalid report in {tree}") from error
    if not isinstance(diagnostics, list):
        raise RatchetError(f"pyright produced invalid diagnostics in {tree}")
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for item in diagnostics:
        if item.get("severity") != "error":
            continue
        path = Path(item["file"])
        relative = path.relative_to(tree) if path.is_absolute() else path
        counts[(relative.as_posix(), item.get("rule") or "?")] += 1
    return counts


def file_size_counts(tree: Path, files: Iterable[str]) -> Mapping[tuple[str, str], int]:
    """Return {(path, "lines"): line count} for oversize files among `files`."""
    checker = _support.load_tool("check_file_size")
    return {
        (str(item.path), "lines"): item.lines
        for item in checker.oversize_files(tree, paths=files)
    }


def path_correspondence_counts(
    tree: Path, files: Iterable[str]
) -> Mapping[tuple[str, str], int]:
    """Return {(test dir, "path-correspondence"): 1} for each violating directory.

    Unlike the others this rule is directory level, so it is compared over the
    whole tree rather than over the changed files: moving a test file can make a
    directory appear or disappear without that file being the one at fault.
    """
    del files
    checker = _support.load_tool("check_test_path_correspondence")
    return {
        (str(item.test_dir), "path-correspondence"): 1
        for item in checker.violations(tree)
    }


def test_structure_counts(
    tree: Path, files: Iterable[str]
) -> Mapping[tuple[str, str], int]:
    """Compare the whole test tree: new targets can resolve unchanged imports."""
    del files
    checker = _support.load_tool("check_test_structure")
    counts: dict[tuple[str, str], int] = {}
    try:
        found = checker.findings(tree)
    except (OSError, SyntaxError, UnicodeError, ValueError) as error:
        raise RatchetError(f"test structure check failed: {error}") from error
    for item in found:
        if item.severity == "blocking":
            key = (item.path, item.rule)
            counts[key] = counts.get(key, 0) + 1
    return counts


def capability_counts(
    tree: Path, files: Iterable[str]
) -> Mapping[tuple[str, str], int]:
    """Return {(module, "capability:<name>"): call sites} for undeclared capabilities.

    Narrowed to the changed files: the unit is the module, and a module can only
    gain a capability by being edited.
    """
    checker = _support.load_tool("check_test_capabilities")
    return {
        (str(item.module), f"capability:{item.capability}"): item.uses
        for item in checker.undeclared_capabilities(tree, paths=files)
    }


def suppression_counts(
    tree: Path, files: Iterable[str]
) -> Mapping[tuple[str, str], int]:
    """Return {(path, "suppression:<kind>"): count} for the changed files.

    Silencing a finding is cheaper than fixing it, so the silencers are counted
    alongside the findings. Clearing a ruff violation with `# noqa` then trades one
    counted thing for another and the total does not fall.
    """
    checker = _support.load_tool("check_suppressions")
    counts = {
        (str(item.path), f"suppression:{item.kind}"): item.count
        for item in checker.suppressions(tree, paths=files)
        if not str(item.path).startswith("pyproject.toml[")
    }
    # Configuration entries are compared unconditionally. They are not keyed by a
    # source file, and one of them can silence a rule for a whole glob without
    # appearing anywhere near the code it covers. The base tree is read through its
    # own configuration, not the candidate's copy, or adding a per-file-ignore
    # would be invisible to the very check meant to see it.
    base_config = _BASE_CONFIG if (tree / _BASE_CONFIG).is_file() else "pyproject.toml"
    counts.update(
        {
            (str(item.path), f"suppression:{item.kind}"): item.count
            for item in checker.config_suppressions(tree, filename=base_config)
        }
    )
    return counts


_DETECTORS: Final = {
    "capabilities": capability_counts,
    "suppressions": suppression_counts,
    "ruff": ruff_counts,
    "pyright": pyright_counts,
    "file-size": file_size_counts,
    "path-correspondence": path_correspondence_counts,
    "test-structure": test_structure_counts,
}


def regressions(
    before: Mapping[tuple[str, str], int], after: Mapping[tuple[str, str], int]
) -> tuple[Regression, ...]:
    """Return every (path, rule) whose count rose, worst increase first."""
    found = [
        Regression(path=path, rule=rule, before=before.get(key, 0), after=count)
        for key, count in after.items()
        for path, rule in (key,)
        if count > before.get(key, 0)
    ]
    return tuple(sorted(found, key=lambda item: (-item.increase, item.path, item.rule)))


def quality_settings(tree: Path, filename: str) -> dict[str, Any]:
    """Flatten quality settings for review, without guessing policy equivalence."""
    with (tree / filename).open("rb") as stream:
        tool = tomllib.load(stream).get("tool", {})
    settings: dict[str, Any] = {}

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                flatten(f"{prefix}.{key}", child)
        else:
            settings[prefix] = value

    for name in ("ruff", "pyright", "pytest"):
        if name in tool:
            flatten(f"pyproject.toml[tool.{name}]", tool[name])
    imports = tree / ".importlinter"
    if imports.is_file():
        settings[".importlinter"] = imports.read_text(encoding="utf-8")
    return settings


def run(
    root: Path,
    base: str | None,
    detectors: Iterable[str],
    *,
    full_pyright: bool = False,
) -> dict[str, Any]:
    resolved = resolve_base(root, base)
    touched = changed_paths(root, resolved)
    files = tuple(
        sorted(
            name
            for name in touched
            if name.endswith(".py") and name.split("/")[0] in _support.CHECKED_ROOTS
        )
    )
    found: dict[str, tuple[Regression, ...]] = {}
    configuration_changes: list[dict[str, Any]] = []
    selected = tuple(detectors)
    if full_pyright and "pyright" not in selected:
        raise RatchetError("--full-pyright requires the pyright detector")

    # pyproject.toml alone can widen a per-file-ignore or turn a Pyright rule off
    # repository-wide without touching one line of Python, so a candidate that
    # changes only the configuration still has to be judged.
    config_touched = bool({"pyproject.toml", ".importlinter"} & touched)
    if files or config_touched or full_pyright:
        scratch_root = root / ".agent_state" / "ratchet"
        scratch_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="comparison-", dir=scratch_root
        ) as scratch:
            destination = Path(scratch)
            extract_base_tree(root, resolved, destination)
            tree = destination / "tree"
            if config_touched:
                before = quality_settings(tree, _BASE_CONFIG)
                after = quality_settings(root, "pyproject.toml")
                configuration_changes = [
                    {"path": key, "before": before.get(key), "after": after.get(key)}
                    for key in sorted(before.keys() | after.keys())
                    if before.get(key) != after.get(key)
                ]
            for name in selected:
                if name == "pyright" and full_pyright:
                    found[name] = regressions(
                        pyright_counts(tree, None), pyright_counts(root, None)
                    )
                else:
                    detector = _DETECTORS[name]
                    found[name] = regressions(
                        detector(tree, files), detector(root, files)
                    )

    return {
        "base": resolved,
        "changed_files": list(files),
        "pyright_scope": "full" if full_pyright else "changed",
        "configuration_changes": configuration_changes,
        "detectors": {
            name: {
                "regression_count": len(items),
                "regressions": [
                    {
                        "after": item.after,
                        "before": item.before,
                        "path": item.path,
                        "rule": item.rule,
                    }
                    for item in items
                ],
            }
            for name, items in found.items()
        },
        "status": "FAIL" if any(found.values()) else "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=None,
        help=f"commit to judge against (default: merge-base with {_DEFAULT_BASE_BRANCH})",
    )
    parser.add_argument(
        "--detector",
        action="append",
        choices=sorted(_DETECTORS),
        help="restrict to one detector; repeatable (default: all)",
    )
    parser.add_argument(
        "--full-pyright",
        action="store_true",
        help="compare whole-tree Pyright errors, including unchanged callers (slow)",
    )
    arguments = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        report = run(
            root,
            arguments.base,
            arguments.detector or sorted(_DETECTORS),
            full_pyright=arguments.full_pyright,
        )
    except RatchetError as error:
        print(f"ratchet failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
