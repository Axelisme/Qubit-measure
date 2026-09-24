from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

_IMPORT_MODE: Final = "--import-mode=importlib"
_ENTRYPOINTS: Final = ("module", "console")
_SUMMARY_PATTERN: Final = re.compile(
    r"^(?P<count>\d+) tests? collected in (?P<duration>.+)$", re.MULTILINE
)
PytestEntrypoint = Literal["module", "console"]


class CollectionOracleError(RuntimeError):
    """Raised when collection evidence is absent, failed, or inconsistent."""


@dataclass(frozen=True)
class CollectionResult:
    command: tuple[str, ...]
    node_ids: tuple[str, ...]
    summary: str

    @property
    def digest(self) -> str:
        payload = "\n".join(self.node_ids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def configured_addopts(root: Path) -> tuple[str, ...]:
    with (root / "pyproject.toml").open("rb") as stream:
        raw = tomllib.load(stream)["tool"]["pytest"]["ini_options"].get("addopts", "")
    if isinstance(raw, str):
        return tuple(shlex.split(raw))
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        return tuple(raw)
    raise CollectionOracleError("pytest addopts must be a string or list of strings")


def console_script_path() -> Path:
    """Return the pytest console script installed beside this interpreter."""
    script_name = "pytest.exe" if sys.platform == "win32" else "pytest"
    return Path(sys.executable).with_name(script_name)


def build_command(
    root: Path, *, parallel: bool, entrypoint: PytestEntrypoint = "module"
) -> tuple[str, ...]:
    tests = str(root / "tests")
    if entrypoint == "module":
        command = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    elif entrypoint == "console":
        command = [str(console_script_path()), "--collect-only", "-q"]
    else:
        raise CollectionOracleError(f"unsupported pytest entrypoint: {entrypoint!r}")
    if parallel:
        command.extend(("-n", "auto"))
    command.append(tests)
    return tuple(command)


def parse_collection(
    *, command: tuple[str, ...], returncode: int, stdout: str, stderr: str
) -> CollectionResult:
    if returncode != 0:
        detail = (stderr or stdout).strip().splitlines()
        tail = detail[-1] if detail else "no pytest output"
        raise CollectionOracleError(f"collection exited {returncode}: {tail}")

    summaries = tuple(_SUMMARY_PATTERN.finditer(stdout))
    if len(summaries) != 1:
        raise CollectionOracleError("collection must emit exactly one pytest summary")

    node_ids = tuple(
        sorted(
            line.strip()
            for line in stdout.splitlines()
            if line.startswith("tests/") and "::" in line
        )
    )
    if not node_ids:
        raise CollectionOracleError("collection emitted no test node ids")
    if len(set(node_ids)) != len(node_ids):
        raise CollectionOracleError("collection emitted duplicate test node ids")

    expected_count = int(summaries[0].group("count"))
    if expected_count != len(node_ids):
        raise CollectionOracleError(
            "pytest summary count does not match emitted node ids: "
            f"{expected_count} != {len(node_ids)}"
        )
    return CollectionResult(
        command=command,
        node_ids=node_ids,
        summary=summaries[0].group(0),
    )


def collect(
    root: Path, *, parallel: bool, entrypoint: PytestEntrypoint = "module"
) -> CollectionResult:
    command = build_command(root, parallel=parallel, entrypoint=entrypoint)
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTHONPATH", None)
    # The summary is parsed as text, so colour escapes would break the match. A
    # terminal that exports FORCE_COLOR makes pytest emit them even through a pipe.
    environment.pop("FORCE_COLOR", None)
    environment["NO_COLOR"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return parse_collection(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def compare_collections(*results: CollectionResult) -> None:
    if len(results) < 2:
        raise CollectionOracleError("at least two collection results are required")
    baseline = results[0]
    for candidate in results[1:]:
        if baseline.node_ids == candidate.node_ids:
            continue
        baseline_only = sorted(set(baseline.node_ids) - set(candidate.node_ids))
        candidate_only = sorted(set(candidate.node_ids) - set(baseline.node_ids))
        raise CollectionOracleError(
            "collection differ: "
            f"baseline_only={baseline_only[:3]!r}, "
            f"candidate_only={candidate_only[:3]!r}"
        )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        addopts = configured_addopts(root)
        if _IMPORT_MODE not in addopts:
            raise CollectionOracleError(f"pytest addopts must include {_IMPORT_MODE}")
        matrix = [
            (
                entrypoint,
                parallel,
                collect(root, parallel=parallel, entrypoint=entrypoint),
            )
            for entrypoint in _ENTRYPOINTS
            for parallel in (False, True)
        ]
        compare_collections(*(result for _, _, result in matrix))
    except CollectionOracleError as exc:
        print(f"collection oracle failed: {exc}", file=sys.stderr)
        return 1

    receipt = {
        "collection_count": len(matrix[0][2].node_ids),
        "collection_sha256": matrix[0][2].digest,
        "collections": [
            {
                "command": list(result.command),
                "entrypoint": entrypoint,
                "exit_code": 0,
                "parallel": parallel,
                "summary": result.summary,
            }
            for entrypoint, parallel, result in matrix
        ],
        "import_mode": _IMPORT_MODE,
        "status": "PASS",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
