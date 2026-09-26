"""Read-only quality snapshots and count comparisons, not acceptance verdicts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import tokenize
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import _support
import check_file_size
import check_suppressions
import check_test_capabilities
import check_test_path_correspondence
import check_test_structure
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Detector = Literal[
    "ruff",
    "pyright",
    "file-size",
    "test-paths",
    "test-structure",
    "test-capabilities",
    "suppressions",
    "radon",
]
DETECTORS: tuple[Detector, ...] = (
    "ruff",
    "pyright",
    "file-size",
    "test-paths",
    "test-structure",
    "test-capabilities",
    "suppressions",
    "radon",
)
NOTICE = "Counts only; not a gate verdict. Relocations are not paired."


class ReportError(ValueError):
    """An observation is invalid or cannot support a comparison."""


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Finding(Record):
    path: str = Field(min_length=1)
    rule: str = Field(min_length=1)
    count: int = Field(default=1, ge=1)
    line: int | None = Field(default=None, ge=1)
    message: str | None = None
    counted: bool = True
    details: dict[str, int | str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def relative_path(self) -> Self:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError("finding path must be repo-relative POSIX")
        if self.path != path.as_posix() or self.path == ".":
            raise ValueError("finding path must be canonical")
        return self


class DetectorResult(Record):
    state: Literal["completed", "skipped", "error"]
    selection: str = Field(min_length=1)
    findings: tuple[Finding, ...]
    reason: str | None = None

    @model_validator(mode="after")
    def consistent_state(self) -> Self:
        if self.state == "completed" and self.reason is not None:
            raise ValueError("completed observations cannot carry a reason")
        if self.state != "completed" and (
            self.findings or not self.reason or not self.reason.strip()
        ):
            raise ValueError("skipped/error needs a reason and cannot carry findings")
        return self


class GateResult(Record):
    state: Literal["pass", "fail", "error"]
    detail: str


class Method(Record):
    python: str
    platform: str
    distributions: dict[str, str]
    tool_versions: dict[str, str]
    configuration: dict[str, str]
    implementation: dict[str, str]


class Candidate(Record):
    commit: str
    tree: str
    status: str
    source_digest: str


class Snapshot(Record):
    schema_version: Literal[2]
    captured_at: str
    candidate: Candidate
    method: Method
    detectors: dict[Detector, DetectorResult]
    import_contracts: GateResult
    notice: str = NOTICE

    @model_validator(mode="after")
    def complete_inventory(self) -> Self:
        if set(self.detectors) != set(DETECTORS):
            raise ValueError("snapshot must account for every detector")
        return self


def relative_path(filename: str, root: Path) -> str:
    """Normalize a detector path without permitting paths outside the worktree."""
    path = Path(filename)
    if path.is_absolute():
        path = path.relative_to(root.resolve())
    return path.as_posix()


def object_value(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ReportError("expected a JSON object")
    return value


def string_value(value: JsonValue) -> str:
    if not isinstance(value, str) or not value:
        raise ReportError("expected a non-empty string")
    return value


def integer_value(value: JsonValue) -> int:
    if type(value) is not int:
        raise ReportError("expected an integer")
    return value


def normalize_diagnostics(
    detector: Literal["ruff", "pyright"], payload: JsonValue, root: Path
) -> tuple[Finding, ...]:
    """Adapt upstream diagnostics; preserve non-error Pyright notes uncounted."""
    rows = (
        payload if detector == "ruff" else object_value(payload)["generalDiagnostics"]
    )
    if not isinstance(rows, list):
        raise ReportError("expected a diagnostic list")
    found: list[Finding] = []
    for raw in rows:
        row = object_value(raw)
        if detector == "ruff":
            filename = string_value(row["filename"])
            line = integer_value(object_value(row["location"])["row"])
            rule = row["code"] or "unknown"
            counted = True
        else:
            filename = string_value(row["file"])
            start = object_value(object_value(row["range"])["start"])
            line = integer_value(start["line"]) + 1
            rule = row.get("rule") or "unknown"
            severity = string_value(row["severity"])
            if severity not in {"error", "warning", "information"}:
                raise ReportError(f"unknown Pyright severity: {severity}")
            counted = severity == "error"
        found.append(
            Finding(
                path=relative_path(filename, root),
                rule=string_value(rule),
                line=line,
                message=string_value(row["message"]),
                counted=counted,
            )
        )
    return tuple(sorted(found, key=lambda item: (item.path, item.rule, item.line or 0)))


class ComplexityBlock(BaseModel):
    """Validate Radon's recursive JSON before interpreting measurements."""

    model_config = ConfigDict(strict=True)
    type: Literal["function", "method", "class"]
    name: str = Field(min_length=1)
    lineno: int = Field(ge=1)
    col_offset: int = Field(default=0, ge=0)
    classname: str | None = None
    complexity: int = Field(ge=1)
    rank: Literal["A", "B", "C", "D", "E", "F"]
    methods: list[ComplexityBlock] = Field(default_factory=list)
    closures: list[ComplexityBlock] = Field(default_factory=list)
    inner_classes: list[ComplexityBlock] = Field(default_factory=list)


def normalize_complexity(payload: JsonValue, root: Path) -> tuple[Finding, ...]:
    """Retain function metrics, excluding class aggregates and violation counts."""
    found: dict[tuple[str, int, int], Finding] = {}

    def visit(path: str, block: ComplexityBlock, parent: str = "") -> None:
        owner = parent or block.classname
        name = f"{owner}.{block.name}" if owner else block.name
        key = (path, block.lineno, block.col_offset)
        if block.type != "class" and key not in found:
            found[key] = Finding(
                path=path,
                rule="cyclomatic-complexity",
                line=block.lineno,
                message=f"{name}: CC {block.complexity} ({block.rank})",
                counted=False,
                details={
                    "name": name,
                    "complexity": block.complexity,
                    "rank": block.rank,
                },
            )
        for child in (*block.methods, *block.closures, *block.inner_classes):
            visit(path, child, name)

    for filename, rows in object_value(payload).items():
        path = (root / filename).resolve().relative_to(root.resolve()).as_posix()
        if not isinstance(rows, list):
            raise ReportError(f"Radon could not analyze {path}: {rows}")
        blocks = [ComplexityBlock.model_validate(row) for row in rows]
        # Class-owned methods also occur at the top level of Radon's JSON.
        for block in sorted(blocks, key=lambda item: item.type != "class"):
            visit(path, block)
    return tuple(sorted(found.values(), key=lambda item: (item.path, item.line or 0)))


def complexity_summary(snapshot: Snapshot, *, top: int = 10) -> dict[str, JsonValue]:
    """Rank advisory CC values separately from counted diagnostic distributions."""
    if top < 1:
        raise ReportError("top must be positive")
    result = snapshot.detectors["radon"]
    groups: dict[str, Counter[str]] = {}
    for item in result.findings:
        scope = scope_for(item.path)
        groups.setdefault(scope, Counter())[str(item.details["rank"])] += 1
    hotspots = sorted(
        result.findings,
        key=lambda item: (-int(item.details["complexity"]), item.path, item.line or 0),
    )[:top]
    return {
        "state": result.state,
        "reason": result.reason,
        "ranks_by_scope": {
            scope: dict(sorted(ranks.items()))
            for scope, ranks in sorted(groups.items())
        },
        "hotspots": [item.model_dump(mode="json") for item in hotspots],
    }


def radon_findings(root: Path) -> tuple[Finding, ...]:
    """Analyze selected source with fixed options, independent of CLI config."""
    from radon.cli.tools import cc_to_dict
    from radon.complexity import cc_visit

    payload: dict[str, JsonValue] = {}
    for path in _support.python_files(root):
        relative = path.relative_to(root)
        if relative.parts[0] not in ("lib", "tools"):
            continue
        with tokenize.open(path) as source:
            blocks = cc_visit(source.read(), no_assert=False)
        serialized: list[JsonValue] = [dict(cc_to_dict(block)) for block in blocks]
        payload[relative.as_posix()] = serialized
    return normalize_complexity(payload, root)


def scope_for(path: str) -> str:
    if path.startswith("pyproject.toml["):
        return "configuration"
    return {
        "lib": "production",
        "tests": "tests",
        "script": "scripts",
        "tools": "tools",
    }.get(path.split("/", 1)[0], "other")


def selected_findings(
    snapshot: Snapshot, parent: str | None
) -> list[tuple[str, Finding]]:
    prefix = PurePosixPath(parent) if parent else None
    if prefix and (prefix.is_absolute() or ".." in prefix.parts):
        raise ReportError("parent must be repo-relative")
    return [
        (name, item)
        for name, result in sorted(snapshot.detectors.items())
        for item in result.findings
        if item.counted
        and (prefix is None or PurePosixPath(item.path).is_relative_to(prefix))
    ]


def module_for(detector: str, path: str) -> str:
    if scope_for(path) == "configuration":
        return "configuration"
    return path if detector == "test-paths" else str(PurePosixPath(path).parent)


def summarize(
    snapshot: Snapshot, parent: str | None = None
) -> dict[str, dict[str, int]]:
    """Return complete deterministic distributions, optionally under a directory."""
    groups: dict[str, Counter[str]] = {
        key: Counter() for key in ("detector", "scope", "module", "rule", "file")
    }
    for name, item in selected_findings(snapshot, parent):
        module = module_for(name, item.path)
        keys = {
            "detector": name,
            "scope": scope_for(item.path),
            "module": module,
            "rule": f"{name}:{item.rule}",
            "file": item.path,
        }
        for group, key in keys.items():
            groups[group][key] += item.count
    return {group: dict(sorted(counts.items())) for group, counts in groups.items()}


def counts(snapshot: Snapshot) -> Counter[tuple[str, str, str]]:
    result: Counter[tuple[str, str, str]] = Counter()
    for detector, finding in selected_findings(snapshot, None):
        result[(detector, finding.path, finding.rule)] += finding.count
    return result


def comparability_issues(before: Snapshot, after: Snapshot) -> list[str]:
    issues = [
        f"method.{field} differs"
        for field in Method.model_fields
        if getattr(before.method, field) != getattr(after.method, field)
    ]
    for name in DETECTORS:
        left, right = before.detectors[name], after.detectors[name]
        if left.selection != right.selection or left.state != right.state:
            issues.append(f"{name}: selection/state differs")
        if "error" in (left.state, right.state):
            issues.append(f"{name}: incomplete observation")
    if "error" in (before.import_contracts.state, after.import_contracts.state):
        issues.append("import contracts: incomplete observation")
    return issues


def compare(before: Snapshot, after: Snapshot) -> dict[str, JsonValue]:
    """Compare exact path/rule counts, never cancel increases against reductions."""
    issues = comparability_issues(before, after)
    if issues:
        raise ReportError("not comparable: " + "; ".join(issues))
    old, new = counts(before), counts(after)
    changes: list[JsonValue] = []
    groups: dict[str, dict[str, dict[str, int]]] = {
        key: {} for key in ("detector", "scope", "module", "rule")
    }
    for key in sorted(old.keys() | new.keys()):
        detector, path, rule = key
        delta = new[key] - old[key]
        values = {
            "before": old[key],
            "after": new[key],
            "introduced_count": max(delta, 0),
            "resolved_count": max(-delta, 0),
            "net": delta,
        }
        changes.append({"detector": detector, "path": path, "rule": rule, **values})
        labels = {
            "detector": detector,
            "scope": scope_for(path),
            "module": module_for(detector, path),
            "rule": f"{detector}:{rule}",
        }
        for group, label in labels.items():
            total = groups[group].setdefault(label, dict.fromkeys(values, 0))
            for metric, value in values.items():
                total[metric] += value
    return {
        "schema_version": 1,
        "status": "comparable",
        "notice": NOTICE,
        "before": before.candidate.model_dump(mode="json"),
        "after": after.candidate.model_dump(mode="json"),
        "changes": changes,
        "groups": {
            group: {
                label: {metric: value for metric, value in totals.items()}
                for label, totals in labels.items()
            }
            for group, labels in groups.items()
        },
        "import_contracts": {
            "before": before.import_contracts.state,
            "after": after.import_contracts.state,
        },
    }


def command(root: Path, args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=root, capture_output=True, text=True, check=False, timeout=180
    )


def git(root: Path, *args: str) -> str:
    result = command(root, ("git", *args))
    if result.returncode:
        raise ReportError(result.stderr.strip())
    return result.stdout.strip()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def configuration_files(root: Path) -> dict[str, str]:
    names = git(
        root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    ).split("\0")
    known = {
        "pyproject.toml",
        "ruff.toml",
        ".ruff.toml",
        "pyrightconfig.json",
        ".importlinter",
        "uv.lock",
        ".gitignore",
    }
    return {
        name: digest((root / name).read_bytes())
        for name in sorted(set(names))
        if PurePosixPath(name).name in known and (root / name).is_file()
    }


def candidate(root: Path) -> Candidate:
    sources = {
        str(path.relative_to(root)): digest(path.read_bytes())
        for path in _support.python_files(root)
    }
    sources.update(configuration_files(root))
    return Candidate(
        commit=git(root, "rev-parse", "HEAD"),
        tree=git(root, "rev-parse", "HEAD^{tree}"),
        status=git(root, "status", "--porcelain=v1", "--untracked-files=all"),
        source_digest=digest(json.dumps(sources, sort_keys=True).encode()),
    )


def normalize_tool_version(tool: str, output: str) -> str:
    """Keep the installed version line, excluding update announcements."""
    label = "import-linter" if tool == "lint-imports" else tool
    pattern = rf"{re.escape(label)} ([0-9]+(?:\.[0-9]+)+(?:[a-zA-Z0-9.+-]*))"
    matches = [
        match.group(1)
        for line in output.splitlines()
        if (match := re.fullmatch(pattern, line.strip()))
    ]
    if len(matches) != 1:
        raise ReportError(f"expected one installed version for {tool}")
    return matches[0]


def measurement_method(root: Path) -> Method:
    tools_dir = Path(__file__).resolve().parent
    versions: dict[str, str] = {}
    for name in ("ruff", "pyright", "lint-imports"):
        result = command(root, (name, "--version"))
        if result.returncode:
            raise ReportError(f"cannot identify {name}: {result.stderr.strip()}")
        versions[name] = normalize_tool_version(name, result.stdout)
    return Method(
        python=sys.version,
        platform=platform.platform(),
        distributions={
            str(dist.metadata["Name"]).lower(): dist.version
            for dist in importlib.metadata.distributions()
        },
        tool_versions=versions,
        configuration=configuration_files(root),
        implementation={
            path.name: digest(path.read_bytes())
            for path in sorted(tools_dir.glob("*.py"))
        },
    )


def local_findings(name: Detector, root: Path) -> tuple[Finding, ...]:
    if name == "file-size":
        return tuple(
            Finding(
                path=str(item.path),
                rule="file-size",
                details={"lines": item.lines, "limit": check_file_size.LINE_LIMIT},
            )
            for item in check_file_size.oversize_files(root)
        )
    if name == "test-paths":
        return tuple(
            Finding(
                path=str(item.test_dir),
                rule="test-path",
                message=f"No module at {item.expected_source_dir}",
            )
            for item in check_test_path_correspondence.violations(root)
        )
    if name == "test-structure":
        return tuple(
            Finding(
                path=item.path,
                rule=item.rule,
                line=item.line,
                message=item.message,
                counted=item.severity == "blocking",
            )
            for item in check_test_structure.findings(root)
        )
    if name == "test-capabilities":
        return tuple(
            Finding(
                path=str(item.module),
                rule=item.capability,
                message=f"Missing {item.marker}",
                details={"uses": item.uses},
            )
            for item in check_test_capabilities.undeclared_capabilities(root)
        )
    if name == "suppressions":
        return tuple(
            Finding(path=str(item.path), rule=item.kind, count=item.count)
            for item in check_suppressions.suppressions(root)
            if item.count
        )
    raise ReportError(f"not a local detector: {name}")


def external_findings(
    name: Literal["ruff", "pyright"], root: Path
) -> tuple[Finding, ...]:
    args = (
        ("ruff", "check", "--output-format", "json", "--no-cache", ".")
        if name == "ruff"
        else ("pyright", "--pythonpath", sys.executable, "--outputjson")
    )
    result = command(root, args)
    if result.returncode not in (0, 1):
        raise ReportError(f"{name} exit {result.returncode}: {result.stderr.strip()}")
    findings = normalize_diagnostics(name, json.loads(result.stdout), root)
    if result.returncode == 1 and not any(item.counted for item in findings):
        raise ReportError(
            f"{name} exited 1 without counted diagnostics: {result.stderr.strip()}"
        )
    return findings


def observe(
    selection: str, collect: Callable[[], tuple[Finding, ...]]
) -> DetectorResult:
    try:
        return DetectorResult(
            state="completed", selection=selection, findings=collect()
        )
    except (
        ImportError,
        OSError,
        ValueError,
        SyntaxError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        return DetectorResult(
            state="error",
            findings=(),
            selection=selection,
            reason=f"{type(error).__name__}: {error}",
        )


def import_contracts(root: Path) -> GateResult:
    try:
        result = command(root, ("lint-imports",))
        state: Literal["pass", "fail", "error"] = "error"
        if result.returncode == 0:
            state = "pass"
        elif result.returncode == 1:
            state = "fail"
        return GateResult(state=state, detail=(result.stdout + result.stderr).strip())
    except (OSError, subprocess.SubprocessError) as error:
        return GateResult(state="error", detail=str(error))


def collect_snapshot(
    root: Path, *, with_pyright: bool = False, with_radon: bool = False
) -> Snapshot:
    """Measure without changing source; reject concurrent source/method changes."""
    root = root.resolve()
    before = candidate(root)
    method = measurement_method(root)
    results: dict[Detector, DetectorResult] = {}
    for name in DETECTORS:
        selection = (
            "configured whole tree; Pyright errors counted, other severities retained"
            if name == "pyright"
            else "whole tree; detector defaults"
        )
        if name == "radon":
            selection = "lib/tools Python functions, methods and closures; class aggregates excluded; asserts counted"
            results[name] = (
                observe(selection, lambda: radon_findings(root))
                if with_radon
                else DetectorResult(
                    state="skipped",
                    findings=(),
                    selection=selection,
                    reason="Radon not requested",
                )
            )
        elif name == "pyright" and not with_pyright:
            results[name] = DetectorResult(
                state="skipped",
                findings=(),
                selection=selection,
                reason="Pyright not requested",
            )
        elif name in ("ruff", "pyright"):
            results[name] = observe(
                selection, lambda name=name: external_findings(name, root)
            )
        else:
            results[name] = observe(
                selection, lambda name=name: local_findings(name, root)
            )
    contracts = import_contracts(root)
    if before != candidate(root) or method != measurement_method(root):
        raise ReportError("source or measurement method changed during snapshot")
    return Snapshot(
        schema_version=2,
        captured_at=datetime.now(timezone.utc).isoformat(),
        candidate=before,
        method=method,
        detectors=results,
        import_contracts=contracts,
    )


def emit_summary(snapshot: Snapshot, top: int) -> None:
    for name, result in sorted(snapshot.detectors.items()):
        if name == "radon":
            continue
        count = (
            str(sum(item.count for item in result.findings if item.counted))
            if result.state == "completed"
            else result.reason
        )
        print(f"{name}: {result.state} {count}", file=sys.stderr)
    print(f"import contracts: {snapshot.import_contracts.state}", file=sys.stderr)
    for group, distribution in summarize(snapshot).items():
        print(
            f"{group}: {sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[:top]}",
            file=sys.stderr,
        )
    radon = complexity_summary(snapshot, top=top)
    print(
        f"Radon CC advisory: {radon['state']} {radon['reason'] or ''}", file=sys.stderr
    )
    print(f"Radon ranks by scope: {radon['ranks_by_scope']}", file=sys.stderr)
    hotspots = radon["hotspots"]
    if isinstance(hotspots, list):
        for item in hotspots:
            row = object_value(item)
            print(f"  {row['path']}:{row['line']} {row['message']}", file=sys.stderr)
    print(NOTICE, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    snapshot_parser = sub.add_parser("snapshot")
    snapshot_parser.add_argument("--with-pyright", action="store_true")
    snapshot_parser.add_argument("--with-radon", action="store_true")
    snapshot_parser.add_argument("--top", type=int, default=10)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    for child in (snapshot_parser, compare_parser):
        child.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        exit_code = 0
        if args.operation == "snapshot":
            if args.top < 1:
                raise ReportError("--top must be positive")
            snapshot = collect_snapshot(
                Path(__file__).resolve().parents[1],
                with_pyright=args.with_pyright,
                with_radon=args.with_radon,
            )
            output = snapshot.model_dump_json(indent=2)
            emit_summary(snapshot, args.top)
            if (
                any(result.state == "error" for result in snapshot.detectors.values())
                or snapshot.import_contracts.state == "error"
            ):
                exit_code = 2
        else:
            before = Snapshot.model_validate_json(
                args.before.read_text(encoding="utf-8")
            )
            after = Snapshot.model_validate_json(args.after.read_text(encoding="utf-8"))
            output = json.dumps(compare(before, after), indent=2, sort_keys=True)
            print(NOTICE, file=sys.stderr)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(output + "\n")
        else:
            print(output)
        return exit_code
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"quality report error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
