"""Behavioral contracts for quality data, independent of tool subprocesses."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from quality_report import (
    DETECTORS,
    Candidate,
    DetectorResult,
    Finding,
    GateResult,
    Method,
    ReportError,
    Snapshot,
    compare,
    normalize_diagnostics,
    normalize_tool_version,
    observe,
    summarize,
)


def snapshot(*findings: Finding) -> Snapshot:
    return Snapshot(
        schema_version=1,
        captured_at="2026-09-25T00:00:00+00:00",
        candidate=Candidate(
            commit="base", tree="tree", status="", source_digest="source"
        ),
        method=Method(
            python="3.13",
            platform="linux",
            distributions={"pydantic": "2"},
            tool_versions={"ruff": "1"},
            configuration={"pyproject.toml": "abc"},
            implementation={"report.py": "def"},
        ),
        detectors={
            name: DetectorResult(
                state="completed",
                selection="whole tree",
                findings=findings if name == "ruff" else (),
            )
            for name in DETECTORS
        },
        import_contracts=GateResult(state="pass", detail="contracts kept"),
    )


def test_counts_keep_scope_rule_and_directory_ownership() -> None:
    report = snapshot(
        Finding(path="lib/pkg/a.py", rule="A", count=2),
        Finding(path="lib/pkg/a.py", rule="B"),
        Finding(path="tests/pkg/test_a.py", rule="A"),
        Finding(path="tools/helper.py", rule="A"),
        Finding(path="script/run.py", rule="A"),
        Finding(
            path="pyproject.toml[per-file-ignores:tests/**]", rule="ignore", count=4
        ),
        Finding(path="tests/pkg/test_a.py", rule="advisory", counted=False),
    )
    result = summarize(report)
    assert result["scope"] == {
        "production": 3,
        "tests": 1,
        "tools": 1,
        "scripts": 1,
        "configuration": 4,
    }
    assert result["rule"]["ruff:A"] == 5
    assert result["module"]["configuration"] == 4
    assert summarize(report, "lib/pkg")["file"] == {"lib/pkg/a.py": 3}
    assert summarize(report, "lib/p")["file"] == {}
    assert list(result["file"]) == sorted(result["file"])


def test_net_zero_does_not_hide_local_increase_or_reduction() -> None:
    before = snapshot(Finding(path="lib/a.py", rule="A", count=2))
    after = snapshot(
        Finding(path="lib/a.py", rule="A"), Finding(path="lib/b.py", rule="A")
    )
    result = compare(before, after)
    assert result["changes"] == [
        {
            "detector": "ruff",
            "path": "lib/a.py",
            "rule": "A",
            "before": 2,
            "after": 1,
            "introduced_count": 0,
            "resolved_count": 1,
            "net": -1,
        },
        {
            "detector": "ruff",
            "path": "lib/b.py",
            "rule": "A",
            "before": 0,
            "after": 1,
            "introduced_count": 1,
            "resolved_count": 0,
            "net": 1,
        },
    ]
    assert result["groups"] == {
        group: {
            key: {
                "before": 2,
                "after": 2,
                "introduced_count": 1,
                "resolved_count": 1,
                "net": 0,
            }
        }
        for group, key in (
            ("detector", "ruff"),
            ("scope", "production"),
            ("module", "lib"),
            ("rule", "ruff:A"),
        )
    }


def test_relocation_is_not_claimed_as_a_repair() -> None:
    result = compare(
        snapshot(Finding(path="lib/old.py", rule="A")),
        snapshot(Finding(path="lib/new.py", rule="A")),
    )
    assert result["status"] == "comparable"
    assert "Relocations are not paired" in str(result["notice"])
    changes = result["changes"]
    assert isinstance(changes, list)
    assert len(changes) == 2


def test_receipt_round_trip_retains_nonblocking_details() -> None:
    report = snapshot(
        Finding(
            path="lib/a.py",
            rule="A",
            line=7,
            message="detail",
            counted=False,
            details={"lines": 1200},
        )
    )
    restored = Snapshot.model_validate_json(report.model_dump_json())
    assert restored == report
    assert summarize(restored)["file"] == {}


@pytest.mark.parametrize("field", list(Method.model_fields))
def test_method_changes_cannot_be_presented_as_improvement(field: str) -> None:
    before = snapshot()
    data = json.loads(before.model_dump_json())
    data["method"][field] = (
        "different" if field in {"python", "platform"} else {"changed": "value"}
    )
    after = Snapshot.model_validate_json(json.dumps(data))
    with pytest.raises(ReportError, match=f"method.{field} differs"):
        compare(before, after)


@pytest.mark.parametrize("state", ["skipped", "error"])
def test_missing_observations_are_not_zero_counts(state: str) -> None:
    before = snapshot()
    data = json.loads(before.model_dump_json())
    data["detectors"]["pyright"] = {
        "state": state,
        "selection": "whole tree",
        "reason": "not measured",
        "findings": [],
    }
    after = Snapshot.model_validate_json(json.dumps(data))
    with pytest.raises(ReportError, match="selection/state differs"):
        compare(before, after)
    if state == "error":
        with pytest.raises(ReportError, match="incomplete observation"):
            compare(after, after)
    else:
        assert compare(after, after)["changes"] == []


@pytest.mark.parametrize(
    "mutation", ["schema", "inventory", "state", "selection", "gate"]
)
def test_invalid_or_incompatible_receipts_fail_explicitly(mutation: str) -> None:
    before = snapshot()
    data = json.loads(before.model_dump_json())
    if mutation == "schema":
        data["schema_version"] = 2
    elif mutation == "inventory":
        del data["detectors"]["ruff"]
    elif mutation == "state":
        data["detectors"]["ruff"]["state"] = "error"
    elif mutation == "selection":
        data["detectors"]["ruff"]["selection"] = "changed files only"
    else:
        data["import_contracts"]["state"] = "error"
    if mutation in {"schema", "inventory", "state"}:
        with pytest.raises(ValidationError):
            Snapshot.model_validate_json(json.dumps(data))
    else:
        after = Snapshot.model_validate_json(json.dumps(data))
        with pytest.raises(ReportError, match="not comparable"):
            compare(before, after)


@pytest.mark.parametrize(
    "path",
    ["/tmp/escape.py", "../escape.py", "lib/../escape.py", "lib\\escape.py", "."],
)
def test_finding_rejects_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="path must"):
        Finding(path=path, rule="A")


def test_ruff_adapter_retains_source_location_and_message(tmp_path: Path) -> None:
    findings = normalize_diagnostics(
        "ruff",
        [
            {
                "filename": str(tmp_path / "lib/a.py"),
                "code": "C901",
                "location": {"row": 9},
                "message": "too complex",
            }
        ],
        tmp_path,
    )
    assert findings == (
        Finding(path="lib/a.py", rule="C901", line=9, message="too complex"),
    )


def test_pyright_adapter_retains_warnings_without_counting_as_errors(
    tmp_path: Path,
) -> None:
    findings = normalize_diagnostics(
        "pyright",
        {
            "generalDiagnostics": [
                {
                    "file": str(tmp_path / "tests/a.py"),
                    "rule": "reportPrivateUsage",
                    "range": {"start": {"line": 4}},
                    "message": "private",
                    "severity": "error",
                },
                {
                    "file": str(tmp_path / "lib/b.py"),
                    "range": {"start": {"line": 0}},
                    "message": "warning",
                    "severity": "warning",
                },
            ]
        },
        tmp_path,
    )
    assert [(item.path, item.line, item.counted) for item in findings] == [
        ("lib/b.py", 1, False),
        ("tests/a.py", 5, True),
    ]


@pytest.mark.parametrize(
    "payload", [None, {}, {"generalDiagnostics": {}}, {"generalDiagnostics": [None]}]
)
def test_malformed_upstream_output_cannot_appear_clean(payload, tmp_path: Path) -> None:
    with pytest.raises((ReportError, KeyError)):
        normalize_diagnostics("pyright", payload, tmp_path)


def test_observation_failure_remains_distinct_from_empty_success() -> None:
    def broken_source() -> tuple[Finding, ...]:
        raise ReportError("malformed upstream report")

    failed = observe("whole tree", broken_source)
    clean = observe("whole tree", lambda: ())
    assert failed.state == "error"
    assert failed.reason == "ReportError: malformed upstream report"
    assert clean.state == "completed"
    assert clean.findings == ()


@pytest.mark.parametrize("missing", ["schema_version", "findings"])
def test_incomplete_receipt_cannot_default_to_a_clean_observation(missing: str) -> None:
    data = json.loads(snapshot().model_dump_json())
    owner = data if missing == "schema_version" else data["detectors"]["ruff"]
    del owner[missing]
    with pytest.raises(ValidationError, match="Field required"):
        Snapshot.model_validate_json(json.dumps(data))


def test_completed_observation_rejects_a_not_measured_reason() -> None:
    with pytest.raises(
        ValidationError, match="completed observations cannot carry a reason"
    ):
        DetectorResult(
            state="completed",
            selection="whole tree",
            findings=(),
            reason="Pyright not requested",
        )


def test_directory_findings_own_the_reported_module_in_summary_and_comparison() -> None:
    before = snapshot()
    data = json.loads(before.model_dump_json())
    data["detectors"]["test-paths"]["findings"] = [
        {"path": "tests/tools/ghost", "rule": "test-path"}
    ]
    after = Snapshot.model_validate_json(json.dumps(data))
    assert summarize(after)["module"] == {"tests/tools/ghost": 1}
    groups = compare(before, after)["groups"]
    assert isinstance(groups, dict)
    assert groups["module"] == {
        "tests/tools/ghost": {
            "before": 0,
            "after": 1,
            "introduced_count": 1,
            "resolved_count": 0,
            "net": 1,
        }
    }


def test_update_announcements_do_not_change_method_but_installed_versions_do() -> None:
    data = json.loads(snapshot().model_dump_json())
    data["method"]["tool_versions"]["pyright"] = normalize_tool_version(
        "pyright", "pyright 1.1.411\nWARNING: new version available 1.1.414"
    )
    before = Snapshot.model_validate_json(json.dumps(data))
    data["method"]["tool_versions"]["pyright"] = normalize_tool_version(
        "pyright", "pyright 1.1.411\nWARNING: new version available 1.1.415"
    )
    after = Snapshot.model_validate_json(json.dumps(data))
    assert compare(before, after)["status"] == "comparable"
    data["method"]["tool_versions"]["pyright"] = normalize_tool_version(
        "pyright", "pyright 1.1.414"
    )
    changed = Snapshot.model_validate_json(json.dumps(data))
    with pytest.raises(ReportError, match="method.tool_versions differs"):
        compare(before, changed)


@pytest.mark.parametrize(
    "tool, output, expected",
    [
        ("ruff", "ruff 0.15.20\n", "0.15.20"),
        ("lint-imports", "import-linter 2.15\n", "2.15"),
    ],
)
def test_upstream_version_labels_are_normalized(
    tool: str, output: str, expected: str
) -> None:
    assert normalize_tool_version(tool, output) == expected


@pytest.mark.parametrize(
    "output", ["", "WARNING: pyright 1.1.411", "pyright 1.1.411\npyright 1.1.414"]
)
def test_missing_or_ambiguous_installed_version_is_not_fingerprinted(
    output: str,
) -> None:
    with pytest.raises(ReportError, match="expected one installed version"):
        normalize_tool_version("pyright", output)


def test_candidate_changes_are_allowed_and_gate_failure_is_separate() -> None:
    before = snapshot()
    data = json.loads(before.model_dump_json())
    data["candidate"]["commit"] = "next"
    data["candidate"]["source_digest"] = "new source"
    data["import_contracts"]["state"] = "fail"
    after = Snapshot.model_validate_json(json.dumps(data))
    result = compare(before, after)
    assert result["status"] == "comparable"
    assert result["import_contracts"] == {"before": "pass", "after": "fail"}
