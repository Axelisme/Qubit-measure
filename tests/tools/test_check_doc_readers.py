"""Reader-checker behavior through synthetic source mappings."""

import json
import subprocess
from pathlib import Path

import pytest
from check_doc_readers import Violation, check_sources, main


@pytest.mark.parametrize("has_declaration", [False, True])
def test_main_reports_json_diagnostics_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    has_declaration: bool,
) -> None:
    source = "**Readers:** agent. Read this guide.\n" if has_declaration else "# Guide\n"
    (tmp_path / "guide.md").write_text(source, encoding="utf-8")
    (tmp_path / "untracked.md").write_text("# Not tracked\n", encoding="utf-8")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout=b"guide.md\\0", stderr=b""
        ),
    )

    assert main(["--root", str(tmp_path)]) == (0 if has_declaration else 1)
    captured = capsys.readouterr()
    violations = [] if has_declaration else [
        {"path": "guide.md", "reason": "missing reader declaration in first 8 body lines"}
    ]
    assert json.loads(captured.out) == {
        "status": "PASS" if has_declaration else "FAIL",
        "violation_count": len(violations),
        "violations": violations,
    }
    assert captured.err == (
        "" if has_declaration else "guide.md: missing reader declaration in first 8 body lines\n"
    )


def test_missing_declaration_reports_path_and_reason() -> None:
    assert check_sources({"guide.md": "# Guide\n\nRun the sweep.\n"}) == (
        Violation("guide.md", "missing reader declaration in first 8 body lines"),
    )


@pytest.mark.parametrize("readers", ["visitor", "agent, visitor", "framework-devish", ""])
def test_invalid_readers_are_rejected(readers: str) -> None:
    assert check_sources({"guide.md": f"**Readers:** {readers}. Run a sweep.\n"}) == (
        Violation("guide.md", "invalid reader list: " + repr(readers)),
    )


@pytest.mark.parametrize(
    "declaration",
    [
        "**Readers:** operator, lab-author, framework-dev, agent. Run a sweep.",
        "**讀者：** framework-dev、agent。維護工具。",
        "**讀者：** operator。執行量測。",
    ],
)
def test_valid_reader_combinations_pass(declaration: str) -> None:
    assert check_sources({"guide.md": "# Guide\n\n" + declaration}) == ()


@pytest.mark.parametrize("prefix", ["", "---\n" + "metadata: value\n" * 12 + "---\n"])
@pytest.mark.parametrize("body_line", [8, 9])
def test_header_window_counts_only_body_lines(prefix: str, body_line: int) -> None:
    source = prefix + "\n" * (body_line - 1) + "**Readers:** agent. Read this guide.\n"
    expected = (
        ()
        if body_line == 8
        else (Violation("guide.md", "missing reader declaration in first 8 body lines"),)
    )
    assert check_sources({"guide.md": source}) == expected


@pytest.mark.parametrize("path", ["lib/zcu_tools/example/README.md", "tests/tools/README.md", "tests/README.md"])
@pytest.mark.parametrize("date_line", ["", "**Last updated:** yesterday", "**Last updated:** 2026-1-01"])
def test_module_readme_requires_iso_date(path: str, date_line: str) -> None:
    source = "# Module\n\n" + date_line + "\n\n**Readers:** framework-dev. Maintain this module.\n"
    assert check_sources({path: source}) == (
        Violation(path, "missing or malformed **Last updated:** YYYY-MM-DD"),
    )


def test_module_date_can_include_a_topic_and_live_after_header() -> None:
    source = "**Readers:** framework-dev. Maintain this module.\n" + "\n" * 9
    source += "**Last updated:** 2026-11-01, reader checks\n"
    assert check_sources({"lib/example/README.md": source}) == ()


@pytest.mark.parametrize(
    "path",
    [
        "docs/adr/0001-example.md",
        "docs/adr/draft/example.md",
        "docs/adr/retired/example.md",
        "docs/discussions/2026-01-01-example.md",
        "docs/discussions/history/example.md",
    ],
)
def test_snapshots_are_exempt(path: str) -> None:
    assert check_sources({path: "# Historical snapshot\n"}) == ()


@pytest.mark.parametrize("path", ["docs/adr/README.md", "docs/discussions/INDEX.md"])
def test_snapshot_indexes_still_require_readers(path: str) -> None:
    assert check_sources({path: "# Index\n"}) == (
        Violation(path, "missing reader declaration in first 8 body lines"),
    )
    assert check_sources({path: "**Readers:** agent. Locate snapshots.\n"}) == ()


def test_non_module_readme_does_not_require_date() -> None:
    source = "**Readers:** agent. Read this guide.\n"
    assert check_sources({"tools/README.md": source, "README.md": source}) == ()


def test_front_matter_declaration_does_not_count_as_body() -> None:
    source = "---\n**Readers:** agent. Metadata only.\n---\n# Guide\n"
    assert check_sources({"guide.md": source}) == (
        Violation("guide.md", "missing reader declaration in first 8 body lines"),
    )
