"""Reader-checker behavior through synthetic source mappings."""

import pytest
from check_doc_readers import Violation, check_sources


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


def test_front_matter_declaration_does_not_count_as_body() -> None:
    source = "---\n**Readers:** agent. Metadata only.\n---\n# Guide\n"
    assert check_sources({"guide.md": source}) == (
        Violation("guide.md", "missing reader declaration in first 8 body lines"),
    )
