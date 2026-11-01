"""Reader-checker behavior through synthetic source mappings."""

from check_doc_readers import Violation, check_sources


def test_missing_declaration_reports_path_and_reason() -> None:
    assert check_sources({"guide.md": "# Guide\n\nRun the sweep.\n"}) == (
        Violation("guide.md", "missing reader declaration in first 8 body lines"),
    )
