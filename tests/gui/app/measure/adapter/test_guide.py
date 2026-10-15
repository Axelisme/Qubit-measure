"""Markdown loading through the public adapter guide contract."""

from pathlib import Path
from typing import ClassVar

import pytest

from tests.gui.app.measure._synthetic import SampleAdapter


def test_guide_reads_utf8_markdown_and_keeps_resource_directory(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    markdown = "# 自由章節\n\n測量 café，$f_q$。\n\n![pulse](pulse.png)\n"
    source.write_text(markdown, encoding="utf-8")

    class DocumentedAdapter(SampleAdapter):
        guide_path: ClassVar[Path | None] = source

    guide = DocumentedAdapter.guide()
    assert guide.markdown == markdown
    assert guide.resource_dir == tmp_path


def test_undeclared_guide_reports_unwritten_and_no_resources() -> None:
    guide = SampleAdapter.guide()
    assert guide.markdown == "(no guide written yet)"
    assert guide.resource_dir is None


def test_declared_missing_guide_raises_instead_of_using_default(tmp_path: Path) -> None:
    class MissingGuideAdapter(SampleAdapter):
        guide_path: ClassVar[Path | None] = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError):
        MissingGuideAdapter.guide()
