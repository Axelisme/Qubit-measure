"""Markdown loading through the public adapter guide contract."""

from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.adapter import Adapter

from tests.gui.app.measure._synthetic import SampleAdapter


def test_guide_reads_utf8_markdown_and_keeps_resource_directory(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    markdown = "# 自由章節\n\n測量 café，$f_q$。\n\n![pulse](pulse.png)\n"
    source.write_text(markdown, encoding="utf-8")

    DocumentedAdapter = Adapter(
        spec=SampleAdapter.spec,
        make_form=SampleAdapter.make_form,
        run=SampleAdapter.run,
        guide=source,
        analysis=SampleAdapter.analysis,
        post_analysis=SampleAdapter.post_analysis,
    )

    guide = DocumentedAdapter.guide()
    assert guide.markdown == markdown
    assert guide.resource_dir == tmp_path


def test_undeclared_guide_reports_unwritten_and_no_resources() -> None:
    guide = SampleAdapter.guide()
    assert guide.markdown == "(no guide written yet)"
    assert guide.resource_dir is None


def test_declared_missing_guide_raises_instead_of_using_default(tmp_path: Path) -> None:
    MissingGuideAdapter = Adapter(
        spec=SampleAdapter.spec,
        make_form=SampleAdapter.make_form,
        run=SampleAdapter.run,
        guide=tmp_path / "missing.md",
        analysis=SampleAdapter.analysis,
        post_analysis=SampleAdapter.post_analysis,
    )

    with pytest.raises(FileNotFoundError):
        MissingGuideAdapter.guide()
