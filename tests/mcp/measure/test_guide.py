"""Markdown delivery from a registered adapter through RPC and MCP."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
from qtpy.QtWidgets import QApplication

from tests.gui.app.measure._synthetic import SampleAdapter
from tests.gui.app.measure.remote._helpers import Fixture, mcp_client

from ._pipeline_recipe import PIPELINE


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    """Keep a headless Qt owner alive for this module's real GUI seam."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QT_QPA_PLATFORM", "offscreen")
        patch.setenv("QT_QPA_PLATFORMTHEME", "generic")
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        assert isinstance(app, QApplication)
        yield app


@pytest.mark.parametrize("declared", [True, False])
def test_recipe_guide_delivers_markdown_from_registered_adapter(
    qapp, tmp_path: Path, request, monkeypatch: pytest.MonkeyPatch, declared: bool
) -> None:
    markdown = "# 指南\n\n自由 **Markdown**。\n![pulse](not-read.png)\n"
    (tmp_path / "guide.md").write_text(markdown, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    class GuideAdapter(SampleAdapter):
        guide_path: ClassVar[Path | None] = Path("guide.md") if declared else None

    fixture = Fixture(empty_project=True, headless=True)
    fixture.registry.register("guide-test", GuideAdapter)
    port = fixture.start()
    _, call = mcp_client(
        port,
        tmp_path,
        request=request,
        recipes=(replace(PIPELINE, adapter_name="guide-test"),),
    )
    try:
        call("connect", {"port": port})
        assert call("recipe_guide", {"recipe": "pipeline"}) == {
            "recipe": "pipeline",
            "adapter": "guide-test",
            "guide": {
                "markdown": markdown if declared else "(no guide written yet)",
                "resource_dir": str(tmp_path) if declared else None,
            },
        }
    finally:
        fixture.stop()
        qapp.processEvents()
