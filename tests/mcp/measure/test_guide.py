"""Markdown delivery from a registered adapter through RPC and MCP."""

from pathlib import Path
from typing import ClassVar

import pytest

from tests.gui.app.measure._synthetic import SampleAdapter
from tests.gui.app.measure.remote._helpers import Fixture, mcp_client


@pytest.mark.parametrize("declared", [True, False])
def test_recipe_guide_delivers_markdown_from_registered_adapter(
    qapp, tmp_path: Path, request, monkeypatch: pytest.MonkeyPatch, declared: bool
) -> None:
    markdown = "# 指南\n\n自由 **Markdown**。\n![pulse](not-read.png)\n"
    (tmp_path / "guide.md").write_text(markdown, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    class GuideAdapter(SampleAdapter):
        guide_path: ClassVar[Path | None] = Path("guide.md") if declared else None

    fixture = Fixture(
        project_root=str(tmp_path / "project"), empty_project=True, headless=True
    )
    fixture.registry.register("guide-test", GuideAdapter)
    port = fixture.start()
    # Only guide delivery is observed, so the bridge is unused; the request
    # finalizer registered by mcp_client closes the MCP session.
    _, call = mcp_client(
        port,
        tmp_path,
        request=request,
    )
    try:
        call("connect", {"port": port})
        assert call("recipe_guide", {"adapter": "guide-test"}) == {
            "adapter": "guide-test",
            "guide": {
                "markdown": markdown if declared else "(no guide written yet)",
                "resource_dir": str(tmp_path) if declared else None,
            },
        }
    finally:
        fixture.stop()
        qapp.processEvents()
