"""Browser-owned image overlays through Gallery and Comparison input events."""

from pathlib import Path

from qtpy.QtCore import Qt
from zcu_tools.gui.app.measure.ui.main_window import MainWindow

from tests.gui._user_input import ComponentDriver


def visible_bounds(ui: ComponentDriver, name: str) -> tuple[int, int, int, int]:
    """Read a shown widget rectangle in global logical pixels."""
    bounds = ui.observe(object_name=name).bounds
    assert bounds is not None, f"{name} must be visible"
    return bounds


def test_run_lightbox_fits_blocks_gallery_and_dismisses_with_browser(
    populated_browser: ComponentDriver,
    browser_host: MainWindow,
    tmp_path: Path,
) -> None:
    ui = populated_browser
    ui.click(object_name="results.figure.0")
    ui.click(object_name="results.figure.1")
    assert "results.lightbox" in {w.object_name for w in ui.visible().widgets}
    assert (
        ui.observe(object_name="results.lightbox.caption").text == "aaaaaa / plot-1.png"
    )
    for width, height in ((1400, 850), (1920, 1200), (1200, 1000)):
        ui.resize_window(width, height)
        overlay = visible_bounds(ui, "results.lightbox")
        browser = visible_bounds(ui, "results.browser")
        assert overlay == browser
        x, y, w, h = visible_bounds(ui, "results.lightbox.image")
        ax, ay, aw, ah = visible_bounds(ui, "results.lightbox.area")
        assert abs(w / h - 400 / 220) < 0.01
        assert abs(x + w / 2 - (ax + aw / 2)) <= 1
        assert abs(y + h / 2 - (ay + ah / 2)) <= 1
        assert w <= aw and h <= ah
        assert min(aw - w, ah - h) <= 1
        ui.screenshot(tmp_path / f"lightbox-{width}.png")
    ui.click(object_name="results.lightbox.image")
    assert ui.observe(object_name="results.lightbox").visible
    # Hit the covered Gallery image through Qt hit testing, not its widget.
    first = visible_bounds(ui, "results.figure.0")
    ui.click_at(first[0] + first[2] // 2, first[1] + first[3] // 2)
    assert ui.observe(object_name="results.lightbox").visible
    ui.press_key(Qt.Key.Key_Escape)
    assert (
        ui.observe(object_name="results.current_figure").text
        == "Current figure: plot-1.png"
    )
    ui.click(object_name="results.figure.0")
    assert (
        ui.observe(object_name="results.lightbox.caption").text == "aaaaaa / plot-0.png"
    )
    ui.click(object_name="results.lightbox.caption")
    assert not ui.observe(object_name="results.lightbox").visible
    ui.click(object_name="results.figure.1")
    ui.press_key(Qt.Key.Key_Escape)
    assert not ui.observe(object_name="results.lightbox").visible
    ui.click(object_name="results.figure.0")
    overlay = visible_bounds(ui, "results.lightbox")
    ui.click_at(overlay[0] + 2, overlay[1] + 2)
    assert not ui.observe(object_name="results.lightbox").visible
    ui.click(object_name="results.figure.0")
    ui.close_window()
    assert not ui.observe(object_name="results.lightbox").visible
    ComponentDriver(browser_host).click("Results browser…")
    assert not ui.observe(object_name="results.lightbox").visible
    ui.click(object_name="results.figure.1")
    assert (
        ui.observe(object_name="results.lightbox.caption").text == "aaaaaa / plot-1.png"
    )
    ui.screenshot(tmp_path / "lightbox-reopened.png")


def test_comparison_lightbox_uses_clicked_run_and_exact_png_not_gallery_focus(
    populated_browser: ComponentDriver, tmp_path: Path
) -> None:
    ui = populated_browser
    ui.resize_window(1600, 1100)
    ui.click(object_name="results.figure.0")
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.1")
    ui.select_tab("Gallery")
    ui.click("Back")
    ui.click(object_name="results.figure.1")
    assert (
        ui.observe(object_name="results.focus").text
        == "20261022T010000Z-bbbbbb · 1 runs"
    )
    ui.invoke_context_action(action="Add to comparison", object_name="results.figure.0")
    ui.click(object_name="results.compare.image.1.0")
    assert "results.lightbox" in {w.object_name for w in ui.visible().widgets}
    assert (
        ui.observe(object_name="results.lightbox.caption").text == "aaaaaa / plot-1.png"
    )
    assert ui.image_center_colour(object_name="results.lightbox.image") == "#00ffff"
    ui.screenshot(tmp_path / "comparison-exact-png-lightbox.png")
    ui.press_key(Qt.Key.Key_Escape)
    ui.click(object_name="results.compare.missing.1.1")
    assert not ui.observe(object_name="results.lightbox").visible
    ui.click(object_name="results.compare.image.0.1")
    assert (
        ui.observe(object_name="results.lightbox.caption").text == "bbbbbb / plot-0.png"
    )
    assert ui.image_center_colour(object_name="results.lightbox.image") == "#ffff00"
    ui.click(object_name="results.lightbox.caption")
    ui.select_tab("Gallery")
    assert (
        ui.observe(object_name="results.focus").text
        == "20261022T010000Z-bbbbbb · 1 runs"
    )
