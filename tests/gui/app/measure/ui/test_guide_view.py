"""Guide subtab presentation of AdapterGuide Markdown in ExpTabWidget.

The tab reads the guide through the public controller query and shows it in a
read-only text browser; every observation goes through public Qt APIs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEvent, QPoint, QPointF, Qt, QUrl
from qtpy.QtGui import (
    QDesktopServices,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QTextDocument,
    QTextImageFormat,
)
from qtpy.QtWidgets import QApplication, QTabWidget, QTextBrowser
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AdapterGuide
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.control_identity import ControlIdentity

from tests.gui.app.measure._cfg_fakes import configure_cfg_lookup

_IMAGE_RESOURCE = QTextDocument.ResourceType.ImageResource.value


@pytest.fixture
def open_guide(qapp: QApplication) -> Iterator[Callable[[AdapterGuide], QTextBrowser]]:
    """Open one tab whose controller serves the given guide; return its browser."""
    owned_tabs: list[ExpTabWidget] = []
    owned_authorities: list[ControlIdentity] = []

    def create(guide: AdapterGuide) -> QTextBrowser:
        ctrl = MagicMock()
        ctrl.control_identity = ControlIdentity()
        owned_authorities.append(ctrl.control_identity)
        configure_cfg_lookup(ctrl)
        ctrl.get_left_panel_width.return_value = 300
        ctrl.get_tab_adapter_name.return_value = "fake"
        ctrl.get_adapter_guide.return_value = guide
        ctrl.progress_control.attach_progress.return_value = lambda: None
        ctrl.progress_control.progress_bars.return_value = []
        tab = ExpTabWidget("tab-1", ctrl, AdapterCapabilities())
        owned_tabs.append(tab)
        ctrl.get_adapter_guide.assert_called_once_with("fake")
        left_tabs = next(
            widget
            for widget in tab.findChildren(QTabWidget)
            if "Guide" in [widget.tabText(i) for i in range(widget.count())]
        )
        guide_index = [left_tabs.tabText(i) for i in range(left_tabs.count())].index(
            "Guide"
        )
        left_tabs.setCurrentIndex(guide_index)
        tab.resize(1000, 600)
        tab.show()
        qapp.processEvents()
        page = left_tabs.widget(guide_index)
        browser = page if isinstance(page, QTextBrowser) else None
        if page is not None and browser is None:
            browser = page.findChild(QTextBrowser)
        assert isinstance(browser, QTextBrowser)
        return browser

    try:
        yield create
    finally:
        for tab in owned_tabs:
            tab.hide()
            tab.deleteLater()
        qapp.processEvents()
        owned_authorities.clear()


def _image_formats(document: QTextDocument) -> list[QTextImageFormat]:
    formats: list[QTextImageFormat] = []
    block = document.begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            char_format = it.fragment().charFormat()
            if char_format.isImageFormat():
                formats.append(char_format.toImageFormat())
            it += 1
        block = block.next()
    return formats


def _anchor_hrefs(document: QTextDocument) -> list[str]:
    hrefs: list[str] = []
    block = document.begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            href = it.fragment().charFormat().anchorHref()
            if href and href not in hrefs:
                hrefs.append(href)
            it += 1
        block = block.next()
    return hrefs


def _document(browser: QTextBrowser) -> QTextDocument:
    document = browser.document()
    assert document is not None
    return document


def _loaded_image(browser: QTextBrowser, name: str) -> QImage | None:
    resource = _document(browser).resource(_IMAGE_RESOURCE, QUrl(name))
    if isinstance(resource, QImage) and not resource.isNull():
        return resource
    return None


def _click(browser: QTextBrowser, point: QPoint) -> None:
    viewport = browser.viewport()
    assert viewport is not None
    local = QPointF(point)
    global_point = QPointF(viewport.mapToGlobal(point))
    for event_type, buttons in (
        (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        event = QMouseEvent(
            event_type,
            local,
            global_point,
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, event)


def _type_key(browser: QTextBrowser) -> None:
    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "a"
    )
    QApplication.sendEvent(browser, event)


def _write_png(path: Path, width: int, height: int) -> None:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.darkCyan)
    assert image.save(str(path), "PNG")


def test_guide_shows_github_markdown_read_only_with_html_as_text(open_guide) -> None:
    markdown = (
        "# Overview\n\n"
        "Some **bold** words and <b>raw tag</b>.\n\n"
        "| knob | meaning |\n|---|---|\n| gain | readout amplitude |\n"
    )
    browser = open_guide(AdapterGuide(markdown=markdown, resource_dir=None))

    text = browser.toPlainText()
    assert "Overview" in text
    assert "<b>raw tag</b>" in text
    assert "gain" in text
    assert "<table" in _document(browser).toHtml()
    assert "|---|" not in text
    assert browser.isReadOnly()
    browser.setFocus()
    _type_key(browser)
    assert browser.toPlainText() == text


def test_clicked_link_opens_in_system_browser_without_navigating(
    open_guide, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    def record_open(url: QUrl) -> bool:
        opened.append(url.toString())
        return True

    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(record_open))
    browser = open_guide(
        AdapterGuide(
            markdown="Read [the manual](https://example.org/manual) first.",
            resource_dir=None,
        )
    )
    text = browser.toPlainText()
    found = _document(browser).find("manual")
    cursor = browser.textCursor()
    cursor.setPosition(found.selectionStart() + 2)
    point: QPoint = browser.cursorRect(cursor).center()
    assert browser.anchorAt(point) == "https://example.org/manual"

    _click(browser, point)
    QApplication.processEvents()

    assert opened == ["https://example.org/manual"]
    assert browser.toPlainText() == text


def test_link_destinations_keep_dollars_as_written(open_guide) -> None:
    markdown = "Read [manual](https://example.org/$x$) or <https://example.org/$y$>."
    browser = open_guide(AdapterGuide(markdown=markdown, resource_dir=None))

    assert _anchor_hrefs(_document(browser)) == [
        "https://example.org/$x$",
        "https://example.org/$y$",
    ]
    assert "Read manual or https://example.org/$y$." in browser.toPlainText()
    assert _image_formats(_document(browser)) == []


@pytest.mark.parametrize(
    ("markdown", "code_line"),
    [
        (
            "> Quoted:\n>\n> ```text\n> keep $x$ and $$y$$\n> ```\n",
            "keep $x$ and $$y$$",
        ),
        ("Indented:\n\n    cost = $x$ + $$y$$\n", "cost = $x$ + $$y$$"),
    ],
    ids=["blockquote-fence", "indented-code"],
)
def test_code_blocks_keep_dollars_as_written(
    open_guide, markdown: str, code_line: str
) -> None:
    browser = open_guide(AdapterGuide(markdown=markdown, resource_dir=None))

    assert code_line in browser.toPlainText()
    assert _image_formats(_document(browser)) == []


def test_inline_and_display_math_render_as_images(open_guide) -> None:
    markdown = "Energy $E=\\hbar\\omega$ per photon.\n\n$$\\frac{a}{b}$$\n\nAfter."
    browser = open_guide(AdapterGuide(markdown=markdown, resource_dir=None))

    text = browser.toPlainText()
    assert "Energy" in text
    assert "per photon." in text
    assert "After." in text
    assert "$" not in text
    assert "hbar" not in text
    assert "frac" not in text
    formats = _image_formats(_document(browser))
    assert len(formats) == 2
    for image_format in formats:
        image = _loaded_image(browser, image_format.name())
        assert image is not None
        assert image.width() > 0


@pytest.mark.parametrize(
    "markdown",
    [
        "Before $ x$ after.",
        "Before $x $ after.",
        "1$x$",
        "$x$2",
        "Before $$y$$ after.",
    ],
    ids=[
        "leading-space",
        "trailing-space",
        "preceding-digit",
        "following-digit",
        "inline-double",
    ],
)
def test_dollar_math_renders_with_library_boundaries(open_guide, markdown: str) -> None:
    browser = open_guide(AdapterGuide(markdown=markdown, resource_dir=None))

    formats = _image_formats(_document(browser))
    assert len(formats) == 1
    image = _loaded_image(browser, formats[0].name())
    assert image is not None
    assert image.width() > 0
    assert "$" not in browser.toPlainText()


def test_unsupported_math_and_escaped_dollars_stay_literal(open_guide) -> None:
    markdown = (
        "Costs \\$5 or \\$10 per run.\n\n"
        "Broken $\\notacommand{x}$ stays.\n\n"
        "Code `$z$` stays."
    )
    browser = open_guide(AdapterGuide(markdown=markdown, resource_dir=None))

    text = browser.toPlainText()
    assert "Costs $5 or $10 per run." in text
    assert "Broken $\\notacommand{x}$ stays." in text
    assert "Code $z$ stays." in text
    assert _image_formats(_document(browser)) == []


def test_local_images_load_and_wide_images_fit_the_tab(
    open_guide, tmp_path: Path
) -> None:
    guide_dir = tmp_path / "exp"
    guide_dir.mkdir()
    _write_png(guide_dir / "narrow.png", 40, 20)
    _write_png(guide_dir / "wide.png", 3000, 300)
    _write_png(tmp_path / "outside.png", 40, 20)
    (guide_dir / "diagram.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>',
        encoding="utf-8",
    )
    markdown = (
        "![narrow](narrow.png)\n\n"
        "![wide](wide.png)\n\n"
        "![missing](missing.png)\n\n"
        "![outside](../outside.png)\n\n"
        "![svg](diagram.svg)\n\n"
        "![remote](https://example.org/pulse.png)\n\n"
        "Text after images."
    )
    browser = open_guide(AdapterGuide(markdown=markdown, resource_dir=guide_dir))

    assert "Text after images." in browser.toPlainText()
    formats = {fmt.name(): fmt for fmt in _image_formats(_document(browser))}
    narrow = _loaded_image(browser, "narrow.png")
    wide = _loaded_image(browser, "wide.png")
    assert narrow is not None
    assert (narrow.width(), narrow.height()) == (40, 20)
    assert wide is not None
    assert formats["narrow.png"].width() == pytest.approx(40)
    available = browser.viewport().width()
    assert 0 < formats["wide.png"].width() <= available
    assert formats["wide.png"].height() == pytest.approx(
        formats["wide.png"].width() / 10, abs=1
    )
    for rejected in (
        "missing.png",
        "../outside.png",
        "diagram.svg",
        "https://example.org/pulse.png",
    ):
        assert _loaded_image(browser, rejected) is None


def test_local_image_with_empty_alt_loads(open_guide, tmp_path: Path) -> None:
    _write_png(tmp_path / "pulse.png", 40, 20)
    browser = open_guide(
        AdapterGuide(markdown="![](pulse.png)\n\nAfter.", resource_dir=tmp_path)
    )

    assert [fmt.name() for fmt in _image_formats(_document(browser))] == ["pulse.png"]
    assert _loaded_image(browser, "pulse.png") is not None


def test_guide_without_resource_dir_does_not_resolve_images(
    open_guide, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_png(tmp_path / "pulse.png", 40, 20)
    monkeypatch.chdir(tmp_path)
    browser = open_guide(
        AdapterGuide(
            markdown="(no guide written yet)\n\n![pulse](pulse.png)",
            resource_dir=None,
        )
    )

    assert "(no guide written yet)" in browser.toPlainText()
    assert _loaded_image(browser, "pulse.png") is None
