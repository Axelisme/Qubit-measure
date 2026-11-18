"""Drive the real retained Results Browser with causally released native reads."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from threading import Event, get_ident
from typing import Never

import h5py
import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QApplication, QWidget
from zcu_tools.datafile import load, save
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.caretaker import create_persistence_caretaker
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import EntryRequest
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    native_save_session,
)


def _refuse_driver(type_name: str, address: str) -> Never:
    raise AssertionError(f"offline results scenario cannot construct {type_name}")


class _CloseReleaseController(Controller):
    """Let a held I/O boundary proceed inside the public drain's owner loop."""

    on_drain: Callable[[], None] | None = None

    def close(self, timeout_ms: int = 5000) -> None:
        callback, self.on_drain = self.on_drain, None
        if callback is not None:
            QTimer.singleShot(0, callback)
        super().close(timeout_ms)


class _BrowserScenario:
    def __init__(
        self, window: MainWindow, controller: _CloseReleaseController, image_dir: Path
    ) -> None:
        self._controller = controller
        self.main = ComponentDriver(window)
        self._window = window
        self._image_dir = image_dir

    def browser(self) -> ComponentDriver:
        # Fixture mount: specialize the driver's root to the owned top-level window.
        browser = self._window.findChild(QWidget, "results.browser")
        return ComponentDriver(browser)

    def capture(self, name: str) -> None:
        self.main.screenshot(self._image_dir / f"{name}.png")

    def release_on_drain(self, gate: _ReadGate) -> None:
        def release() -> None:
            try:
                self.capture("02-main-close-draining")
            finally:
                gate.release.set()

        self._controller.on_drain = release


def _make_saved_rows(tmp_path: Path, qapp: QApplication) -> tuple[Path, Path]:
    with native_save_session(tmp_path, qapp) as session:
        session.save()
        first = session.native_data_path
    content = load(first)
    later = tmp_path / "Database/zeta/runs"
    second = later / "one" / "data.h5"
    other = later / "two" / "data.h5"
    for path in (second, other):
        path.parent.mkdir(parents=True)
        metadata = dict(content.docs["metadata"])
        if path == other:
            metadata["experiment"] = "aaa"
        save(path, replace(content, docs=dict(content.docs, metadata=metadata)))
    for data in (first, second, other):
        analysis = data.parent / "analysis"
        analysis.mkdir()
        (analysis / "summary.json").write_text('{"summary":{"label":"Needle"}}')
    figures = first.parent / "figures"
    figures.mkdir()
    image = QImage(320, 176, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.cyan)
    assert image.save(str(figures / "first.png"))
    return first, second


class _ReadGate:
    def __init__(self) -> None:
        self.arrived = Event()
        self.release = Event()

    def hold(self) -> None:
        self.arrived.set()
        if not self.release.wait(timeout=25):
            raise TimeoutError("test never released an in-flight native read")


@pytest.fixture
def progress_browser(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[_BrowserScenario, _ReadGate, _ReadGate]]:
    first, second = _make_saved_rows(tmp_path, qapp)
    first_gate, second_gate = _ReadGate(), _ReadGate()
    gates = {first: first_gate, second: second_gate}
    open_file = h5py.File
    owner_thread = get_ident()

    def gated_read(path: Path, mode: str) -> h5py.File:
        # In the old synchronous implementation, do not block the test thread;
        # the visible zero-row assertion will show that Indexing is absent.
        if (gate := gates.get(path)) is not None and get_ident() != owner_thread:
            gate.hold()
        return open_file(path, mode)

    monkeypatch.setattr(h5py, "File", gated_read)
    registry = Registry()
    registry.register("saving", AnalysisSavingAdapter)
    ctrl = _CloseReleaseController(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
        progress_transport=DirectProgressTransport(),
        driver_factory=_refuse_driver,
    )
    ctrl.entry_control.open_entry(EntryRequest("original"))
    ctrl.context_control.use_point("p")
    ctrl.attach_caretaker(
        create_persistence_caretaker(ctrl, cache_dir=tmp_path / "cache")
    )
    window = MainWindow(ctrl, templates=lambda: {})
    ctrl.add_view(window)
    window.show()
    qapp.processEvents()
    try:
        yield _BrowserScenario(window, ctrl, tmp_path), first_gate, second_gate
    finally:
        first_gate.release.set()
        second_gate.release.set()
        ctrl.close()
        window.dispose()
        qapp.processEvents()


def test_browse_open_and_reopen_while_indexing(
    progress_browser: tuple[_BrowserScenario, _ReadGate, _ReadGate],
) -> None:
    scenario, first, second = progress_browser
    main = scenario.main
    main.click("Results browser…")
    browser = scenario.browser()
    assert browser.observe(object_name="results.browser").visible
    assert browser.observe(object_name="results.visible_count").text == "0 / 0 runs"
    assert browser.status().text == "Indexing results…"
    qt_until(first.arrived.is_set, label="first native read is held")
    scenario.capture("01-indexing-empty")

    browser.close_window()
    assert not browser.observe(object_name="results.browser").visible
    scenario.capture("02-hidden-before-first-entry")
    first.release.set()
    qt_until(second.arrived.is_set, label="work reached the second entry while hidden")
    main.click("Results browser…")
    qt_until(
        lambda: browser.status().text == "Indexing results: 1 / 2 entries · 1 runs",
        label="first batch visible while second is held",
    )
    assert browser.observe(object_name="results.visible_count").text == "1 / 1 runs"
    scenario.capture("03-first-entry-after-hidden")

    browser.enter_text(value="needle", object_name="results.search")
    browser.select_combo("test/saving", object_name="results.experiment")
    browser.invoke_context_action(
        action="Add to comparison", object_name="results.figure.0"
    )
    assert browser.tabs("Comparison (1)").current == "Comparison (1)"
    scenario.capture("04-filtered-and-pinned")
    browser.select_tab("Gallery")
    browser.click(object_name="results.figure.0")
    browser.click("Load into analysis tab")
    assert browser.observe(object_name="results.feedback").text.startswith(
        "Loaded history:"
    )
    assert main.tabs("saving").current == main.tabs("saving").captions[-1]
    scenario.capture("05-open-during-scan")
    browser.click("All results")

    browser.close_window()
    assert not browser.observe(object_name="results.browser").visible
    scenario.capture("06-browser-hidden")
    main.click("Results browser…")
    assert browser.observe(object_name="results.browser").visible
    assert browser.status().text == "Indexing results: 1 / 2 entries · 1 runs"
    assert browser.observe(object_name="results.search").text == "needle"
    assert browser.observe(object_name="results.experiment").text == "test/saving"
    assert "Comparison (1)" in browser.tabs("Gallery").captions
    scenario.capture("07-browser-reopened")

    second.release.set()
    qt_until(
        lambda: browser.status().text == "Index ready: 3 runs · 0 issues",
        label="index completed in retained view",
    )
    assert browser.observe(object_name="results.visible_count").text == "2 / 3 runs"
    assert browser.observe(object_name="results.experiment").text == "test/saving"
    assert "Comparison (1)" in browser.tabs("Gallery").captions
    scenario.capture("08-second-entry-filtered")
    browser.press_key(Qt.Key.Key_Up, object_name="results.experiment")
    assert browser.observe(object_name="results.experiment").text == "aaa"
    assert browser.observe(object_name="results.visible_count").text == "1 / 3 runs"
    scenario.capture("09-new-filter-option")


def test_main_window_closes_without_confirmation_while_indexing(
    progress_browser: tuple[_BrowserScenario, _ReadGate, _ReadGate],
) -> None:
    scenario, first, _second = progress_browser
    main = scenario.main
    main.click("Results browser…")
    browser = scenario.browser()
    qt_until(first.arrived.is_set, label="native read held before application close")
    assert browser.status().text == "Indexing results…"
    scenario.capture("01-main-close-active-index")

    scenario.release_on_drain(first)
    main.close_window()
    qt_until(
        lambda: (
            not main.observe(object_name="empty_tabs_hint").visible or main.dialogs()
        ),
        label="main window drained or requested confirmation",
    )
    assert not main.dialogs()
    assert not main.observe(object_name="empty_tabs_hint").visible
    assert not browser.observe(object_name="results.browser").visible
