"""Measure composition releases its Qt tree before the application owner exits."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QWidget
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner

from tests._lab import synthetic_lab
from tests.gui.app.measure._reload_fakes import Loader


@pytest.fixture
def restored_exception_hooks():
    """Restore both process exception hooks even when construction or close fails."""
    previous_hook = sys.excepthook
    previous_thread_hook = threading.excepthook
    try:
        yield
    finally:
        sys.excepthook = previous_hook
        threading.excepthook = previous_thread_hook


def test_lab_composition_uses_injected_definitions(
    qapp, tmp_path, restored_exception_hooks
):
    from zcu_tools.cfg_model import PulseSpec, TemplateDef
    from zcu_tools.gui.app.measure.controller import Controller

    from tests.gui.app.measure._operation_plots import attachment
    from tests.gui.app.measure._synthetic import SampleAdapter

    registry = Registry()
    registry.register("injected", SampleAdapter)
    effects: list[str] = []

    def register_components(_registry):
        effects.append("components")

    template = TemplateDef(PulseSpec, "Injected pulse", {})
    lab = replace(
        synthetic_lab(registry),
        register_components=register_components,
        templates={"injected-pulse": template},
        workflow_attachments=(attachment(),),
    )
    behavior = None
    try:
        behavior = MeasureGuiBehavior(lab, clean=True, project_root=str(tmp_path))
        assembly = behavior.assemble(None)
        assert isinstance(assembly.controller, Controller)
        assert assembly.controller.get_adapter_names() == ["injected"]
        assert assembly.controller.get_template("injected-pulse") is template
        assert assembly.controller.workflow_control().names() == ("operation-plots",)
        assert effects == ["components"]
    finally:
        if behavior is not None:
            behavior.close(None)


def construct_with_legacy_keyword(
    constructor: Callable[..., object], keyword: str
) -> object:
    """Exercise invalid API calls without claiming they are statically valid."""

    def factory():
        return Registry(), Loader()

    if keyword == "registry_factory":
        return constructor(registry_factory=factory)
    return constructor(factory, configure_workflows=lambda _control: None)


@pytest.mark.parametrize("keyword", ["registry_factory", "configure_workflows"])
def test_measure_behavior_rejects_legacy_keywords(keyword, restored_exception_hooks):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        construct_with_legacy_keyword(MeasureGuiBehavior, keyword)


class ControlledDrain(BackgroundRunner):
    """Record the runner's public terminal seam without starting a worker."""

    ready = True

    def __init__(self) -> None:
        super().__init__()
        self.effects: list[str] = []

    def quiesce(self, timeout_ms: int = 5000) -> bool:
        self.effects.append("drain")
        return self.ready and super().quiesce(timeout_ms)


@pytest.fixture
def composition(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restored_exception_hooks,
):
    runner = ControlledDrain()
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
        lambda: runner,
    )
    behavior = MeasureGuiBehavior(
        synthetic_lab(), clean=True, project_root=str(tmp_path)
    )
    try:
        yield behavior, runner
    finally:
        runner.ready = True
        behavior.close(None)


def test_close_destroys_window_and_children_once(composition):
    behavior, runner = composition
    assembly = behavior.assemble(None)
    behavior.before_show(assembly)
    behavior.after_show(assembly)
    window = assembly.window
    child = QWidget(window)
    window.destroyed.connect(lambda: runner.effects.append("window"))
    child.destroyed.connect(lambda: runner.effects.append("child"))

    behavior.close(assembly)
    behavior.close(assembly)

    assert runner.effects.count("window") == 1
    assert runner.effects.count("child") == 1
    assert runner.effects.index("drain") < runner.effects.index("window")


def test_close_timeout_keeps_window_and_children_for_retry(composition):
    behavior, runner = composition
    assembly = behavior.assemble(None)
    behavior.before_show(assembly)
    behavior.after_show(assembly)
    child = QWidget(assembly.window)
    assembly.window.destroyed.connect(lambda: runner.effects.append("window"))
    child.destroyed.connect(lambda: runner.effects.append("child"))
    runner.ready = False

    with pytest.raises(TimeoutError):
        behavior.close(assembly)
    assert runner.effects == ["drain"]
    child.setWindowTitle("still usable")

    runner.ready = True
    behavior.close(assembly)
    assert runner.effects == ["drain", "drain", "window", "child"]


def test_remote_assembly_failure_disposes_retained_window(composition, monkeypatch):
    behavior, runner = composition

    def fail_adapter(*, render_view, **_kwargs):
        render_view.destroyed.connect(lambda: runner.effects.append("window"))
        raise RuntimeError("synthetic adapter assembly failure")

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.remote.RemoteControlAdapter", fail_adapter
    )
    with pytest.raises(RuntimeError, match="synthetic adapter assembly failure"):
        behavior.assemble(ControlOptions(port=0))

    assert runner.effects == ["drain", "window"]


def test_close_releases_named_dialog_subscriptions(composition, caplog):
    behavior, _runner = composition
    assembly = behavior.assemble(None)
    behavior.before_show(assembly)
    behavior.after_show(assembly)
    # Hidden named windows must finish too, rather than relying on closeEvent.
    assembly.window.open_dialog(DialogName.SOC)
    assembly.window.open_dialog(DialogName.DEVICE)
    assembly.window.open_dialog(DialogName.INSPECT)
    assembly.window.open_dialog(DialogName.PREDICTOR)
    finished: list[str] = []
    from qtpy.QtWidgets import QDialog

    dialogs = assembly.window.findChildren(QDialog)
    for dialog in dialogs:
        dialog.finished.connect(lambda _result: finished.append("finished"))
        dialog.hide()

    behavior.close(assembly)

    assert len(finished) == len(dialogs)
    assert "ERROR" not in caplog.text
