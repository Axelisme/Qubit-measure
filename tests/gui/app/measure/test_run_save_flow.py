"""User-component Run and Save through the real measure-gui composition root."""

from __future__ import annotations

import logging
from collections.abc import Generator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QWidget
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.resources.entry import ComponentSchema

from tests.experiment._saving_support import SavingCfg, SavingResult
from tests.gui import _user_input
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._app_launch import launched_measure_app
from tests.gui.app.measure._native_save_support import (
    PaneParams,
    PaneSummary,
    SavingAdapter,
)
from tests.resources.entry.fakes import registry_state
from tests.resources.entry.view_support import registered_model


class CalibrationSchema(ComponentSchema):
    """Two independently addressable values for selective writeback in Inspect."""

    first: float = 1.0
    second: float = 2.0


def PlottedSavingAdapter_analyze(
    req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams], *, plots: Plots
) -> PaneSummary:
    _, axes = plots.subplots("fit")
    axes.plot(
        req.run_result.result.frequency,
        req.run_result.result.signal.real,
        label="Synthetic fit",
    )
    axes.legend()
    return PaneSummary()


def PlottedSavingAdapter_get_writeback_items(
    req: WritebackRequest[RunRecord[SavingCfg, SavingResult], PaneSummary],
) -> Sequence[WritebackItem]:
    return [
        ParameterWriteback("calibration.first", "First", 11.0),
        ParameterWriteback("calibration.second", "Second", 22.0),
    ]


def PlottedSavingAdapter_run(
    cfg: SavingCfg, *, context: RunContext
) -> RunRecord[SavingCfg, SavingResult]:
    record = SavingAdapter.run(cfg, context=context)
    context.plots.subplots("signal")

    def draw(figure: Figure) -> None:
        axes = figure.axes[0]
        axes.plot(record.result.frequency, record.result.signal.real, label="Signal")
        axes.legend()

    context.plots.draw("signal", draw)
    return record


PlottedSavingAdapter = Adapter(
    spec=SavingAdapter.spec,
    make_form=SavingAdapter.make_form,
    run=PlottedSavingAdapter_run,
    guide=None,
    analysis=Analysis(
        fn=PlottedSavingAdapter_analyze,
        options=lambda source, env: PaneParams(),
        writeback=PlottedSavingAdapter_get_writeback_items,
    ),
    post_analysis=None,
)


@pytest.fixture
def measure_window(qapp, monkeypatch: pytest.MonkeyPatch) -> Generator[QWidget]:
    registry = Registry()
    registry.register("synthetic", PlottedSavingAdapter)
    before = registry_state()
    with (
        TemporaryDirectory(prefix="measure-chooser-", dir=Path.cwd()) as directory,
        monkeypatch.context() as context,
        registered_model("synthetic/calibration", CalibrationSchema),
        launched_measure_app(qapp, Path(directory), registry=registry) as app,
    ):
        installed = registry_state()
        # Fixture bootstrap only: create an isolated entry/point before opening a tab.
        app.controller.setup_control.new_entry(EntryRequest("component-flow"))
        app.controller.setup_control.new_point(
            PointCreateRequest("p", "Synthetic component flow", CommandOrigin("gui"))
        )
        app.controller.context_control.add_component(
            "calibration", "synthetic/calibration", {}
        )
        app.controller.install_soc(MagicMock(), make_mock_soccfg())
        app.controller.new_tab("synthetic")
        # Load's File chooser starts at the opened entry's database directory.
        # Keep both relative-path validation and real chooser input rooted there.
        chooser_directory = Path(app.controller.get_session_env().database_path)
        chooser_directory.mkdir(parents=True, exist_ok=True)
        context.chdir(chooser_directory)
        yield app.window
        assert registry_state() == installed
    assert registry_state() == before


def test_run_curve_then_save_all_reports_native_save_in_bottom_status(
    measure_window: QWidget, tmp_path: Path
) -> None:
    flow = _user_input.ComponentDriver(measure_window)
    shots = tmp_path / "run-save"
    flow.screenshot(shots / "01-setup.png")
    assert flow.dialogs()[0].title == "Setup"
    flow.click("Close")
    flow.screenshot(shots / "02-ready.png")
    assert flow.observe("Run").enabled
    assert flow.legends().texts == ()
    flow.click("Run")
    flow.screenshot(shots / "03-run-requested.png")
    # The offline adapter returns after its host draw; terminal owner delivery restores Run.
    # qt_until's 30-second watchdog only guards a stuck worker/Qt delivery.
    qt_until(
        lambda: (
            flow.legends().texts == ("Signal",)
            and any(
                w.kind == "QPushButton" and w.text == "Run" and w.enabled and w.visible
                for w in flow.visible().widgets
            )
        ),
        label="synthetic Run result and idle button",
    )
    flow.screenshot(shots / "04-run-complete.png")
    assert flow.observe("Run").text == "Run"
    assert flow.legends().texts == ("Signal",)
    flow.select_tab("Data")
    flow.screenshot(shots / "05-data.png")
    assert flow.tabs("Data").current == "Data"
    assert flow.observe("Save All").enabled
    flow.click("Save All")
    flow.screenshot(shots / "06-saved.png")
    assert flow.status().text.startswith("Saved ")
    assert " for run " in flow.status().text
    assert "failed" not in flow.status().text
    assert flow.observe("✓ SAVED").visible
    assert flow.dialogs() == ()


def test_disabled_save_all_click_keeps_no_result_and_no_save_message(
    measure_window: QWidget, tmp_path: Path
) -> None:
    flow = _user_input.ComponentDriver(measure_window)
    shots = tmp_path / "disabled-save"
    flow.screenshot(shots / "01-setup.png")
    flow.click("Close")
    flow.screenshot(shots / "02-ready.png")
    flow.select_tab("Data")
    flow.screenshot(shots / "03-disabled-save.png")
    before = flow.status()
    button = flow.observe("Save All")
    assert button.visible and not button.enabled
    flow.click("Save All")
    flow.screenshot(shots / "04-after-disabled-click.png")
    assert flow.status() == before
    assert not flow.observe("Save All").enabled
    assert flow.dialogs() == ()
    flow.select_tab("Run")
    flow.screenshot(shots / "05-still-no-result.png")
    assert flow.legends().texts == ()
    assert flow.observe("Run").enabled


def test_invalid_cfg_enter_explains_run_block_and_repair_clears_reason(
    measure_window: QWidget, tmp_path: Path
) -> None:
    flow = _user_input.ComponentDriver(measure_window)
    shots = tmp_path / "invalid-cfg"
    flow.screenshot(shots / "01-setup.png")
    flow.click("Close")
    flow.screenshot(shots / "02-ready.png")
    assert flow.observe("Run").enabled
    flow.enter_text(object_name="form.input.gain", value="not-a-number")
    flow.screenshot(shots / "03-invalid.png")
    assert not flow.observe("Run").enabled
    reason = flow.status().text
    assert "gain" in reason.lower()
    assert reason.strip()
    flow.enter_text(object_name="form.input.gain", value="2.0")
    flow.screenshot(shots / "04-repaired.png")
    assert flow.observe("Run").enabled
    assert flow.status().text != reason
    assert "gain" not in flow.status().text.lower()
    assert flow.dialogs() == ()


def _run_and_save_signal(flow: _user_input.ComponentDriver, shots: Path) -> str:
    """Run the offline signal and return the native path shown in Save's tooltip."""
    flow.click("Run")
    flow.screenshot(shots / "03-run-requested.png")
    # The offline worker draws Signal before its terminal delivery restores Run.
    qt_until(
        lambda: (
            flow.legends().texts == ("Signal",)
            and any(w.text == "Run" and w.enabled for w in flow.visible().widgets)
        ),
        label="synthetic run before native save",
    )
    flow.screenshot(shots / "04-run-complete.png")
    flow.select_tab("Data")
    flow.screenshot(shots / "05-data.png")
    flow.click("Save All")
    flow.screenshot(shots / "06-saved.png")
    assert flow.status().text.startswith("Saved ")
    paths = [
        w.tooltip for w in flow.visible().widgets if w.tooltip.endswith("/data.h5")
    ]
    assert len(paths) == 1
    return paths[0]


def _load_saved_signal(
    flow: _user_input.ComponentDriver,
    shots: Path,
    native_path: str,
) -> None:
    """Choose the saved file through Load's modal and the external file chooser."""
    dialog_errors: list[Exception] = []

    def dismiss_modals() -> None:
        # Failure cleanup only: release nested Qt loops even if a caption changed.
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QDialog) and widget.isVisible():
                widget.reject()

    def timed_out() -> None:
        dialog_errors.append(TimeoutError("Load dialog input delivery did not finish"))
        dismiss_modals()

    def choose_native_file() -> None:
        try:
            assert flow.dialogs()[0].title == "Load Data"
            flow.screenshot(shots / "07-load-dialog.png")
            flow.choose_file(
                "File…",
                Path(native_path).relative_to(Path.cwd(), walk_up=True),
                evidence=shots / "07-file-chooser.png",
            )
        except Exception as error:
            logging.exception("Load dialog input delivery failed")
            dialog_errors.append(error)
            dismiss_modals()

    # Clicking Load synchronously enters its modal loop; that loop must deliver
    # the queued File callback. The 30-second timer guards a broken delivery only.
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(timed_out)
    delivery = QTimer()
    delivery.setSingleShot(True)
    delivery.timeout.connect(choose_native_file)
    watchdog.start(30_000)
    delivery.start(0)
    try:
        flow.click("Load Data")
    except Exception:
        dismiss_modals()
        raise
    finally:
        delivery.stop()
        watchdog.stop()
    if dialog_errors:
        raise dialog_errors[0]
    flow.screenshot(shots / "08-loaded.png")
    assert flow.status().text == f"Loaded data from {native_path}"
    assert flow.dialogs() == ()


def test_load_then_analyze_and_apply_only_selected_proposal(
    measure_window: QWidget, tmp_path: Path
) -> None:
    flow = _user_input.ComponentDriver(measure_window)
    shots = tmp_path / "load-analyze-apply"
    flow.screenshot(shots / "01-setup.png")
    flow.click("Close")
    flow.screenshot(shots / "02-ready.png")
    native_path = _run_and_save_signal(flow, shots)
    _load_saved_signal(flow, shots, native_path)
    flow.select_tab("Analysis")
    flow.screenshot(shots / "09-analysis-ready.png")
    assert flow.observe("Analyze").enabled
    assert flow.legends().texts == ()
    flow.click("Analyze")
    flow.screenshot(shots / "10-analyze-requested.png")
    # This deterministic analysis returns its named fit; terminal delivery mounts
    # that canvas and publishes the proposal ledger on the owner thread.
    qt_until(
        lambda: (
            flow.legends().texts == ("Synthetic fit",)
            and any(w.text == "calibration.first*" for w in flow.visible().widgets)
        ),
        label="loaded result fit and writeback preview",
    )
    flow.screenshot(shots / "11-analyzed.png")
    assert flow.legends().texts == ("Synthetic fit",)
    assert flow.observe("calibration.first*").checked is True
    assert flow.observe("calibration.second*").checked is True
    flow.click("calibration.second*")
    flow.screenshot(shots / "12-second-unselected.png")
    assert flow.observe("calibration.second*").checked is False
    assert flow.observe("Apply Selected").enabled
    flow.click("Apply Selected")
    flow.screenshot(shots / "13-applied.png")
    assert any(
        w.kind == "QCheckBox" and w.text == "calibration.first" and w.checked is True
        for w in flow.visible().widgets
    )
    assert flow.observe("calibration.second*").visible
    assert flow.observe("* = not applied").visible
    flow.click("Inspect…")
    flow.screenshot(shots / "14-inspect.png")
    assert flow.dialogs()[0].title == "Inspect Context"
    flow.expand_tree(("calibration",), object_name="component_tree")
    flow.screenshot(shots / "15-inspect-values.png")
    values = {
        row.path: row.columns[1]
        for row in flow.visible().tree_rows
        if row.tree_name == "component_tree"
    }
    assert values[("calibration", "first")] == "11.0"
    assert values[("calibration", "second")] == "2.0"
    flow.click("Close")
    flow.screenshot(shots / "16-inspect-closed.png")
    assert flow.dialogs() == ()
