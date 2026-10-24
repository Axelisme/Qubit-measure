"""Shared real measure UI composition and teardown for public view contracts."""

from collections.abc import Callable, Generator, Iterator
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QApplication, QWidget
from zcu_tools.datafile import Container, ContainerAttrs, JsonObject, save
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.results_index import scan_results
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.resources.entry import AcquiredPayload, LedgerEvent, ResultEntry

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    native_save_session,
)
from tests.gui.app.measure._record_support import require_record
from tests.gui.app.measure._workflow_window import (
    workflow_window as make_workflow_window,
)
from tests.gui.app.measure.ui._main_window_fakes import apply_window_defaults, mock_ctrl


@pytest.fixture
def browser_host(
    qapp: QApplication,
    tmp_path: Path,
    main_window_control_factory: Callable[[], ControlIdentity],
) -> Iterator[MainWindow]:
    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.query_results.side_effect = lambda: scan_results(tmp_path)
    ctrl.resolve_experiment.side_effect = Registry().resolve_experiment
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    window.show()
    qapp.processEvents()
    yield window
    window.deleteLater()
    qapp.processEvents()


@pytest.fixture
def saved_cfg() -> JsonObject:
    return {
        "values": {
            "readout": {"frequency": 5100.0, "tone": "Q2"},
            "deep": {"level": {"branch": {"leaf": "bounded"}}},
            "long": "x" * 240,
        },
        "cfg_type": "synthetic.history.ReadoutCfg",
        "schema_version": "7.3",
    }


@pytest.fixture
def populated_browser(
    browser_host: MainWindow, tmp_path: Path, saved_cfg: JsonObject
) -> ComponentDriver:
    for entry_name, point, run_id, experiment, day in (
        ("alpha", "p1", "20261021T010000Z-aaaaaa", "mux/readout", "21"),
        ("alpha", "p2", "20261022T010000Z-bbbbbb", "mux/readout", "22"),
        ("beta", "p1", "20261021T010000Z-aaaaaa", "mux/readout", "23"),
    ):
        entry_path = tmp_path / "result" / entry_name
        entry = (
            ResultEntry.open(
                entry_name,
                result_root=tmp_path / "result",
                database_root=tmp_path / "Database",
            )
            if entry_path.exists()
            else ResultEntry.create(
                entry_name,
                result_root=tmp_path / "result",
                database_root=tmp_path / "Database",
            )
        )
        run = tmp_path / "Database" / entry_name / "runs" / run_id
        (run / "figures").mkdir(parents=True)
        metadata = RunMetadata(
            run_id=run_id,
            experiment=experiment,
            started_at=f"2026-10-{day}T01:00:00Z",
            finished_at=None,
            completion="unknown",
            snapshot=RunSnapshot(
                entry_id=UUID(entry.entry_id),
                entry_name=entry_name,
                point=point,
                description=None,
                roles={"qubit": "Q1"},
            ),
            provenance=SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname=None,
            ),
        )
        save(
            run / "data.h5",
            Container(
                vars={},
                docs={
                    "metadata": metadata.model_dump(mode="json"),
                    "cfg": saved_cfg,
                },
                groups={},
                attrs=ContainerAttrs("3.0", "synthetic-browser", None, {}),
            ),
        )
        (run / "analysis").mkdir()
        (run / "analysis" / "analysis.json").write_text(
            '{"format":"zcu.analysis-result","format_version":"1.0",'
            '"pane":"analysis","parameters":{"n_iter":12},'
            '"summary":{"tones":[{"name":"Q2","freq":5100.0}]}}',
            encoding="utf-8",
        )
        for i in range(8 if point == "p1" else 1):
            image = QImage(400, 220, QImage.Format.Format_RGB32)
            image.fill(Qt.GlobalColor.cyan if i % 2 else Qt.GlobalColor.yellow)
            assert image.save(str(run / "figures" / f"plot-{i}.png"))
        if point == "p1":
            entry.ledger.append(
                LedgerEvent(
                    id=str(uuid4()),
                    entry_id=entry.entry_id,
                    kind="acquired",
                    at="2026-10-21T01:05:00Z",
                    origin="notebook",
                    payload=AcquiredPayload(
                        run_id=run_id,
                        tab="Synthetic",
                        experiment=experiment,
                        point=point,
                        roles={"qubit": "Q1"},
                    ),
                )
            )
    ui = ComponentDriver(browser_host)
    ui.select_menu("More ▾", ("Results browser…",))
    window = browser_host.findChild(QWidget, "results.browser")
    return ComponentDriver(window)


@pytest.fixture
def workflow_window(
    qapp: QApplication, tmp_path: Path
) -> Iterator[tuple[Controller, MainWindow]]:
    yield from make_workflow_window(qapp, tmp_path)


@pytest.fixture
def main_window_control_factory(
    qapp: QApplication,
) -> Iterator[Callable[[], ControlIdentity]]:
    """Keep fake application authority alive until its Qt view tree is deleted."""
    baseline = set(qapp.topLevelWidgets())
    authorities: list[ControlIdentity] = []

    def create() -> ControlIdentity:
        authority = ControlIdentity()
        authorities.append(authority)
        return authority

    yield create
    for widget in qapp.topLevelWidgets():
        if widget not in baseline:
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    authorities.clear()


@pytest.fixture
def live_registry() -> Registry:
    """Inject the synthetic adapter catalog used by real Controller UI scenarios."""
    registry = Registry()
    registry.register("saving", AnalysisSavingAdapter)
    return registry


@pytest.fixture
def live_ui(
    tmp_path: Path,
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    live_registry: Registry,
) -> Generator[tuple[NativeSaveSession, Controller, MainWindow, str]]:
    """Open a real tab and view over the existing offline native Save fixture."""
    with native_save_session(tmp_path, qapp) as session:
        monkeypatch.setattr(
            "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
            lambda: session.background,
        )
        ctrl = Controller(
            session.state,
            live_registry,
            None,
            session.bus,
            progress_transport=DirectProgressTransport(),
            project_root=str(tmp_path),
        )
        tab_id = ctrl.new_tab("saving")
        original = session.state.get_tab("tab")
        session.state.update_tab_result(
            tab_id,
            replace(require_record(original.run), metadata=session.metadata),
            parameters=original.run.parameters,
        )
        window = MainWindow(
            ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
        )
        window.add_tab_widget(tab_id, "saving")
        try:
            yield session, ctrl, window, tab_id
        finally:
            assert session.background.quiesce()
            ctrl.tab_control.close_tab(tab_id)
            ctrl.close()
            window.deleteLater()
            qapp.processEvents()
