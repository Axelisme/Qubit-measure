"""Shared real measure UI composition and teardown for public view contracts."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.gui.app.measure.adapter import WaveformWriteback
from zcu_tools.gui.app.measure.cfg_schemas import waveform_cfg_to_value
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.app.measure.services.writeback import WritebackService
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv as ParameterSessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    native_save_session,
)
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@pytest.fixture
def live_ui(
    tmp_path: Path, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> Generator[tuple[NativeSaveSession, Controller, MainWindow, str]]:
    """Open a real tab and view over the existing offline native Save fixture."""
    with native_save_session(tmp_path, qapp) as session:
        registry = Registry()
        registry.register("saving", AnalysisSavingAdapter)
        monkeypatch.setattr(
            "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
            lambda: session.background,
        )
        ctrl = Controller(
            session.state,
            registry,
            None,
            session.bus,
            progress_transport=DirectProgressTransport(),
            project_root=str(tmp_path),
        )
        tab_id = ctrl.new_tab("saving")
        original = session.state.get_tab("tab")
        session.state.update_tab_result(
            tab_id,
            original.run.result,
            metadata=session.metadata,
            parameters=original.run.parameters,
        )
        window = MainWindow(ctrl, dialog_presenter=RecordingDialogPresenter())
        window.add_tab_widget(tab_id, "saving")
        try:
            yield session, ctrl, window, tab_id
        finally:
            assert session.background.quiesce()
            ctrl.tab_control.close_tab(tab_id)
            window.deleteLater()
            qapp.processEvents()


@pytest.fixture
def waveform_writeback(tmp_path: Path):
    before = registry_state()
    draft = None
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "Result",
            database_root=tmp_path / "Database",
        )
        bus = BaseEventBus()
        context = ContextService(
            SessionState(ParameterSessionEnv(parameters=None, soc=None, soccfg=None)),
            bus,
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        control = ContextControlFacet(context=context)
        entry.new_point("working")
        control.use_point(PointUseRequest("working"))
        control.add_component("Q1", "fake/drive/a", {})
        target = control.read_context().target
        control.set_waveform(target, "Q1.waveforms.base", ConstWaveformCfg(length=0.1))
        editor = CfgEditorService(MagicMock(), control, MagicMock(), MagicMock(), bus)
        service = WritebackService(editor, control)
        spec, value = waveform_cfg_to_value(ConstWaveformCfg(length=0.1))
        draft = service.create_draft(
            [
                WaveformWriteback(
                    "Q1.waveforms.base", "Waveform", edit_schema=CfgSchema(spec, value)
                )
            ],
            parameters=control.read_context(),
            experiment_name="test/writeback",
        )
        service.apply_draft(draft)
        yield control, service, draft
        assert registry_state() == installed
    finally:
        if draft is not None:
            draft.teardown()
        restore_registry(before)
