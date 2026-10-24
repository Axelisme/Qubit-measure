"""Visible adapter notes and manual selection in the writeback review component."""

from collections.abc import Generator, Sequence
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, WritebackRequest
from zcu_tools.gui.app.measure.writeback import (
    WritebackItem,
    WritebackService,
    WritebackWidget,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._synthetic import (
    SampleAdapter,
    SampleAnalysis,
    SampleCfg,
    SampleParams,
    SampleResult,
    SampleRun,
)


def ReviewAdapter_get_writeback_items(
    req: WritebackRequest[SampleRun, SampleAnalysis],
) -> Sequence[WritebackItem]:
    item = SampleAdapter.get_writeback_items(req)[0]
    item.selected = False
    item.note = "Fit needs manual review"
    return [item]


ReviewAdapter = Adapter(
    spec=SampleAdapter.spec,
    make_form=SampleAdapter.make_form,
    run=SampleAdapter.run,
    guide=None,
    analysis=Analysis(
        fn=SampleAdapter.analyze,
        options=lambda source, env: SampleParams(),
        writeback=ReviewAdapter_get_writeback_items,
    ),
    post_analysis=SampleAdapter.post_analysis,
)


@pytest.fixture
def review(qapp: QApplication, tmp_path: Path) -> Generator[ComponentDriver]:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    entry.new_point("a")
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    context.install_entry(entry)
    context.use_point(PointUseRequest("a"))
    service = WritebackService(
        ContextControlFacet(context=context), arb_names=lambda: ()
    )
    adapter = ReviewAdapter
    run = RunRecord(
        SampleCfg(sweep=SweepCfg(start=0.0, stop=1.0, expts=2, step=1.0)),
        SampleResult(np.array([1.0])),
    )
    items = adapter.get_writeback_items(
        WritebackRequest(
            run_result=run,
            analyze_result=SampleAnalysis(peak=5.0),
            ctx=context.get_session_env(),
        )
    )
    draft = service.create_draft(
        items, parameters=context.read_context(), experiment_name="test/review"
    )
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.side_effect = lambda *_: (
        service.get_all_applied(draft)
    )
    ctrl.set_writeback_item_for_pane.side_effect = lambda _tab, _pane, sid, **changes: (
        service.edit_draft(draft, sid, **changes)
    )
    widget = WritebackWidget(ctrl, tab_id="review", pane="analysis")

    def populate() -> None:
        widget.populate(service.preview_draft(draft), service.preview_values(draft))

    def apply() -> None:
        service.apply_draft(draft)
        populate()

    widget.apply_requested.connect(apply)
    populate()
    widget.resize(680, 360)
    widget.show()
    driver = ComponentDriver(widget)
    try:
        yield driver
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_adapter_note_is_visible_through_manual_selection_and_apply(
    review: ComponentDriver,
    tmp_path: Path,
) -> None:
    driver = review
    driver.screenshot(tmp_path / "01-initial-note.png")
    assert not driver.observe("general.ext.fake_peak*").checked
    note = driver.observe(object_name="writebackNote")
    assert note.visible and note.fully_visible
    assert note.text == "Fit needs manual review"
    assert driver.observe("general.ext.fake_peak*").enabled
    assert not driver.observe("Apply Selected").enabled
    driver.click("general.ext.fake_peak*")
    driver.screenshot(tmp_path / "02-manual-selection.png")
    assert driver.observe("general.ext.fake_peak*").checked
    assert driver.observe("Apply Selected").enabled
    assert driver.observe(object_name="writebackNote").text == note.text
    driver.click("Apply Selected")
    driver.screenshot(tmp_path / "03-applied.png")
    applied_checks = [
        widget
        for widget in driver.visible().widgets
        if widget.kind == "QCheckBox" and widget.text == "general.ext.fake_peak"
    ]
    assert len(applied_checks) == 1 and applied_checks[0].checked
    assert driver.observe(object_name="writebackNote").text == note.text
    assert driver.observe(object_name="writebackCurrent").text == "5.0"
