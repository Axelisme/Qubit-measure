"""Visible adapter notes and manual selection in the writeback review component."""

import json
import os
from collections.abc import Generator, Sequence
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QVBoxLayout, QWidget
from zcu_tools.experiment import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    ArbAssetWriteback,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.writeback import (
    WritebackItem,
    WritebackService,
    WritebackWidget,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._context_reads import capture_focus
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._synthetic import (
    ARB_PROPOSAL_KEY,
    ArbProposalAdapter,
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
def review(
    qapp: QApplication, tmp_path: Path, request: pytest.FixtureRequest
) -> Generator[ComponentDriver]:
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
    context.use_point("a")
    service = WritebackService(context, arb_names=lambda: ())
    asset_only = getattr(request, "param", None) == "arb"
    adapter = ArbProposalAdapter if asset_only else ReviewAdapter
    run = RunRecord(
        SampleCfg(sweep=SweepCfg(start=0.0, stop=1.0, expts=2, step=1.0)),
        SampleResult(np.array([1.0])),
    )
    items = adapter.get_writeback_items(
        WritebackRequest(
            run_result=run,
            analyze_result=SampleAnalysis(peak=0.3 if asset_only else 5.0),
            ctx=context.get_session_env(),
        )
    )
    if asset_only:
        items = [item for item in items if isinstance(item, ArbAssetWriteback)]
    draft = service.create_draft(
        items,
        parameters=capture_focus(context),
        experiment_name="test/review",
    )
    ctrl = MagicMock()
    ctrl.writeback_control.get_writeback_applied_for_pane.side_effect = lambda *_: (
        draft.get_all_applied()
    )
    ctrl.writeback_control.set_writeback_item_for_pane.side_effect = (
        lambda _tab, _pane, sid, **changes: draft.edit(sid, **changes)
    )
    host = QWidget()
    layout = QVBoxLayout(host)
    widget = WritebackWidget(ctrl, tab_id="review", pane="analysis")
    layout.addWidget(widget)
    if asset_only:
        paste = QLineEdit()
        paste.setObjectName("copiedArbSummary")
        layout.addWidget(QLabel("Paste copied summary JSON here"))
        layout.addWidget(paste)

    def populate() -> None:
        widget.populate(draft.preview(), draft.preview_values())

    def apply() -> None:
        draft.apply()
        populate()

    widget.apply_requested.connect(apply)
    populate()
    host.resize(680, 360)
    host.show()
    driver = ComponentDriver(host)
    try:
        yield driver
    finally:
        widget.detach()
        host.close()
        host.deleteLater()
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


@pytest.mark.parametrize("review", ["arb"], indirect=True)
def test_arb_asset_row_can_select_and_copy_summary_but_not_edit(
    review: ComponentDriver, tmp_path: Path
) -> None:
    evidence = Path(os.environ.get("WRITEBACK_ARB_EVIDENCE", str(tmp_path)))
    driver = review
    target = f"{ARB_PROPOSAL_KEY} · arb asset"
    driver.screenshot(evidence / "01-asset-row.png")
    assert driver.observe(f"{target}*").checked
    assert driver.observe(object_name="writebackCurrent").text == "absent"
    assert (
        driver.observe(object_name="writebackProposed").text
        == "3 samples · 0.75 µs · peak 0.5"
    )
    assert driver.observe("Copy").enabled
    assert not any(w.text == "Edit" for w in driver.visible().widgets)

    driver.click(f"{target}*")
    driver.screenshot(evidence / "02-deselected.png")
    assert not driver.observe("Apply Selected").enabled
    driver.click("Copy")
    driver.screenshot(evidence / "03-copy.png")
    driver.invoke_context_action(
        object_name="copiedArbSummary",
        action="&Paste\tCtrl+V",
        evidence=evidence / "04-paste-menu.png",
    )
    driver.screenshot(evidence / "05-pasted.png")
    assert json.loads(driver.observe(object_name="copiedArbSummary").text) == {
        "sample_count": 3,
        "duration_us": 0.75,
        "peak_abs": 0.5,
    }
    driver.click(f"{target}*")
    driver.screenshot(evidence / "06-reselected.png")
    assert driver.observe("Apply Selected").enabled
    driver.click("Apply Selected")
    driver.screenshot(evidence / "07-applied.png")
    assert driver.observe(target).checked
    assert driver.observe(object_name="writebackCurrent").text == "identical"
    assert driver.dialogs() == ()
