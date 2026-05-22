from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import QPushButton
from zcu_tools.experiment.v2_gui.adapters.onetone.fakefreq import FakeFreqAdapter
from zcu_tools.gui.adapter import ExpContext
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.ui.cfg_form import CfgFormWidget
from zcu_tools.gui.ui.writeback_dialog import WritebackDialog
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
    )


def test_writeback_dialog_lists_items_and_edit_buttons(qapp):
    ctx = _make_ctx()
    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)
    result = adapter.run(ctx, schema)
    analyze_result = adapter.analyze(result, ctx)
    items = adapter.get_writeback_items(analyze_result, ctx)

    dialog = WritebackDialog(items, MagicMock())
    selected = dialog.get_selected_items()
    edit_buttons = [w for w in dialog.findChildren(QPushButton) if w.text() == "Edit"]

    assert len(selected) == len(items)
    assert len(edit_buttons) >= 4


def test_readout_writeback_drag_schema_is_initially_valid(qapp):
    ctx = _make_ctx()
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_current_md.return_value = ctx.md
    ctrl.get_current_ml.return_value = ctx.ml
    ctrl.is_running.return_value = False
    ctrl.has_soc.return_value = False

    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)
    result = adapter.run(ctx, schema)
    analyze_result = adapter.analyze(result, ctx)
    items = adapter.get_writeback_items(analyze_result, ctx)
    readout_item = next(item for item in items if item.key == "readout_rf")

    form = CfgFormWidget()
    form.populate(readout_item.edit_schema, ctrl)  # type: ignore[arg-type]

    assert form.is_valid() is True
