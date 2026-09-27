"""A GUI form edit publishes invalid cfg to socket and MCP before Run."""

from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtWidgets import QLineEdit
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.gui.app.main.state import Session
from zcu_tools.gui.cfg import CfgSchema, DirectValue, ScalarSpec
from zcu_tools.gui.cfg.binding import ScalarField
from zcu_tools.gui.widgets.cfg import CfgFormWidget
from zcu_tools.gui.widgets.cfg.fields.common import ScalarWidget

from ._helpers import Fixture, call, mcp_client, open_client

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture
def live_cfg_form(qapp):
    fx = Fixture()
    cfg = FakeAdapter().make_default_cfg(fx.state.exp_context)
    gain_spec = cfg.spec.fields["gain"]
    assert isinstance(gain_spec, ScalarSpec)
    # An optional numeric text field uses QLineEdit; the other FakeAdapter
    # inputs remain the same, so the generic run guard owns invalidity.
    cfg = CfgSchema(
        spec=replace(
            cfg.spec,
            fields={**cfg.spec.fields, "gain": replace(gain_spec, optional=True)},
        ),
        value=cfg.value,
    )
    tab_id = "tab-live"
    fx.state.add_tab(
        tab_id, Session(adapter_name="fake", adapter=FakeAdapter(), cfg_schema=cfg)
    )
    editor_id, _ = fx.ctrl.open_seeded_cfg_editor(cfg, gc=False, owner_key=tab_id)
    draft = fx.ctrl.get_cfg_editor_draft(editor_id)
    gain = draft.root.fields["gain"]
    assert isinstance(gain, ScalarField)
    form = CfgFormWidget()
    form.attach(draft)
    gain_widget = next(
        field_widget
        for field_widget in form.findChildren(ScalarWidget)
        if field_widget.field is gain
    )
    entry = gain_widget.findChild(QLineEdit)
    assert entry is not None
    fx.start()
    try:
        yield fx, tab_id, editor_id, form, entry
    finally:
        form.detach()
        form.close()
        form.deleteLater()
        fx.stop()


@pytest.mark.parametrize("text", ["nan", "inf"])
def test_form_edit_publishes_complete_gui_and_mcp_cfg_then_blocks_run(
    live_cfg_form, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    # The MCP connect status orientation is unrelated to cfg observation.
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    fx, tab_id, editor_id, form, entry = live_cfg_form
    entry.setText(text)
    observed_form = form.read_values().fields["gain"]
    assert isinstance(observed_form, DirectValue)
    assert observed_form.raw == text
    assert observed_form.value is None
    assert observed_form.error is not None
    published = fx.state.get_tab(tab_id).cfg_schema.value.fields["gain"]
    assert published == observed_form

    sock = open_client(fx.service.port)
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        tab = call(sock, "tab.get_cfg", {"tab_id": tab_id}, rid="tab")
        editor = call(sock, "editor.get", {"editor_id": editor_id}, rid="editor")
        assert tab["ok"] and editor["ok"]
        tree = tab["result"]["tree"]
        assert tree == editor["result"]["tree"]
        assert not tree["children"]["gain"]["valid"]
        assert tree["children"]["gain"]["input"]["raw"] == text
        assert tree["children"]["gain"]["input"]["resolved"] is None
        assert "finite" in tree["children"]["gain"]["input"]["error"]

        invoke("connect", {"port": fx.service.port})
        for method, params in (
            ("tab.get_cfg", {"tab_id": tab_id}),
            ("editor.get", {"editor_id": editor_id}),
        ):
            assert (
                invoke("rpc_call", {"method": method, "params": params})["tree"] == tree
            )

        rejected = call(sock, "tab.run_start", {"tab_id": tab_id}, rid="run")
        assert not rejected["ok"]
        assert rejected["error"]["reason"] == "invalid_cfg"
        assert "gain" in rejected["error"]["message"]
    finally:
        bridge.disconnect()
        sock.close()
