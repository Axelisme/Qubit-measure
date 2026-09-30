"""A GUI form edit publishes invalid cfg to socket and MCP before Run."""

from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtWidgets import QLineEdit, QWidget
from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter
from zcu_tools.gui.cfg import CfgSchema, DirectValue, ScalarSpec
from zcu_tools.gui.cfg.edit_codec import encode_ref
from zcu_tools.gui.widgets.cfg.resource_form import ResourceCfgFormWidget

from ._helpers import Fixture, call, mcp_client, observe_run_inputs, open_client

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture
def live_cfg_form(qapp):
    fx = Fixture()
    cfg = FakeAdapter().make_default_cfg(fx.state.session_env)
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
    resource = fx.prepare_tab(tab_id, FakeAdapter(), cfg)
    form = ResourceCfgFormWidget()
    form.attach(resource)
    gain_widget = form.findChild(QWidget, "cfgInput:gain")
    assert gain_widget is not None
    entry = gain_widget.findChild(QLineEdit)
    assert entry is not None
    fx.start()
    try:
        yield fx, tab_id, resource, form, entry
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
    fx, tab_id, resource, form, entry = live_cfg_form
    entry.setText(text)
    observed_form = resource.observe().tree.children["gain"].value
    assert isinstance(observed_form, DirectValue)
    assert observed_form.raw == text
    assert observed_form.value is None
    assert observed_form.error is not None
    assert form.is_valid() is False

    sock = open_client(fx.service.port)
    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        tab = call(sock, "tab.get_cfg", {"tab_id": tab_id}, rid="tab")
        assert tab["ok"]
        tree = tab["result"]["tree"]
        assert not tree["children"]["gain"]["valid"]
        assert tree["children"]["gain"]["input"]["raw"] == text
        assert tree["children"]["gain"]["input"]["resolved"] is None
        assert "finite" in tree["children"]["gain"]["input"]["error"]

        invoke("connect", {"port": fx.service.port})
        for method, params in (("tab.get_cfg", {"tab_id": tab_id}),):
            assert (
                invoke("rpc_call", {"method": method, "params": params})["tree"] == tree
            )

        observe_run_inputs(
            fx, tab_id, lambda method, params: call(sock, method, params)["result"]
        )
        rejected = call(
            sock,
            "tab.run_start",
            {
                "tab_id": tab_id,
                "expected": encode_ref(
                    fx.ctrl.cfg_resources.lookup(tab_id).observe().ref
                ),
            },
            rid="run",
        )
        assert not rejected["ok"]
        assert rejected["error"]["reason"] == "not_valid"
        assert tab["result"]["diagnostics"][0]["path"] == ["gain"]
    finally:
        bridge.disconnect()
        sock.close()
