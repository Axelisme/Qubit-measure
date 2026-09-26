"""Full GUI cfg observation and frozen Run rejection for non-finite edits."""

from pathlib import Path

import pytest
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.gui.app.main.state import Session
from zcu_tools.gui.cfg.binding import ScalarField

from ._helpers import Fixture, call, mcp_client, open_client

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture
def live_cfg(qapp):
    fx = Fixture()
    cfg = FakeAdapter().make_default_cfg(fx.state.exp_context)
    tab_id = "tab-live"
    fx.state.add_tab(
        tab_id, Session(adapter_name="fake", adapter=FakeAdapter(), cfg_schema=cfg)
    )
    editor_id, _ = fx.ctrl.open_seeded_cfg_editor(cfg, gc=False, owner_key=tab_id)
    fx.start()
    try:
        yield fx, tab_id, editor_id
    finally:
        fx.stop()


@pytest.mark.parametrize("text", ["nan", "inf"])
def test_full_cfg_reads_preserve_invalid_input_and_run_rejects_it(
    live_cfg, text: str
) -> None:
    fx, tab_id, editor_id = live_cfg
    draft = fx.ctrl.get_cfg_editor_draft(editor_id)
    gain = draft.root.fields["gain"]
    assert isinstance(gain, ScalarField)
    gain.set_text(text)

    sock = open_client(fx.service.port)
    try:
        tab = call(sock, "tab.get_cfg", {"tab_id": tab_id}, rid="tab")
        editor = call(sock, "editor.get", {"editor_id": editor_id}, rid="editor")
        assert tab["ok"] and editor["ok"]
        assert tab["result"]["tree"] == editor["result"]["tree"]
        observed = tab["result"]["tree"]["children"]["gain"]
        assert not observed["valid"]
        assert observed["input"]["raw"] == text
        assert observed["input"]["resolved"] is None
        assert "finite" in observed["input"]["error"]

        rejected = call(sock, "tab.run_start", {"tab_id": tab_id}, rid="run")
        assert not rejected["ok"]
        assert rejected["error"]["reason"] == "invalid_cfg"
    finally:
        sock.close()


def test_mcp_reads_full_invalid_cfg_from_gui(
    live_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This test owns cfg observation, not unrelated SoC status orientation.
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")
    monkeypatch.setattr("zcu_tools.mcp.measure.tools_lifecycle.status", lambda *_: {})
    fx, tab_id, editor_id = live_cfg
    draft = fx.ctrl.get_cfg_editor_draft(editor_id)
    gain = draft.root.fields["gain"]
    assert isinstance(gain, ScalarField)
    gain.set_text("nan")

    bridge, invoke = mcp_client(fx.service.port, tmp_path)
    try:
        invoke("connect", {"port": fx.service.port})
        for method, params in (
            ("tab.get_cfg", {"tab_id": tab_id}),
            ("editor.get", {"editor_id": editor_id}),
        ):
            observed = invoke("rpc_call", {"method": method, "params": params})["tree"][
                "children"
            ]["gain"]
            assert not observed["valid"]
            assert observed["input"]["raw"] == "nan"
            assert observed["input"]["resolved"] is None
            assert "finite" in observed["input"]["error"]
    finally:
        bridge.disconnect()
