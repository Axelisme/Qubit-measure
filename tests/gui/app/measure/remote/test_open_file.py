"""New-tab loading through the public GUI socket contract."""

from dataclasses import dataclass, replace
from typing import ClassVar

import pytest
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    LoadDataRequest,
)
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.cfg import DirectValue
from zcu_tools.gui.session.types import ContextReadiness

from tests.gui.app.measure._reload_fakes import OldAdapter

from ._helpers import Fixture, call, open_client, reset_inbox

pytestmark = pytest.mark.uses_wall_clock


class LoadedCfg(ExpCfgModel):
    knob: int = 42


@dataclass
class LoadedResult:
    cfg_snapshot: LoadedCfg | None


class FileAdapter(OldAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        load_data=True, analysis=AnalysisMode.NONE
    )

    def load(self, req: LoadDataRequest) -> LoadedResult:
        if req.data_path == "bad.h5":
            raise ValueError("wrong experiment")
        return LoadedResult(None if req.data_path == "no-cfg.h5" else LoadedCfg())


@pytest.fixture
def app(qapp):
    fx = Fixture(empty_project=True)
    fx.state.set_context(
        replace(fx.state.session_env, readiness=ContextReadiness.DRAFT)
    )
    fx.registry.register("file", FileAdapter)
    fx.start()
    sock = open_client(fx.service.port)
    try:
        yield fx, sock
    finally:
        reset_inbox(sock)
        sock.close()
        fx.stop()


def test_open_file_requires_current_context_before_creating_tab(app):
    fx, sock = app
    params = {"adapter_name": "file", "data_path": "saved.h5"}
    reply = call(sock, "tab.open_file", params)
    assert reply["error"]["reason"] == "stale_version"
    assert fx.state.list_tab_ids() == []
    assert call(sock, "context.snapshot")["ok"] is True
    fx.ctrl.context_control.create_md_attr("changed", 1)
    reply = call(sock, "tab.open_file", params)
    assert reply["error"]["reason"] == "stale_version"
    assert fx.state.list_tab_ids() == []
    assert call(sock, "context.snapshot")["ok"] is True
    assert call(sock, "tab.open_file", params)["ok"] is True


@pytest.mark.parametrize(
    "path, backfill, knob",
    [("saved.h5", "applied", 42), ("no-cfg.h5", "not_applied", 7)],
)
def test_open_file_loads_without_soc_but_does_not_observe_new_subresources(
    app, path, backfill, knob
):
    fx, sock = app
    previous = fx.ctrl.new_tab("file")
    assert call(sock, "context.snapshot")["ok"] is True
    reply = call(sock, "tab.open_file", {"adapter_name": "file", "data_path": path})
    assert reply["ok"] is True
    outcome = reply["result"]
    tab = outcome["tab_id"]
    assert tab != previous
    assert outcome["cfg_backfill"] == backfill
    assert fx.state.active_tab_id == tab
    assert fx.state.session_env.soc is None
    assert fx.state.get_tab(tab).run.result == LoadedResult(
        None if backfill == "not_applied" else LoadedCfg()
    )
    assert fx.state.get_tab(tab).cfg_schema.value.fields["knob"] == DirectValue(knob)
    stale = call(sock, "tab.load_data", {"tab_id": tab, "data_path": path})
    assert stale["error"]["reason"] == "stale_version"
    keys = stale["error"]["data"]["stale"]
    assert f"tab:{tab}:result" in keys
    assert f"tab:{tab}:analyze" in keys
    assert f"tab:{tab}" not in keys
    run = call(sock, "tab.run_start", {"tab_id": tab})
    assert run["error"]["reason"] == "stale_version"
    assert f"tab:{tab}:cfg" in run["error"]["data"]["stale"]
    snapshot = call(sock, "tab.snapshot", {"tab_id": tab})["result"]["tabs"][0]
    assert snapshot["artifacts"][0]["status"] == "not_saved"
    assert snapshot["artifacts"][0]["last_saved_path"] is None
    assert snapshot["artifacts"][0]["is_saveable"] is True
    assert call(sock, "tab.load_data", {"tab_id": tab, "data_path": path})["ok"] is True


def test_open_file_backfill_failure_retains_loaded_result(app, monkeypatch):
    fx, sock = app

    def reject_replacement(self, *args, **kwargs):
        raise ValueError("cfg cannot be prepared")

    monkeypatch.setattr(CfgEditorService, "prepare_replacement", reject_replacement)
    assert call(sock, "context.snapshot")["ok"] is True
    reply = call(
        sock, "tab.open_file", {"adapter_name": "file", "data_path": "saved.h5"}
    )
    assert reply["ok"] is True
    assert reply["result"]["cfg_backfill"] == "not_applied"
    tab = reply["result"]["tab_id"]
    assert fx.state.active_tab_id == tab
    assert fx.state.get_tab(tab).run.result == LoadedResult(LoadedCfg())
    assert fx.state.get_tab(tab).cfg_schema.value.fields["knob"] == DirectValue(7)


@pytest.mark.parametrize("has_previous", [False, True])
def test_open_file_failed_load_cleans_up_and_restores_focus(app, has_previous):
    fx, sock = app
    previous = fx.ctrl.new_tab("file") if has_previous else None
    assert call(sock, "context.snapshot")["ok"] is True
    reply = call(sock, "tab.open_file", {"adapter_name": "file", "data_path": "bad.h5"})
    assert reply["error"]["reason"] == "invalid_data_file"
    assert fx.state.list_tab_ids() == ([] if previous is None else [previous])
    assert fx.state.active_tab_id == previous
