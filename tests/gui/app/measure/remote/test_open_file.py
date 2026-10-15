"""New-tab loading through the public GUI socket contract."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from zcu_tools.datafile import StoredRun
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.native_persistence import RestoredRun
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
)
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)

from tests.gui.app.measure._load_support import write_native_file
from tests.gui.app.measure._reload_fakes import OldAdapter

from ._helpers import Fixture, call, open_client, reset_inbox

pytestmark = pytest.mark.uses_wall_clock


class LoadedCfg(ExpCfgModel):
    knob: int = 42


@dataclass
class LoadedResult:
    gain: object


class FileAdapter(OldAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        load_data=True, analysis=AnalysisMode.NONE
    )

    experiment_tag: ClassVar[str | None] = "test/load"

    def restore(self, stored: StoredRun):
        gain = stored.cfg.values["gain"]
        cfg = None if gain == 2.0 else LoadedCfg()
        return RestoredRun(
            RunRecord(cfg=cfg, result=LoadedResult(gain)),
            "cfg unavailable" if cfg is None else None,
        )


@pytest.fixture
def app(qapp, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_native_file(tmp_path / "saved.h5")
    write_native_file(tmp_path / "no-cfg.h5", gain=2.0)
    write_native_file(tmp_path / "bad.h5", tag="wrong/experiment")
    fx = Fixture(empty_project=True, project_root=str(tmp_path))
    fx.ctrl.setup_control.new_entry(EntryRequest("fake_qubit"))
    fx.ctrl.setup_control.new_point(
        PointCreateRequest("working", "", CommandOrigin("gui"))
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


@pytest.mark.parametrize("path", ["saved.h5", "no-cfg.h5"])
def test_open_file_loads_without_soc_with_result(app, path):
    fx, sock = app
    previous = fx.ctrl.new_tab("file")
    reply = call(sock, "tab.open_file", {"adapter_name": "file", "data_path": path})
    assert reply["ok"] is True
    outcome = reply["result"]
    tab = outcome["tab_id"]
    assert tab != previous
    assert fx.state.active_tab_id == tab
    assert fx.state.session_env.soc is None
    assert fx.state.get_tab(tab).run.result == RunRecord(
        cfg=None if path == "no-cfg.h5" else LoadedCfg(),
        result=LoadedResult(2.0 if path == "no-cfg.h5" else 1.0),
    )
    snapshot = call(sock, "tab.snapshot", {"tab_id": tab})["result"]["tabs"][0]
    assert snapshot["run_id"] is None
    assert snapshot["result_state"]["available"] is True
    assert snapshot["result_state"]["source_path"] == path


@pytest.mark.parametrize("has_previous", [False, True])
def test_open_file_failed_load_cleans_up_and_restores_focus(app, has_previous):
    fx, sock = app
    previous = fx.ctrl.new_tab("file") if has_previous else None
    assert call(sock, "context.snapshot")["ok"] is True
    reply = call(sock, "tab.open_file", {"adapter_name": "file", "data_path": "bad.h5"})
    assert reply["error"]["reason"] == "invalid_data_file"
    assert fx.state.list_tab_ids() == ([] if previous is None else [previous])
    assert fx.state.active_tab_id == previous
