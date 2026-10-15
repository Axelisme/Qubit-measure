"""New-tab loading through the public GUI socket contract."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    LoadDataRequest,
)
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)

from tests.gui.app.measure._load_support import historical_metadata
from tests.gui.app.measure._reload_fakes import OldAdapter

from ._helpers import Fixture, call, open_client, reset_inbox

pytestmark = pytest.mark.uses_wall_clock


class LoadedCfg(ExpCfgModel):
    knob: int = 42


@dataclass
class LoadedResult:
    data_path: str


class FileAdapter(OldAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        load_data=True, analysis=AnalysisMode.NONE
    )

    def load(self, req: LoadDataRequest):
        if req.data_path == "bad.h5":
            raise ValueError("wrong experiment")
        cfg = None if req.data_path == "no-cfg.h5" else LoadedCfg()
        return RunRecord(
            cfg=cfg, result=LoadedResult(req.data_path)
        ), historical_metadata()


@pytest.fixture
def app(qapp):
    fx = Fixture(empty_project=True)
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
        result=LoadedResult(path),
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
