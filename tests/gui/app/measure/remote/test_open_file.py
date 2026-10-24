"""New-tab loading through the public GUI socket contract."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)

from tests.experiment._saving_support import SavingCfg
from tests.gui.app.measure._load_support import historical_metadata, write_native_file
from tests.gui.app.measure._native_save_support import SavingAdapter
from tests.gui.app.measure._record_support import require_record

from ._helpers import Fixture, call, open_client, reset_inbox

pytestmark = pytest.mark.uses_wall_clock


FileAdapter = Adapter(
    spec=replace(SavingAdapter.spec, tag="test/load"),
    make_form=SavingAdapter.make_form,
    run=SavingAdapter.run,
    guide=None,
    analysis=None,
    post_analysis=None,
)


@pytest.fixture
def app(qapp, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_native_file(tmp_path / "saved.h5")
    write_native_file(tmp_path / "no-cfg.h5", gain=2.0, cfg_schema_version="2.0")
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
    record = require_record(fx.state.get_tab(tab).run)
    assert record.cfg == (None if path == "no-cfg.h5" else SavingCfg())
    gain = 2.0 if path == "no-cfg.h5" else 1.0
    np.testing.assert_array_equal(
        record.result.signal, np.array([1 + 2j, 3 - 4j, 5 + 6j]) * gain
    )
    assert record.metadata == historical_metadata()
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
