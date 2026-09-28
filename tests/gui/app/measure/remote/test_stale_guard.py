"""Connection-owned guard behavior through the shipped GUI socket contract."""

from dataclasses import replace

import pytest
from zcu_tools.program.v2 import WaveformCfgFactory
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from ._helpers import Fixture, call, open_client, reset_inbox

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture()
def service(qapp):
    fx = Fixture(active_label="ctx001")
    fx.state.set_context(
        replace(fx.state.session_env, md=MetaDict(), ml=ModuleLibrary())
    )
    fx.start()
    yield fx
    fx.stop()


@pytest.mark.parametrize("prefix", ["", "length", "missing"])
def test_partial_editor_read_cannot_unlock_commit(service, prefix: str) -> None:
    service.state.session_env.ml.waveforms["seed"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.1}
    )
    sock = open_client(service.service.port)
    try:
        editor = call(
            sock, "editor.new", {"item_kind": "waveform", "from_name": "seed"}
        )["result"]["editor_id"]
        assert call(sock, "context.snapshot")["ok"] is True
        assert (
            call(sock, "editor.get", {"editor_id": editor, "prefix": prefix})["ok"]
            is True
        )
        params = {"editor_id": editor, "name": "copy"}
        rejected = call(sock, "editor.commit", params)
        assert rejected["error"]["reason"] == "stale_version"
        assert f"editor:{editor}" in rejected["error"]["data"]["stale"]
        assert call(sock, "editor.get", {"editor_id": editor})["ok"] is True
        assert call(sock, "editor.commit", params)["ok"] is True
        assert (
            service.state.session_env.ml.waveforms["copy"]
            == service.state.session_env.ml.waveforms["seed"]
        )
    finally:
        reset_inbox(sock)
        sock.close()


def test_versions_and_tab_index_do_not_establish_seen_and_reconnect_forgets_it(
    service,
) -> None:
    tab = service.ctrl.new_tab("fake")
    params = {"tab_id": tab, "data_path": "missing.h5"}
    sock = open_client(service.service.port)
    try:
        assert call(sock, "resources.versions")["ok"] is True
        assert call(sock, "tab.snapshot")["ok"] is True
        assert call(sock, "context.snapshot")["ok"] is True
        stale = call(sock, "tab.load_data", params)
        assert stale["error"]["reason"] == "stale_version"
        assert f"tab:{tab}:result" in stale["error"]["data"]["stale"]
        assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"] is True
        assert (
            call(sock, "tab.load_data", params)["error"]["reason"] == "unsupported_load"
        )
    finally:
        reset_inbox(sock)
        sock.close()

    reconnect = open_client(service.service.port)
    try:
        assert (
            call(reconnect, "tab.load_data", params)["error"]["reason"]
            == "stale_version"
        )
    finally:
        reset_inbox(reconnect)
        reconnect.close()


def test_reading_another_tab_does_not_refresh_changed_result(service) -> None:
    tab = service.ctrl.new_tab("fake")
    other = service.ctrl.new_tab("fake")
    sock = open_client(service.service.port)
    try:
        assert call(sock, "context.snapshot")["ok"] is True
        assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"] is True
        service.state.update_tab_result(tab, object())
        assert call(sock, "tab.snapshot", {"tab_id": other})["ok"] is True
        params = {"tab_id": tab, "data_path": "missing.h5"}
        assert call(sock, "tab.load_data", params)["error"]["reason"] == "stale_version"
        assert call(sock, "tab.snapshot", {"tab_id": tab})["ok"] is True
        assert (
            call(sock, "tab.load_data", params)["error"]["reason"] == "unsupported_load"
        )
    finally:
        reset_inbox(sock)
        sock.close()
