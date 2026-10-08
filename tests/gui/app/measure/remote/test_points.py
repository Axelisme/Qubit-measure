"""Point RPCs through real remote dispatch, context and tab owners."""

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from socket import socket

import pytest
from zcu_tools.cfg_model import WaveformAssetInfo
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.handlers.point import h_point_new
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    EntryInfo,
    EntryRequest,
    PointBindingError,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.resources.entry import ResultEntry, RoleSpec, source_path
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import (
    register_fakes,
    registry_state,
    restore_registry,
)
from zcu_lab.v2.twotone.time_domain.t1.gui import T1Adapter

from ._helpers import open_client, recv_response, send


class PointAdapter(T1Adapter):
    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        return {"driver": RoleSpec("fake/drive/*")}

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        return CfgSchema(
            CfgSectionSpec(fields={"rate": ScalarSpec("Rate", float)}),
            CfgSectionValue(fields={"rate": EvalValue("driver.rate")}),
        )


class PointController(Controller):
    """Give the fixture owned background teardown without exposing app APIs."""

    def quiesce(self) -> None:
        self._background_svc.quiesce()


@dataclass
class PointApplication:
    state: State
    controller: PointController
    remote: RemoteControlAdapter
    client: socket

    def rpc(self, method: str, params: Mapping[str, object]) -> dict[str, object]:
        send(
            self.client,
            {"id": "point-contract", "method": method, "params": dict(params)},
        )
        return recv_response(self.client, "point-contract")

    def result(self, method: str, params: Mapping[str, object]) -> dict[str, object]:
        reply = self.rpc(method, params)
        assert reply["ok"] is True, reply
        result = reply["result"]
        assert isinstance(result, dict)
        return result

    def open_entry(self) -> EntryInfo:
        self.controller.setup_control.new_entry(EntryRequest("entry"))
        info = self.controller.context_control.get_entry_info()
        assert info is not None
        return info


@pytest.fixture(scope="module", autouse=True)
def entry_models() -> Generator[None]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        yield
        assert registry_state() == installed, "point RPC module polluted registries"
    finally:
        restore_registry(before)


@pytest.fixture(autouse=True)
def registry_guard(request: pytest.FixtureRequest) -> Generator[None]:
    before = registry_state()
    yield
    assert registry_state() == before, f"registry polluter: {request.node.nodeid}"


@pytest.fixture
def application(qapp, tmp_path: Path) -> Generator[PointApplication]:
    registry = Registry()
    registry.register("pointed", PointAdapter)
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    controller = PointController(
        state, registry, None, BaseEventBus(), project_root=str(tmp_path)
    )
    remote = RemoteControlAdapter(
        controller, ControlOptions(port=0, app_slug=""), controller.owner_scheduler
    )
    client = open_client(remote.start())
    try:
        yield PointApplication(state, controller, remote, client)
    finally:
        client.close()
        remote.stop()
        controller.quiesce()
        for tab_id in tuple(state.list_tab_ids()):
            controller.tab_control.close_tab(tab_id)
        qapp.processEvents()


def new_params(info: EntryInfo, label: str, **extra: object) -> dict[str, object]:
    return {
        "entry_id": info.entry_id,
        "label": label,
        **extra,
    }


def opened_resource(info: EntryInfo) -> ResultEntry:
    return ResultEntry.open(
        info.name,
        result_root=info.result_path.parent,
        database_root=info.database_path.parent,
    )


def test_list_empty_and_refresh_persisted_metadata(
    application: PointApplication,
) -> None:
    assert application.result("point.list", {}) == {"entry": None, "points": []}
    info = application.open_entry()
    opened_resource(info).new_point("base", description="original")
    # A separate resource writer changes disk state after the GUI's cached list.
    with opened_resource(info).use_point("base").edit() as draft:
        draft.set("general.description", "external edit")
    assert application.result("point.list", {}) == {
        "entry": {
            "entry_id": info.entry_id,
            "name": info.name,
            "result_path": str(info.result_path),
            "database_path": str(info.database_path),
        },
        "points": [{"label": "base", "description": "external edit"}],
    }


def test_new_uses_setup_not_unbound_draft_and_clone_replaces_description(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    context = application.controller.context_control
    context.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    context.add_component("draft_only", "fake/drive/a", {"rate": 99.0})
    expected = {"kind": "saved_point", "entry_id": info.entry_id, "point": "seed"}
    assert (
        application.result(
            "point.new", new_params(info, "seed", description="seed description")
        )
        == expected
    )
    captured = context.read_context()
    assert captured.target == SavedPoint(entry_id=info.entry_id, point="seed")
    assert captured.lookup("Q1.rate").value == 10.0
    assert "draft_only" not in captured.document().components
    context.set_parameter(captured.target, "Q1.rate", 20.0)
    assert application.result(
        "point.new", new_params(info, "clone", clone_from="seed")
    ) == {"kind": "saved_point", "entry_id": info.entry_id, "point": "clone"}
    assert context.read_context().lookup("Q1.rate").value == 20.0
    assert context.read_context().document().general.description == ""
    assert application.result("point.list", {})["points"] == [
        {"label": "clone", "description": ""},
        {"label": "seed", "description": "seed description"},
    ]


def test_switch_keeps_old_tab_capture_and_new_tab_uses_selected_point(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    context = application.controller.context_control
    context.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    entry = opened_resource(info)
    entry.new_point("first")
    application.result("point.use", {"entry_id": info.entry_id, "label": "first"})
    context.set_parameter(context.read_context().target, "Q1.rate", 20.0)
    old_tab = application.result(
        "tab.new", {"adapter_name": "pointed", "target": "Q1"}
    )["tab_id"]
    assert isinstance(old_tab, str)
    entry.new_point("second")
    application.result("point.use", {"entry_id": info.entry_id, "label": "second"})
    new_tab = application.result(
        "tab.new", {"adapter_name": "pointed", "target": "Q1"}
    )["tab_id"]
    assert isinstance(new_tab, str)
    assert application.result(
        "tab.parameter_read",
        {"tab_id": old_tab, "paths": ["driver.rate"], "module_refs": []},
    ) == {"values": {"driver.rate": 20.0}, "modules": {}}
    assert application.result(
        "tab.parameter_read",
        {"tab_id": new_tab, "paths": ["driver.rate"], "module_refs": []},
    ) == {"values": {"driver.rate": 10.0}, "modules": {}}
    assert application.result(
        "point.use", {"entry_id": info.entry_id, "label": "first"}
    ) == {"kind": "saved_point", "entry_id": info.entry_id, "point": "first"}
    assert application.result(
        "tab.parameter_read",
        {"tab_id": new_tab, "paths": ["driver.rate"], "module_refs": []},
    )["values"] == {"driver.rate": 10.0}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entry_id", ""),
        ("label", ""),
        ("label", 24),
        ("description", []),
        ("clone_from", False),
    ],
)
def test_new_rejects_malformed_wire_without_creating(
    application: PointApplication,
    field: str,
    value: object,
) -> None:
    info = application.open_entry()
    params = new_params(info, "invalid")
    params[field] = value
    reply = application.rpc("point.new", params)
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.INVALID_PARAMS
    assert opened_resource(info).list_points() == []


@pytest.mark.parametrize("method", ["point.new", "point.use"])
def test_entry_preconditions_preserve_binding(
    application: PointApplication,
    method: str,
) -> None:
    reply = application.rpc(method, {"entry_id": "not-open", "label": "base"})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "no_entry"
    info = application.open_entry()
    before = application.controller.context_control.read_context().target
    reply = application.rpc(method, {"entry_id": "other-entry", "label": "base"})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["reason"] == "entry_not_open"
    assert application.controller.context_control.read_context().target == before
    assert opened_resource(info).list_points() == []


def test_missing_point_io_is_not_reclassified_or_replaced(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    opened_resource(info).new_point("good")
    application.result("point.use", {"entry_id": info.entry_id, "label": "good"})
    before = application.controller.context_control.read_context().target
    reply = application.rpc(
        "point.use", {"entry_id": info.entry_id, "label": "missing"}
    )
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.CONTROLLER_ERROR
    assert application.controller.context_control.read_context().target == before
    assert opened_resource(info).list_points() == ["good"]


def test_list_failure_keeps_cache_instead_of_returning_cached_success(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    opened_resource(info).new_point("base", description="cached")
    context = application.controller.context_control
    before = context.list_points(refresh=True)
    source_path(opened_resource(info).use_point("base")).write_text(
        "[invalid YAML", encoding="utf-8"
    )
    reply = application.rpc("point.list", {})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.CONTROLLER_ERROR
    assert context.list_points() == before


def test_metadata_read_and_point_write_do_not_unlock_context_guard(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    application.result("point.list", {})
    opened_resource(info).new_point("base")
    application.result("point.use", {"entry_id": info.entry_id, "label": "base"})
    reply = application.rpc(
        "context.ml_edit", {"kind": "module", "name": "unused", "edits": []}
    )
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "stale_version"
    assert error["data"] == {"stale": ["context"]}


def test_created_binding_failure_preserves_destination_and_recovers_with_use(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    entry = opened_resource(info)
    entry.new_point("previous")
    fail = False

    def bind_assets(path: Path) -> WaveformAssetInfo:
        if fail:
            raise OSError("synthetic asset binding failure")
        return ArbWaveformDatabase.info_at(path)

    service = ContextService(application.state, BaseEventBus(), bind_assets=bind_assets)
    service.install_entry(entry)
    service.use_point(PointUseRequest(info.entry_id, "previous"))
    application.remote.context_control = ContextControlFacet(context=service)
    before = service.read_context().target
    fail = True
    with pytest.raises(RemoteError) as failure:
        h_point_new(application.remote, new_params(info, "retained"))
    error = failure.value
    assert error.code == ErrorCode.CONTROLLER_ERROR
    assert error.reason == "point_binding_failed"
    assert error.data == {
        "entry_id": info.entry_id,
        "point": "retained",
        "created": True,
    }
    assert isinstance(error.__cause__, PointBindingError)
    assert isinstance(error.__cause__.__cause__, OSError)
    assert entry.list_points() == ["previous", "retained"]
    assert service.read_context().target == before
    fail = False
    assert application.result(
        "point.use", {"entry_id": info.entry_id, "label": "retained"}
    ) == {"kind": "saved_point", "entry_id": info.entry_id, "point": "retained"}
