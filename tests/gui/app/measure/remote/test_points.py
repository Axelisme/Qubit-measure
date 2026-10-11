"""Point RPCs through real remote dispatch, context and tab owners."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import WaveformAssetInfo
from zcu_tools.gui.app.measure.remote.handlers.point import h_point_new
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
from zcu_tools.resources.entry import ResultEntry, source_path
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from ._point_application import PointApplication


def new_params(label: str, **extra: object) -> dict[str, object]:
    return {"label": label, **extra}


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
            "point.new", new_params("seed", description="seed description")
        )
        == expected
    )
    captured = context.read_context()
    assert captured.target == SavedPoint(entry_id=info.entry_id, point="seed")
    assert captured.lookup("Q1.rate").value == 10.0
    assert "draft_only" not in captured.document().components
    context.set_parameter(captured.target, "Q1.rate", 20.0)
    assert application.result("point.new", new_params("clone", clone_from="seed")) == {
        "kind": "saved_point",
        "entry_id": info.entry_id,
        "point": "clone",
    }
    assert context.read_context().lookup("Q1.rate").value == 20.0
    assert context.read_context().document().general.description == ""
    assert application.result("point.list", {})["points"] == [
        {"label": "clone", "description": ""},
        {"label": "seed", "description": "seed description"},
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
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
    params = new_params("invalid")
    params[field] = value
    reply = application.rpc("point.new", params)
    assert reply["ok"] is False
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.INVALID_PARAMS
    assert opened_resource(info).list_points() == []


@pytest.mark.parametrize("method", ["point.new", "point.use"])
def test_mutations_without_entry_keep_owner_no_entry(
    application: PointApplication,
    method: str,
) -> None:
    reply = application.rpc(method, {"label": "base"})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "no_entry"


@pytest.mark.parametrize("method", ["point.new", "point.use"])
def test_mutations_use_current_entry_after_switch(
    application: PointApplication,
    method: str,
) -> None:
    previous = application.open_entry()
    opened_resource(previous).new_point("selected")
    application.controller.setup_control.new_entry(EntryRequest("other"))
    current = application.controller.context_control.get_entry_info()
    assert current is not None
    if method == "point.use":
        opened_resource(current).new_point("selected")

    assert application.result(method, {"label": "selected"}) == {
        "kind": "saved_point",
        "entry_id": current.entry_id,
        "point": "selected",
    }
    assert application.controller.context_control.read_context().target == SavedPoint(
        entry_id=current.entry_id, point="selected"
    )
    assert opened_resource(previous).list_points() == ["selected"]
    assert opened_resource(current).list_points() == ["selected"]


def test_missing_point_io_is_not_reclassified_or_replaced(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    opened_resource(info).new_point("good")
    application.result("point.use", {"label": "good"})
    before = application.controller.context_control.read_context().target
    reply = application.rpc("point.use", {"label": "missing"})
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
    application.result("point.use", {"label": "base"})
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
    service.use_point(PointUseRequest("previous"))
    application.remote.context_control = ContextControlFacet(context=service)
    before = service.read_context().target
    fail = True
    with pytest.raises(RemoteError) as failure:
        h_point_new(application.remote, new_params("retained"))
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
    assert application.result("point.use", {"label": "retained"}) == {
        "kind": "saved_point",
        "entry_id": info.entry_id,
        "point": "retained",
    }
