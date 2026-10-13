"""Context/value remote dispatch through real point-local library owners."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.remote.errors import ErrorCode
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryInfo
from zcu_tools.gui.session.value_lookup import ValueInfo
from zcu_tools.resources.entry import (
    ResultEntry,
    document_meta,
    document_value,
    snapshot,
)

from ._helpers import dispatch_handler as _dispatch
from ._point_application import PointApplication

Kind = Literal["module", "waveform"]


def entry_resource(info: EntryInfo) -> ResultEntry:
    return ResultEntry.open(
        info.name,
        result_root=info.result_path.parent,
        database_root=info.database_path.parent,
    )


def rpc_error(
    application: PointApplication, method: str, params: dict[str, object]
) -> dict[str, object]:
    reply = application.rpc(method, params)
    assert reply["ok"] is False, reply
    error = reply["error"]
    assert isinstance(error, dict)
    return error


@pytest.mark.parametrize("selection", ["no_entry", "entry", "point"])
def test_alias_apply_returns_authored_rows_without_requiring_resolution(
    application: PointApplication, selection: str
) -> None:
    info = application.open_entry() if selection != "no_entry" else None
    if selection == "point":
        application.result("point.new", {"label": "base"})
    control = application.controller.context_control
    rows = [
        {"alias": "driver", "value": "missing"},
        {"alias": "control", "value": "$driver"},
        {"alias": "empty", "value": ""},
    ]
    assert application.result("context.aliases_set", {"rows": rows}) == {"rows": rows}
    assert control.get_alias_rows() == (
        AliasRow("driver", "missing"),
        AliasRow("control", "$driver"),
        AliasRow("empty", ""),
    )
    if info is not None:
        entry = entry_resource(info)
        stored = entry.load_aliases()
        assert stored is not None
        assert list(stored.aliases.items()) == [
            ("driver", "missing"),
            ("control", "$driver"),
            ("empty", ""),
        ]
        assert entry.ledger.events() == ()
    if selection != "point":
        error = rpc_error(application, "context.snapshot", {"path": "aliases"})
        assert (error["code"], error["reason"]) == (
            ErrorCode.PRECONDITION_FAILED,
            "no_context",
        )
    assert application.result("context.aliases_set", {"rows": []}) == {"rows": []}
    assert control.get_alias_rows() == ()
    if info is not None:
        stored = entry_resource(info).load_aliases()
        assert stored is not None
        assert stored.aliases == {}


@pytest.mark.parametrize(
    "rows",
    [
        {},
        [None],
        [{"alias": "driver"}],
        [{"alias": "driver", "value": 1}],
        [{"alias": "driver", "value": "Q1", "resolved": "Q1"}],
        [{"alias": "driver", "value": "Q1"}, {"alias": "driver", "value": "Q2"}],
        [{"alias": "driver", "value": "Q1"}, {"alias": "control", "value": "$later"}],
    ],
)
def test_alias_apply_rejects_entire_payload_and_preserves_persisted_table(
    application: PointApplication, rows: object
) -> None:
    info = application.open_entry()
    control = application.controller.context_control
    original = (AliasRow("driver", "original"),)
    control.apply_alias_rows(original)
    old_file = (info.result_path / "aliases.yaml").read_bytes()
    error = rpc_error(application, "context.aliases_set", {"rows": rows})
    assert error["code"] == ErrorCode.INVALID_PARAMS
    assert control.get_alias_rows() == original
    assert (info.result_path / "aliases.yaml").read_bytes() == old_file
    assert entry_resource(info).ledger.events() == ()


def test_alias_apply_storage_failure_does_not_install_candidate(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    control = application.controller.context_control
    original = (AliasRow("driver", "original"),)
    control.set_alias_rows(original)
    # A directory destination forces the real entry's atomic replacement to fail.
    (info.result_path / "aliases.yaml").mkdir()
    error = rpc_error(application, "context.aliases_set", {"rows": []})
    assert error["code"] == ErrorCode.CONTROLLER_ERROR
    assert control.get_alias_rows() == original
    assert (info.result_path / "aliases.yaml").is_dir()


def test_parameter_set_commits_current_point_preserving_metadata_and_captures(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    control = application.controller.context_control
    control.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    application.result("point.new", {"label": "first"})
    first = control.read_context()
    application.result("point.new", {"label": "selected"})
    selected = control.read_context()
    metadata = {"comment": "manual", "stderr": 0.2}
    control.set_parameter(selected.target, "Q1.rate", 12.0, metadata=metadata)
    before = control.read_context()
    assert application.result(
        "context.set_parameter", {"path": "Q1.rate", "value": 25.0}
    ) == {"path": "Q1.rate", "before": 12.0, "value": 25.0}
    after = control.read_context()
    persisted = snapshot(entry_resource(info).use_point("selected"))
    assert after.target == selected.target
    assert document_value(persisted, "Q1.rate").value == 25.0
    assert document_meta(persisted, "Q1.rate") == metadata
    assert before.lookup_point("Q1.rate").value == 12.0
    assert control.read_target(first.target).lookup_point("Q1.rate").value == 10.0
    assert document_value(snapshot(entry_resource(info).setup), "Q1.rate").value == 10.0
    assert entry_resource(info).ledger.events() == ()


def test_parameter_set_absent_before_and_nested_yaml_are_not_post_write_readback(
    application: PointApplication,
) -> None:
    application.open_entry()
    application.result("point.new", {"label": "base"})
    value = {"samples": [1, None, {"enabled": True}], "label": "manual"}
    assert application.result(
        "context.set_parameter", {"path": "general.ext.test", "value": value}
    ) == {"path": "general.ext.test", "before": None, "value": value}
    assert (
        application.controller.context_control.read_context()
        .lookup_point("general.ext.test")
        .value
        == value
    )


@pytest.mark.parametrize("opened", [False, True])
def test_parameter_set_without_point_preserves_failed_precondition(
    application: PointApplication, opened: bool
) -> None:
    if opened:
        application.open_entry()
    error = rpc_error(
        application,
        "context.set_parameter",
        {"path": "general.description", "value": "new"},
    )
    assert (error["code"], error["reason"]) == (
        ErrorCode.PRECONDITION_FAILED,
        "no_context",
    )
    assert application.state.session_env.parameters is None


@pytest.mark.parametrize("path", ["$driver.rate", "Q1.pulses.pi"])
def test_parameter_set_rejects_nonconcrete_parameter_path(
    application: PointApplication, path: str
) -> None:
    info = application.open_entry()
    control = application.controller.context_control
    control.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    application.result("point.new", {"label": "base"})
    control.apply_alias_rows((AliasRow("driver", "Q1"),))
    before = control.read_context()
    error = rpc_error(
        application, "context.set_parameter", {"path": path, "value": 20.0}
    )
    assert error["code"] == ErrorCode.INVALID_PARAMS
    assert control.read_context().document() == before.document()
    assert snapshot(entry_resource(info).use_point("base")) == before.document()


@pytest.mark.parametrize(
    "method,params",
    [
        ("context.aliases_set", {"rows": []}),
        ("context.set_parameter", {"path": "general.description", "value": "new"}),
    ],
)
def test_context_mutations_require_holder_without_domain_effects(
    application: PointApplication, method: str, params: dict[str, object]
) -> None:
    application.open_entry()
    application.result("point.new", {"label": "base"})
    control = application.controller.context_control
    before = control.read_context()
    rows = control.get_alias_rows()
    application.result("control.release", {})
    error = rpc_error(application, method, params)
    assert error["reason"] == "not_holder"
    assert control.read_context().document() == before.document()
    assert control.get_alias_rows() == rows


def seed_library(application: PointApplication) -> None:
    """Install a saved point with both library families and an occupied target."""
    application.open_entry()
    application.result("point.new", {"label": "base"})
    control = application.controller.context_control
    target = control.read_context().target
    for name, number in (("seed", 0.2), ("occupied", 0.3)):
        control.set_module(
            target,
            f"Q1.pulses.{name}",
            PulseSpec(
                ch=0, nqz=1, freq=100.0, gain=number, waveform="Q1.waveforms.seed"
            ),
        )
        control.set_waveform(
            target, f"Q1.waveforms.{name}", ConstWaveformCfg(length=number)
        )


@pytest.mark.parametrize("kind", ["module", "waveform"])
@pytest.mark.parametrize("action", ["del", "rename"])
def test_library_structure_rpc_commits_only_selected_entry(
    application: PointApplication, kind: Kind, action: str
) -> None:
    seed_library(application)
    control = application.controller.context_control
    before = control.read_context()
    partition = "pulses" if kind == "module" else "waveforms"
    path = f"Q1.{partition}.seed"
    new_path = f"Q1.{partition}.moved"
    other_kind: Kind = "waveform" if kind == "module" else "module"
    other_path = "Q1.waveforms.seed" if kind == "module" else "Q1.pulses.seed"
    value = before.library.entry_value(path, kind=kind)
    params = {"name": path} if action == "del" else {"old": path, "new": new_path}

    reply = application.rpc(f"context.ml_{action}_{kind}", params)
    assert reply["ok"] is True, reply
    assert reply["result"] == (
        {"deleted": path} if action == "del" else {"renamed": new_path}
    )
    after = control.read_context()
    assert after.target == before.target
    with pytest.raises(KeyError):
        after.library.entry_value(path, kind=kind)
    if action == "rename":
        assert after.library.entry_value(new_path, kind=kind) == value
    assert after.library.entry_value(other_path, kind=other_kind) == (
        before.library.entry_value(other_path, kind=other_kind)
    )
    # Both the old capture and a fresh persisted-target read retain their contract.
    assert before.library.entry_value(path, kind=kind) == value
    assert (
        control.read_target(after.target).library.document() == after.library.document()
    )


@pytest.mark.parametrize("kind", ["module", "waveform"])
@pytest.mark.parametrize("action", ["del", "rename"])
@pytest.mark.parametrize("source", ["seed", "Q1.pulses.missing"])
def test_library_structure_rpc_preserves_owner_path_rejection(
    application: PointApplication, kind: Kind, action: str, source: str
) -> None:
    seed_library(application)
    control = application.controller.context_control
    before = control.read_context()
    path = source.replace("pulses", "waveforms") if kind == "waveform" else source
    partition = "pulses" if kind == "module" else "waveforms"
    params = (
        {"name": path}
        if action == "del"
        else {"old": path, "new": f"Q1.{partition}.moved"}
    )

    reply = application.rpc(f"context.ml_{action}_{kind}", params)
    assert reply["ok"] is False, reply
    error = reply["error"]
    assert isinstance(error, dict)
    # Resource ValueError/KeyError remain unexpected, not InvalidInputError.
    assert error["code"] == ErrorCode.CONTROLLER_ERROR
    assert error["message"] == (
        "Invalid library root path: 'seed'" if source == "seed" else "'missing'"
    ), error
    assert control.read_context().target == before.target
    assert control.read_context().library.document() == before.library.document()
    assert (
        control.read_target(before.target).library.document()
        == before.library.document()
    )


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_library_rename_rpc_preserves_owner_collision_rejection(
    application: PointApplication, kind: Kind
) -> None:
    seed_library(application)
    control = application.controller.context_control
    before = control.read_context()
    partition = "pulses" if kind == "module" else "waveforms"
    reply = application.rpc(
        f"context.ml_rename_{kind}",
        {"old": f"Q1.{partition}.seed", "new": f"Q1.{partition}.occupied"},
    )
    assert reply["ok"] is False, reply
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.CONTROLLER_ERROR
    assert "Rename destination already exists" in str(error["message"])
    assert control.read_context().library.document() == before.library.document()
    assert (
        control.read_target(before.target).library.document()
        == before.library.document()
    )


def test_context_ml_and_value_handlers_use_context_control_facet() -> None:
    ctx = MagicMock()
    ctx.list_value_sources.return_value = (
        ValueInfo("predictor.loaded", bool, "predictor", "Whether loaded."),
    )
    ctx.read_value_source.return_value = (
        ValueInfo("predictor.loaded", bool, "predictor", "Whether loaded."),
        True,
    )
    ctrl = SimpleNamespace(context_control=ctx, resources_versions=lambda: {})

    assert _dispatch(ctrl, "value.list", {}) == {
        "values": [
            {
                "key": "predictor.loaded",
                "type": "bool",
                "owner": "predictor",
                "description": "Whether loaded.",
            }
        ]
    }
    assert _dispatch(ctrl, "value.read", {"key": "predictor.loaded"}) == {
        "key": "predictor.loaded",
        "type": "bool",
        "owner": "predictor",
        "description": "Whether loaded.",
        "value": True,
    }


@pytest.mark.parametrize(
    "params",
    [
        {"path": "general.description"},
        {"path": "general.description", "value": None},
        {"path": "", "value": "new"},
        {"path": 12, "value": "new"},
    ],
)
def test_parameter_set_rejects_malformed_payload_before_effects(
    application: PointApplication, params: dict[str, object]
) -> None:
    info = application.open_entry()
    application.result("point.new", {"label": "base"})
    before = application.controller.context_control.read_context()
    error = rpc_error(application, "context.set_parameter", params)
    assert error["code"] == ErrorCode.INVALID_PARAMS
    assert snapshot(entry_resource(info).use_point("base")) == before.document()
    assert entry_resource(info).ledger.events() == ()


def test_parameter_set_preserves_schema_failure_and_old_point(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    control = application.controller.context_control
    control.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    application.result("point.new", {"label": "base"})
    before = control.read_context()
    error = rpc_error(
        application, "context.set_parameter", {"path": "Q1.rate", "value": "invalid"}
    )
    assert error["code"] == ErrorCode.CONTROLLER_ERROR
    assert "validation error" in str(error["message"])
    assert control.read_context().document() == before.document()
    assert snapshot(entry_resource(info).use_point("base")) == before.document()


@pytest.mark.parametrize(
    "method,params",
    [
        ("context.aliases_set", {"rows": [{"alias": "driver", "value": "missing"}]}),
        ("context.set_parameter", {"path": "general.description", "value": "new"}),
    ],
)
def test_context_direct_write_admission_allows_active_run_without_changing_capture(
    application: PointApplication, method: str, params: dict[str, object]
) -> None:
    application.open_entry()
    application.result("point.new", {"label": "base"})
    control = application.controller.context_control
    before = control.read_context()
    tab_id = application.controller.new_tab("pointed")
    application.state.set_tab_running(tab_id, running=True)
    try:
        reply = application.rpc(method, params)
        assert reply["ok"] is True, reply
        assert before.document().general.description == ""
        assert before.aliases == ()
        assert application.state.is_run_active()
    finally:
        application.state.set_tab_running(tab_id, running=False)
