"""Full native parameter-context reads and per-connection observations."""

from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from zcu_tools.cfg_model import ArbWaveformCfg, PulseSpec
from zcu_tools.gui.remote.errors import ErrorCode
from zcu_tools.gui.session.context_types import ContextSourceFault
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.entry import ResultEntry

from ._helpers import open_client, recv_response, send
from ._point_application import PointApplication


def test_snapshot_returns_complete_native_documents_without_materializing_assets(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    context = application.controller.context_control
    context.add_component("Q1", "fake/drive/a", {"rate": 10.0}, setup=True)
    context.add_component("S1", "fake/sensor", {"rate": 20.0}, setup=True)
    entry = ResultEntry.open(
        info.name,
        result_root=info.result_path.parent,
        database_root=info.database_path.parent,
    )
    point = entry.new_point("base", description="native snapshot")
    point.general.ext = {"nested": {"trace": [1.0, 2.0], "unset": None}}
    library = entry.module_library("base")
    with library.edit() as draft:
        draft.set_waveform(
            "Q1.waveforms.arbitrary", ArbWaveformCfg(data="missing-asset")
        )
        draft.set_module(
            "Q1.pulses.pi",
            PulseSpec(
                ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q1.waveforms.arbitrary"
            ),
        )
    application.result("point.use", {"label": "base"})
    captured = context.read_context()
    result = application.result("context.snapshot", {})
    assert result == {
        "target": captured.target.model_dump(),
        "point": captured.document().model_dump(serialize_as_any=True),
        "library": captured.library.document().model_dump(serialize_as_any=True),
    }
    wire_point = result["point"]
    assert isinstance(wire_point, dict)
    components = wire_point["components"]
    assert isinstance(components, dict)
    q1 = components["Q1"]
    s1 = components["S1"]
    assert isinstance(q1, dict) and isinstance(s1, dict)
    assert q1["rate"] == 10.0 and s1["rate"] == 20.0
    assert wire_point["provenance"]
    wire_library = result["library"]
    assert isinstance(wire_library, dict)
    assert wire_library["provenance"]
    # The asset was never created: a full Spec read must still succeed.
    with pytest.raises(FileNotFoundError):
        captured.library.get_waveform("Q1.waveforms.arbitrary").length


def test_snapshot_reads_the_memory_draft_and_unlocks_only_its_connection(
    application: PointApplication,
) -> None:
    info = application.open_entry()
    context = application.controller.context_control
    context.add_component("Q1", "fake/drive/a", {"rate": 10.0})
    result = application.result("context.snapshot", {})
    assert result["target"] == {"kind": "entry_draft", "entry_id": info.entry_id}
    # Empty edits are rejected by the handler only after a valid context seen.
    params = {"kind": "module", "name": "unused", "edits": []}
    admitted = application.rpc("context.ml_edit", params)
    error = admitted["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.INVALID_PARAMS
    second = open_client(application.remote.port)
    try:
        send(second, {"id": "unseen", "method": "context.ml_edit", "params": params})
        unseen = recv_response(second, "unseen")["error"]
        assert isinstance(unseen, dict)
        assert unseen["reason"] == "stale_version"
        assert unseen["data"] == {"stale": ["context"]}
    finally:
        second.close()
    context.set_parameter(context.read_context().target, "Q1.rate", 11.0)
    stale = application.rpc("context.ml_edit", params)["error"]
    assert isinstance(stale, dict)
    assert stale["reason"] == "stale_version"
    application.result("context.snapshot", {})
    refreshed = application.rpc("context.ml_edit", params)["error"]
    assert isinstance(refreshed, dict)
    assert refreshed["code"] == ErrorCode.INVALID_PARAMS


def test_empty_snapshot_fails_without_unlocking_context(
    application: PointApplication,
) -> None:
    reply = application.rpc("context.snapshot", {})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "no_context"
    guarded = application.rpc(
        "context.ml_edit", {"kind": "module", "name": "x", "edits": []}
    )
    error = guarded["error"]
    assert isinstance(error, dict)
    assert error["reason"] == "stale_version"


@pytest.mark.parametrize("document", ["point", "module_cfg"])
def test_source_unavailable_snapshot_does_not_unlock_context(
    application: PointApplication,
    document: Literal["point", "module_cfg"],
    tmp_path: Path,
) -> None:
    application.open_entry()
    captured = application.controller.context_control.read_context()
    fault = ContextSourceFault(
        document, tmp_path / f"{document}.yaml", "OSError", "source fault"
    )
    unavailable = ParameterContext(
        point=captured.document(),
        library=captured.library,
        target=captured.target,
        roles=None,
        source_faults=(fault,),
    )
    application.state.set_context(
        replace(application.state.session_env, parameters=unavailable)
    )
    reply = application.rpc("context.snapshot", {})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "source_unavailable"
    guarded = application.rpc(
        "context.ml_edit", {"kind": "module", "name": "x", "edits": []}
    )
    error = guarded["error"]
    assert isinstance(error, dict)
    assert error["reason"] == "stale_version"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_snapshot_never_substitutes_null_or_unlocks_context(
    application: PointApplication, value: float
) -> None:
    application.open_entry()
    context = application.controller.context_control
    context.set_parameter(
        context.read_context().target, "general.ext.nested", {"values": [value]}
    )
    reply = application.rpc("context.snapshot", {})
    error = reply["error"]
    assert isinstance(error, dict)
    assert error["code"] == ErrorCode.PRECONDITION_FAILED
    assert error["reason"] == "unserializable_context"
    guarded = application.rpc(
        "context.ml_edit", {"kind": "module", "name": "x", "edits": []}
    )
    error = guarded["error"]
    assert isinstance(error, dict)
    assert error["reason"] == "stale_version"
