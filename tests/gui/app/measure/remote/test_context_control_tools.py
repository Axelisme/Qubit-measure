"""Context/value remote dispatch through real point-local library owners."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.remote.errors import ErrorCode
from zcu_tools.gui.session.value_lookup import ValueInfo

from ._helpers import dispatch_handler as _dispatch
from ._point_application import PointApplication

Kind = Literal["module", "waveform"]


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
