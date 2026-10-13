"""Explicit RPC target admission and independent named/global draft lifetimes."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.rpc_endpoint import ClientLink

from tests.gui.app.fluxdep.remote._route_harness import (
    ErrorReply,
    RouteHarness,
    RouteReply,
    copy_sources,
)


def _result(reply: RouteReply) -> dict[str, object]:
    assert reply["ok"] is True, reply
    return reply["result"]


def _context(reply: RouteReply) -> dict[str, object]:
    context = _result(reply)["context"]
    assert isinstance(context, dict)
    return context


def _error(reply: RouteReply, code: str, reason: str | None) -> ErrorReply:
    assert reply["ok"] is False, reply
    assert reply["error"]["code"] == code
    assert reply["error"].get("reason") == reason
    return reply["error"]


def _open_picker(
    harness: RouteHarness, link: ClientLink, name: str, kind: str
) -> dict[str, object]:
    assert harness.request(link, "spectrum.snapshot", name=name)["ok"]
    return _context(
        harness.request(link, "spectrum.interactive.open", name=name, kind=kind)
    )


def _picker_command(
    harness: RouteHarness,
    link: ClientLink,
    context: Mapping[str, object],
    command: str,
    **params: object,
) -> RouteReply:
    return harness.request(
        link,
        "spectrum.interactive.command",
        name=context["spectrum_name"],
        context_id=context["context_id"],
        command=command,
        params=params,
    )


def _joint_command(
    harness: RouteHarness,
    link: ClientLink,
    context: Mapping[str, object],
    command: str,
    **params: object,
) -> RouteReply:
    return harness.request(
        link,
        "selection.interactive.command",
        context_id=context["context_id"],
        command=command,
        params=params,
    )


def test_explicit_targets_keep_selection_name_and_filter_independent(
    route_harness: RouteHarness,
    selection_name_controller: Controller,
):
    harness = copy_sources(route_harness, selection_name_controller)
    link = harness.client()
    named = _open_picker(harness, link, "selection", "line")
    joint = _context(harness.request(link, "selection.interactive.open"))
    assert named["context_id"] != joint["context_id"]
    assert _picker_command(
        harness, link, named, "move_line", role="half", position=0.5
    )["ok"]
    assert _joint_command(harness, link, joint, "set_min_distance", min_distance=0.05)[
        "ok"
    ]
    observer = harness.client(mode="readonly", label="observer")
    versions = harness.ctrl.state.version.snapshot()
    active_name = harness.ctrl.state.active_spectrum
    for target, draft, extra in (
        ("spectrum", named, {"name": "selection"}),
        ("selection", joint, {}),
    ):
        read = _context(
            harness.request(observer, "interactive.read", target=target, **extra)
        )
        assert read["context_id"] == draft["context_id"]
        assert read["can_undo"] is True
        assert harness.request(observer, "interactive.render", target=target, **extra)[
            "ok"
        ]
    listed = _result(harness.request(observer, "spectrum.list"))
    rows = listed["spectrums"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    assert row["pending"] == "align"
    snapshot = _result(harness.request(observer, "spectrum.snapshot", name="selection"))
    assert snapshot["pending"] == "align"
    assert harness.ctrl.state.version.snapshot() == versions
    assert harness.ctrl.state.active_spectrum == active_name
    for target, wrong_id, extra in (
        ("spectrum", joint["context_id"], {"name": "selection"}),
        ("selection", named["context_id"], {}),
    ):
        _error(
            harness.request(
                observer,
                "interactive.read",
                target=target,
                context_id=wrong_id,
                **extra,
            ),
            "precondition_failed",
            "interactive_context_changed",
        )
    _error(
        _picker_command(harness, link, joint | {"spectrum_name": "selection"}, "undo"),
        "precondition_failed",
        "interactive_context_changed",
    )
    _error(
        _joint_command(harness, link, named, "undo"),
        "precondition_failed",
        "interactive_context_changed",
    )
    assert _joint_command(harness, link, joint, "cancel")["ok"]
    read_named = _context(
        harness.request(
            observer, "interactive.read", target="spectrum", name="selection"
        )
    )
    assert read_named["context_id"] == named["context_id"]
    sample = _open_picker(harness, link, "selection", "sample")
    assert sample["kind"] == "onetone"
    assert _picker_command(harness, link, sample, "set_threshold", threshold=0.2)["ok"]
    assert (
        _result(harness.request(observer, "spectrum.snapshot", name="selection"))[
            "pending"
        ]
        == "sample"
    )
    _error(
        harness.request(
            observer,
            "interactive.read",
            target="spectrum",
            name="selection",
            context_id=named["context_id"],
        ),
        "precondition_failed",
        "interactive_context_changed",
    )
    joint = _context(harness.request(link, "selection.interactive.open"))
    assert _picker_command(harness, link, sample, "cancel")["ok"]
    assert (
        _context(harness.request(observer, "interactive.read", target="selection"))[
            "context_id"
        ]
        == joint["context_id"]
    )
    assert (
        _result(harness.request(observer, "spectrum.snapshot", name="selection"))[
            "pending"
        ]
        is None
    )
    harness.ctrl.interactive.dispose()
    for target, extra in (("spectrum", {"name": "selection"}), ("selection", {})):
        assert (
            _result(
                harness.request(observer, "interactive.read", target=target, **extra)
            )["context"]
            is None
        )


@pytest.mark.parametrize("method", ["interactive.read", "interactive.render"])
def test_explicit_targets_reject_ambiguous_envelopes(
    joint_route: RouteHarness, method: str
):
    link = joint_route.client()
    _context(joint_route.request(link, "selection.interactive.open"))
    versions = joint_route.ctrl.state.version.snapshot()
    for params in (
        {"name": "selection"},
        {"target": "wrong", "name": "selection"},
        {"target": "selection", "name": "selection"},
        {"target": "selection", "name": None},
        {"target": "spectrum"},
        {"target": "spectrum", "name": None},
        {"target": "spectrum", "name": ""},
    ):
        try:
            reply = joint_route.request(link, method, **params)
        except RemoteError as error:
            assert error.code == ErrorCode.INVALID_PARAMS
        else:
            _error(reply, "invalid_params", None)
    assert joint_route.request(link, method, target="selection")["ok"]
    assert joint_route.ctrl.state.version.snapshot() == versions
