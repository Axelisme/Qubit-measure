"""Atomic terminal publication receipts through the native RPC seam."""

import copy
from collections.abc import Mapping

import pytest
from zcu_tools.gui.app.fluxdep.event_bus import SpectrumChangedPayload
from zcu_tools.gui.remote.rpc_endpoint import ClientLink

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply


def _context(reply: RouteReply) -> dict[str, object]:
    assert reply["ok"] is True, reply
    context = reply["result"]["context"]
    assert isinstance(context, dict)
    return context


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


@pytest.mark.parametrize("kind", ["line", "onetone"])
def test_finish_reports_transitive_publication_invalidated(
    one_route: RouteHarness,
    kind: str,
) -> None:
    link = one_route.client()
    context = _open_picker(one_route, link, "one", kind)
    finished = _picker_command(one_route, link, context, "finish")
    assert finished["ok"] and "result" in finished
    expected = ["filter mask of one", "fit"]
    if kind == "line":
        expected.insert(0, "points of one")
    assert finished["result"].get("invalidated") == expected


def test_twotone_finish_reports_filter_and_fit_invalidated(
    two_route: RouteHarness,
) -> None:
    link = two_route.client()
    context = _open_picker(two_route, link, "two", "twotone")
    finished = _picker_command(two_route, link, context, "finish")
    assert finished["ok"] and "result" in finished
    assert finished["result"].get("invalidated") == ["filter mask of two", "fit"]


def test_finish_invalidated_is_the_original_publication_despite_reentrant_changes(
    one_route: RouteHarness,
) -> None:
    neighbor = copy.deepcopy(one_route.ctrl.state.get_spectrum("one"))
    neighbor.name = "neighbor"
    one_route.ctrl.state.put_spectrum(neighbor)
    link = one_route.client()
    context = _open_picker(one_route, link, "one", "line")
    changed = False

    def publish_neighbor_and_reopen(payload: SpectrumChangedPayload) -> None:
        nonlocal changed
        if payload.name == "one" and not changed:
            changed = True
            one_route.ctrl.set_alignment("neighbor", 0.0, 1.0)
            one_route.ctrl.interactive.begin_line_pick("one")

    subscription = one_route.ctrl.bus.subscribe(
        SpectrumChangedPayload, publish_neighbor_and_reopen
    )
    try:
        finished = _picker_command(one_route, link, context, "finish")
    finally:
        subscription.unsubscribe()
    assert finished["ok"] and "result" in finished
    assert finished["result"].get("invalidated") == [
        "points of one",
        "filter mask of one",
        "fit",
    ]
    successor = one_route.ctrl.interactive.inspect("one")
    assert successor is not None
    assert successor.context_id != context["context_id"]
    assert _context(finished)["closed"] is True
