"""Native interactive commands, retained receipts and source guards via route."""

from __future__ import annotations

import base64
import copy
import io
from collections.abc import Mapping

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image
from zcu_tools.analysis.fluxdep.cross_selection import project_cross_selection
from zcu_tools.analysis.fluxdep.twotone import project_twotone_pick
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import (
    InteractiveChangedPayload,
    SpectrumChangedPayload,
)
from zcu_tools.gui.app.fluxdep.interactive import (
    CrossSelectionContext,
    OneTonePickContext,
    TwoTonePickContext,
)
from zcu_tools.gui.app.fluxdep.remote import interactive
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.onetone import OneToneWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import EventMeta
from zcu_tools.gui.plotting.figure_export import render_figure_png
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.plotting.fluxdep.cross_selection import make_cross_selection_figure
from zcu_tools.plotting.fluxdep.twotone import make_twotone_pick_figure

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness, RouteReply


def _copy_sources(harness: RouteHarness, ctrl: Controller) -> RouteHarness:
    for entry in ctrl.state.spectrums.values():
        harness.ctrl.state.put_spectrum(copy.deepcopy(entry))
    harness.ctrl.set_active_spectrum(ctrl.state.active_spectrum)
    return harness


@pytest.fixture
def one_route(
    route_harness: RouteHarness, onetone_controller: Controller
) -> RouteHarness:
    return _copy_sources(route_harness, onetone_controller)


@pytest.fixture
def two_route(
    route_harness: RouteHarness, twotone_controller: Controller
) -> RouteHarness:
    return _copy_sources(route_harness, twotone_controller)


@pytest.fixture
def joint_route(
    route_harness: RouteHarness, cross_controller: Controller
) -> RouteHarness:
    return _copy_sources(route_harness, cross_controller)


def _context(reply: RouteReply) -> dict[str, object]:
    assert reply["ok"] is True, reply
    context = reply["result"]["context"]
    assert isinstance(context, dict)
    return context


def _state(context: Mapping[str, object]) -> dict[str, object]:
    state = context["state"]
    assert isinstance(state, dict)
    return state


def _changes(reply: RouteReply) -> dict[str, object]:
    assert reply["ok"] is True, reply
    effect = reply["result"]["effect"]
    assert isinstance(effect, dict)
    changes = effect["changes"]
    assert isinstance(changes, dict)
    return changes


def _image(context: Mapping[str, object]) -> np.ndarray:
    png = context["figure"]
    assert isinstance(png, dict)
    encoded = png["png_b64"]
    assert isinstance(encoded, str)
    data = base64.b64decode(encoded, validate=True)
    assert png["bytes"] == len(data)
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG"
        assert image.size == (640, 480)
        return np.array(image)


def _native_image(figure: Figure) -> np.ndarray:
    with Image.open(io.BytesIO(render_figure_png(figure))) as image:
        return np.array(image)


def _error(reply: RouteReply, code: str, reason: str | None) -> None:
    assert reply["ok"] is False, reply
    assert reply["error"]["code"] == code
    assert reply["error"].get("reason") == reason


def _read_sources(harness: RouteHarness, link: ClientLink) -> None:
    assert harness.request(link, "spectrum.list")["ok"]
    for name in harness.ctrl.state.spectrums:
        assert harness.request(link, "spectrum.snapshot", name=name)["ok"]
    assert harness.request(link, "selection.snapshot")["ok"]


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


def test_inactive_read_is_inert_and_live_read_does_not_unlock_or_consume_undo(
    one_route: RouteHarness,
):
    link = one_route.client()
    facts = []
    one_route.ctrl.bus.subscribe(InteractiveChangedPayload, facts.append)
    versions = one_route.ctrl.state.version.snapshot()
    reply = one_route.request(link, "interactive.read")
    assert reply["ok"] is True
    assert reply["result"] == {"context": None, "effect": None}
    assert one_route.ctrl.interactive.inspect() is None
    assert not facts
    assert one_route.ctrl.state.version.snapshot() == versions
    native = one_route.ctrl.interactive.begin_onetone_pick("one")
    native.plugin.set_threshold.execute(native.session, 0.1)
    prior_facts = len(facts)
    context = _context(one_route.request(link, "interactive.read"))
    assert context["can_undo"] is True
    assert _state(context)["threshold"] == 0.1
    _image(context)
    assert native.session.can_undo()
    assert len(facts) == prior_facts
    _error(
        _picker_command(one_route, link, context, "undo"),
        "precondition_failed",
        "stale_version",
    )
    assert native.session.can_undo()


def test_line_commands_share_gui_actions_and_feedback_undo(one_route: RouteHarness):
    link = one_route.client()
    facts: list[tuple[InteractiveChangedPayload, EventMeta]] = []
    one_route.ctrl.bus.subscribe_with_meta(
        InteractiveChangedPayload, lambda fact, meta: facts.append((fact, meta))
    )
    initial = _open_picker(one_route, link, "one", "line")
    assert initial["kind"] == "line" and initial["plugin"] == "flux_pick"
    assert _state(initial)["magnitude_only"] is True
    assert initial["info"] == {"alignment_busy": False, "alignment_error": None}
    commands = initial["commands"]
    assert isinstance(commands, list)
    move = next(item for item in commands if item["name"] == "move_line")
    assert move["schema"]["properties"]["role"]["enum"] == ["half", "integer"]
    again = _open_picker(one_route, link, "one", "line")
    assert again["context_id"] == initial["context_id"]
    native = one_route.ctrl.interactive.current_line_pick()
    assert native is not None
    native.plugin.actions.move.execute(native.session, ("half", 0.1))
    edited = _context(one_route.request(link, "interactive.read"))
    assert _state(edited)["flux_half"] == 0.1
    moved = _picker_command(
        one_route, link, initial, "move_line", role="integer", position=0.9
    )
    assert _state(_context(moved))["flux_int"] == 0.9
    assert _changes(moved)["flux_int_before"] == 0.7
    assert _changes(moved)["flux_int_after"] == 0.9
    undone = _picker_command(one_route, link, initial, "undo")
    assert _state(_context(undone))["flux_int"] == 0.7
    assert _context(undone)["can_undo"] is False
    assert _changes(undone)["flux_int_before"] == 0.9
    assert _changes(undone)["flux_int_after"] == 0.7
    _image(_context(undone))
    assert facts[-1][1].origin.kind == "agent"
    assert facts[-1][0].context_id == initial["context_id"]
    finished = _picker_command(one_route, link, initial, "finish")
    terminal = _context(finished)
    assert (
        terminal["context_id"] == initial["context_id"] and terminal["closed"] is True
    )
    assert terminal["commands"] == [] and terminal["can_undo"] is False
    assert one_route.ctrl.state.spectrums["one"].flux_half == 0.1
    assert one_route.ctrl.state.spectrums["one"].flux_int == 0.7
    assert one_route.ctrl.interactive.inspect() is None
    _image(terminal)
    # Successful publication advanced this connection's previously read source.
    assert one_route.request(
        link, "spectrum.interactive.open", name="one", kind="onetone"
    )["ok"]


def test_onetone_threshold_counts_undo_and_empty_finish(one_route: RouteHarness):
    link = one_route.client()
    context = _open_picker(one_route, link, "one", "onetone")
    low = _picker_command(one_route, link, context, "set_threshold", threshold=0.1)
    peaks = _state(_context(low))["peak_indices"]
    assert isinstance(peaks, list) and len(peaks) == 2
    high = _picker_command(one_route, link, context, "set_threshold", threshold=5)
    assert _state(_context(high))["peak_indices"] == []
    assert _changes(high)["peaks_removed"] == 2
    assert _changes(high)["peaks_added"] == 0
    inverse = _picker_command(one_route, link, context, "undo")
    assert _state(_context(inverse))["peak_indices"] == peaks
    assert _changes(inverse)["peaks_added"] == 2
    assert _changes(inverse)["peaks_removed"] == 0
    assert _picker_command(one_route, link, context, "set_threshold", threshold=5)["ok"]
    receipt = _context(_picker_command(one_route, link, context, "finish"))
    assert receipt["closed"] is True
    entry = one_route.ctrl.state.spectrums["one"]
    assert entry.points_completed and entry.point_count == 0
    _image(receipt)


def test_twotone_brush_numeric_and_png_use_same_capture_and_undo_inverse(
    two_route: RouteHarness,
):
    link = two_route.client()
    context = _open_picker(two_route, link, "two", "twotone")
    native = two_route.ctrl.interactive.current_twotone_pick()
    assert isinstance(native, TwoTonePickContext)
    before = native.session.snapshot()
    stroke = _picker_command(
        two_route,
        link,
        context,
        "stroke",
        vertices=[[-1.0, 4.5], [1.0, 5.1]],
        width=0.08,
        mode="erase",
    )
    after = native.session.snapshot()
    expected = project_twotone_pick(native.plugin.inputs, after, previous=before)
    assert expected.mask_removed > 0
    assert _changes(stroke) == {
        "mask_added": expected.mask_added,
        "mask_removed": expected.mask_removed,
        "points_added": expected.added_points.shape[0],
        "points_removed": expected.removed_points.shape[0],
    }
    state = _state(_context(stroke))
    assert state["mask_shape"] == [24, 80]
    assert state["masked_count"] == np.count_nonzero(after.mask)
    assert state["point_count"] == expected.result.dev_values.size
    np.testing.assert_array_equal(
        _image(_context(stroke)),
        _native_image(
            make_twotone_pick_figure(
                native.plugin.inputs,
                after,
                previous=before,
                show_changes=True,
                show_mask=True,
            )
        ),
    )
    undo = _picker_command(two_route, link, context, "undo")
    inverse = project_twotone_pick(native.plugin.inputs, before, previous=after)
    assert _changes(undo) == {
        "mask_added": inverse.mask_added,
        "mask_removed": inverse.mask_removed,
        "points_added": inverse.added_points.shape[0],
        "points_removed": inverse.removed_points.shape[0],
    }
    np.testing.assert_array_equal(
        _image(_context(undo)),
        _native_image(
            make_twotone_pick_figure(
                native.plugin.inputs,
                before,
                previous=after,
                show_changes=True,
                show_mask=True,
            )
        ),
    )
    settings = _picker_command(
        two_route, link, context, "set_tool", mode="erase", width=0.04
    )
    assert _state(_context(settings))["width"] == 0.04
    assert _changes(settings) == {
        "mask_added": 0,
        "mask_removed": 0,
        "points_added": 0,
        "points_removed": 0,
    }
    assert _picker_command(two_route, link, context, "clear")["ok"]
    terminal = _context(_picker_command(two_route, link, context, "finish"))
    assert terminal["closed"] and _state(terminal)["point_count"] == 0
    assert two_route.ctrl.state.spectrums["two"].points_completed
    assert two_route.ctrl.state.spectrums["two"].point_count == 0


def test_selection_stroke_duplicate_identities_apply_and_undo(
    joint_route: RouteHarness,
):
    link = joint_route.client()
    _read_sources(joint_route, link)
    context = _context(joint_route.request(link, "selection.interactive.open"))
    assert context["kind"] == "selection" and context["spectrum_name"] is None
    again = _context(joint_route.request(link, "selection.interactive.open"))
    assert again["context_id"] == context["context_id"]
    native = joint_route.ctrl.interactive.current_cross_selection()
    assert isinstance(native, CrossSelectionContext)
    before = native.session.snapshot()
    stroke = _joint_command(
        joint_route,
        link,
        context,
        "stroke",
        vertices=[[0.5, 4.5]],
        width=0,
        mode="erase",
    )
    after = native.session.snapshot()
    assert _changes(stroke) == {"points_added": 0, "points_removed": 2}
    assert _state(_context(stroke))["selected"] == after.selected.tolist()
    np.testing.assert_array_equal(
        _image(_context(stroke)),
        _native_image(
            make_cross_selection_figure(
                native.plugin.inputs, after, previous=before, show_changes=True
            )
        ),
    )
    applied = _context(_joint_command(joint_route, link, context, "apply"))
    assert applied["closed"] is False and applied["can_undo"] is True
    np.testing.assert_array_equal(
        joint_route.ctrl.state.selection.selected, after.selected
    )
    assert joint_route.ctrl.interactive.current_cross_selection() is native
    undo = _joint_command(joint_route, link, context, "undo")
    assert _changes(undo) == {"points_added": 2, "points_removed": 0}
    assert _state(_context(undo))["selected"] == before.selected.tolist()
    assert _joint_command(joint_route, link, context, "apply")["ok"]
    np.testing.assert_array_equal(
        joint_route.ctrl.state.selection.selected, before.selected
    )
    closed = _context(_joint_command(joint_route, link, context, "cancel"))
    assert closed["closed"] and closed["commands"] == []
    assert joint_route.ctrl.interactive.inspect() is None


@pytest.mark.parametrize(
    "method,values",
    [
        ("spectrum.interactive.open", {"name": "one", "kind": "line"}),
        (
            "spectrum.interactive.command",
            {"name": "one", "context_id": 1, "command": "undo"},
        ),
    ],
)
def test_picker_sources_are_per_client_and_gui_publication_requires_reread(
    one_route: RouteHarness, method: str, values: dict[str, object]
):
    link, other = one_route.client(), one_route.client()
    one_route.request(link, "spectrum.snapshot", name="one")
    assert one_route.request(other, "interactive.read")["ok"]
    _error(
        one_route.request(other, method, **values),
        "precondition_failed",
        "stale_version",
    )
    one_route.ctrl.set_alignment("one", 0.1, 0.6)
    _error(
        one_route.request(link, method, **values),
        "precondition_failed",
        "stale_version",
    )
    one_route.request(link, "spectrum.snapshot", name="one")
    assert one_route.request(
        link, "spectrum.interactive.open", name="one", kind="line"
    )["ok"]


@pytest.mark.parametrize("missing", ["spectrums:__set__", "selection", "empty"])
def test_joint_all_source_guard_includes_zero_point_spectrum(
    joint_route: RouteHarness, missing: str
):
    link = joint_route.client()
    if missing != "spectrums:__set__":
        joint_route.request(link, "spectrum.list")
    if missing != "selection":
        joint_route.request(link, "selection.snapshot")
    for name in joint_route.ctrl.state.spectrums:
        if name != missing:
            joint_route.request(link, "spectrum.snapshot", name=name)
    reply = joint_route.request(link, "selection.interactive.open")
    _error(reply, "precondition_failed", "stale_version")
    assert joint_route.ctrl.interactive.inspect() is None
    _read_sources(joint_route, link)
    context = _context(joint_route.request(link, "selection.interactive.open"))
    old_id = context["context_id"]
    # Zero-point source publication retires the input and leaves seen stale.
    joint_route.ctrl.reset_points("empty")
    _error(
        _joint_command(joint_route, link, context, "clear"),
        "precondition_failed",
        "stale_version",
    )
    _read_sources(joint_route, link)
    new = _context(joint_route.request(link, "selection.interactive.open"))
    assert new["context_id"] != old_id
    _error(
        _joint_command(joint_route, link, context, "clear"),
        "precondition_failed",
        "interactive_context_changed",
    )


@pytest.mark.parametrize(
    "case",
    [
        "identity",
        "name",
        "selection_target",
        "apply",
        "reserved_params",
        "unknown",
        "bad_domain",
    ],
)
def test_invalid_commands_do_not_change_snapshot_undo_versions_or_facts(
    one_route: RouteHarness, case: str
):
    link = one_route.client()
    context = _open_picker(one_route, link, "one", "onetone")
    native = one_route.ctrl.interactive.current_onetone_pick()
    assert isinstance(native, OneTonePickContext)
    native.plugin.set_threshold.execute(native.session, 0.1)
    before = native.session.snapshot()
    versions = one_route.ctrl.state.version.snapshot()
    facts = []
    one_route.ctrl.bus.subscribe(InteractiveChangedPayload, facts.append)
    values: dict[str, object] = {
        "name": "one",
        "context_id": context["context_id"],
        "command": "undo",
    }
    code = "precondition_failed"
    reason: str | None = "interactive_context_changed"
    method = "spectrum.interactive.command"
    if case == "identity":
        values["context_id"] = 999
    elif case == "name":
        values["name"] = "other"
        one_route.request(link, "spectrum.snapshot", name="other")
    elif case == "selection_target":
        _read_sources(one_route, link)
        method = "selection.interactive.command"
        values.pop("name")
    elif case == "apply":
        values["command"] = "apply"
        code, reason = "invalid_params", "invalid_interactive_command"
    elif case == "reserved_params":
        values["params"] = {"extra": 1}
        code, reason = "invalid_params", "invalid_interactive_command"
    elif case == "unknown":
        values["command"] = "not_a_command"
        code, reason = "invalid_params", None
    else:
        values.update(command="set_threshold", params={"threshold": 99})
        code, reason = "invalid_params", None
    _error(one_route.request(link, method, **values), code, reason)
    assert native.session.snapshot() == before
    assert native.session.can_undo()
    assert one_route.ctrl.state.version.snapshot() == versions
    assert not facts


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, "1"])
def test_context_id_admission_and_no_live_context(one_route: RouteHarness, bad: object):
    link = one_route.client()
    one_route.request(link, "spectrum.snapshot", name="one")
    values = {"name": "one", "context_id": bad, "command": "undo"}
    if isinstance(bad, bool) or not isinstance(bad, int):
        with pytest.raises(RemoteError):
            one_route.request(link, "spectrum.interactive.command", **values)
    else:
        _error(
            one_route.request(link, "spectrum.interactive.command", **values),
            "invalid_params",
            "invalid_interactive_command",
        )
    _error(
        one_route.request(
            link,
            "spectrum.interactive.command",
            name="one",
            context_id=1,
            command="undo",
        ),
        "precondition_failed",
        "no_interactive_context",
    )


def test_terminal_reply_retains_old_identity_despite_synchronous_successor(
    one_route: RouteHarness,
):
    link = one_route.client()
    old = _open_picker(one_route, link, "one", "line")
    successors = []

    def open_next(fact: SpectrumChangedPayload) -> None:
        successors.append(one_route.ctrl.interactive.begin_onetone_pick(fact.name))

    unsubscribe = one_route.ctrl.bus.subscribe(SpectrumChangedPayload, open_next)
    reply = _picker_command(one_route, link, old, "finish")
    unsubscribe.unsubscribe()
    terminal = _context(reply)
    assert successors and terminal["context_id"] == old["context_id"]
    assert terminal["kind"] == "line" and terminal["closed"] is True
    assert reply["ok"] is True
    assert reply["result"]["effect"] == {
        "command": "finish",
        "closed": True,
        "changes": _changes(reply),
    }
    next_context = _context(one_route.request(link, "interactive.read"))
    assert next_context["context_id"] != terminal["context_id"]
    assert next_context["kind"] == "onetone" and next_context["closed"] is False
    _image(terminal)


def test_render_failure_does_not_rollback_successful_finish_or_refresh_seen(
    one_route: RouteHarness, monkeypatch: pytest.MonkeyPatch
):
    link = one_route.client()
    context = _open_picker(one_route, link, "one", "onetone")

    def fail(figure: Figure) -> bytes:
        raise OSError("PNG unavailable")

    monkeypatch.setattr(interactive, "render_figure_png", fail)
    reply = _picker_command(one_route, link, context, "finish")
    assert reply["ok"] is False
    assert reply["error"]["code"] == "controller_error"
    assert one_route.ctrl.state.spectrums["one"].points_completed
    assert one_route.ctrl.interactive.inspect() is None
    _error(
        one_route.request(
            link, "spectrum.interactive.open", name="one", kind="onetone"
        ),
        "precondition_failed",
        "stale_version",
    )


def test_route_agent_facts_drive_native_gui_and_reads_do_not_switch(
    one_route: RouteHarness, qapp
):
    window = MainWindow(one_route.ctrl)
    try:
        link = one_route.client()
        context = _open_picker(one_route, link, "one", "line")
        assert window.findChild(LinePickerWidget) is not None
        reply = _picker_command(one_route, link, context, "finish")
        terminal = _context(reply)
        assert terminal["closed"] and terminal["kind"] == "line"
        widget = window.findChild(OneToneWidget)
        assert widget is not None
        current = _context(one_route.request(link, "interactive.read"))
        assert (
            current["kind"] == "onetone"
            and current["context_id"] != context["context_id"]
        )
        assert window.findChild(OneToneWidget) is widget
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_literal_picker_name_does_not_guard_same_prefix_neighbor(
    one_route: RouteHarness,
):
    name = "量測:literal*"
    for new_name in (name, name + "neighbor"):
        entry = copy.deepcopy(one_route.ctrl.state.spectrums["one"])
        entry.name = new_name
        one_route.ctrl.state.put_spectrum(entry)
    one_route.ctrl.set_active_spectrum(name)
    link = one_route.client()
    context = _open_picker(one_route, link, name, "onetone")
    one_route.ctrl.reset_points(name + "neighbor")
    receipt = _picker_command(one_route, link, context, "set_threshold", threshold=0.1)
    assert _context(receipt)["spectrum_name"] == name
    assert _state(_context(receipt))["peak_indices"]
    assert (
        _context(_picker_command(one_route, link, context, "cancel"))["closed"] is True
    )
    assert not one_route.ctrl.state.spectrums[name].points_completed


@pytest.mark.parametrize("condition", ["inactive", "unaligned", "wrong_type", "absent"])
def test_open_uses_native_prerequisites_without_changing_active(
    one_route: RouteHarness, condition: str
):
    name, kind = "one", "onetone"
    if condition == "inactive":
        one_route.ctrl.set_active_spectrum(None)
    elif condition == "unaligned":
        one_route.ctrl.reset_alignment(name)
    elif condition == "wrong_type":
        kind = "twotone"
    else:
        name = "absent"
    link = one_route.client()
    one_route.request(link, "spectrum.snapshot", name=name)
    versions = one_route.ctrl.state.version.snapshot()
    active = one_route.ctrl.state.active_spectrum
    reply = one_route.request(link, "spectrum.interactive.open", name=name, kind=kind)
    _error(
        reply,
        "invalid_params" if condition == "absent" else "precondition_failed",
        None,
    )
    assert one_route.ctrl.state.active_spectrum == active
    assert one_route.ctrl.state.version.snapshot() == versions
    assert one_route.ctrl.interactive.inspect() is None


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"extra": 1},
        {"threshold": True},
        {"threshold": float("inf")},
        {"threshold": float("nan")},
        {"threshold": 0.2, "extra": 1},
    ],
)
def test_domain_parameter_validation_is_owned_by_plugin_and_atomic(
    one_route: RouteHarness, params: dict[str, object]
):
    link = one_route.client()
    context = _open_picker(one_route, link, "one", "onetone")
    native = one_route.ctrl.interactive.current_onetone_pick()
    assert native is not None
    native.plugin.set_threshold.execute(native.session, 0.1)
    before = native.session.snapshot()
    _error(
        _picker_command(one_route, link, context, "set_threshold", **params),
        "invalid_params",
        None,
    )
    assert native.session.snapshot() == before
    assert native.session.can_undo()


@pytest.mark.parametrize("values", [{"kind": "invalid"}, {"name": ""}, {"kind": None}])
def test_open_envelope_rejects_malformed_values_before_begin(
    one_route: RouteHarness, values: dict[str, object]
):
    link = one_route.client()
    params: dict[str, object] = {"name": "one", "kind": "line"}
    params.update(values)
    with pytest.raises(RemoteError):
        one_route.request(link, "spectrum.interactive.open", **params)
    assert one_route.ctrl.interactive.inspect() is None


@pytest.mark.parametrize(
    "command,params", [("finish", {}), ("apply", {"extra": 1}), ("undo", {"extra": 1})]
)
def test_joint_reserved_commands_reject_wrong_kind_and_params(
    joint_route: RouteHarness, command: str, params: dict[str, object]
):
    link = joint_route.client()
    _read_sources(joint_route, link)
    context = _context(joint_route.request(link, "selection.interactive.open"))
    native = joint_route.ctrl.interactive.current_cross_selection()
    assert native is not None
    before = native.session.snapshot()
    versions = joint_route.ctrl.state.version.snapshot()
    _error(
        _joint_command(joint_route, link, context, command, **params),
        "invalid_params",
        "invalid_interactive_command",
    )
    np.testing.assert_array_equal(native.session.snapshot().selected, before.selected)
    assert joint_route.ctrl.state.version.snapshot() == versions
    assert not native.session.can_undo()


def test_command_receipt_survives_updated_callback_retirement(one_route: RouteHarness):
    link = one_route.client()
    context = _open_picker(one_route, link, "one", "onetone")

    def replace_input(fact: InteractiveChangedPayload) -> None:
        if fact.phase == "updated" and fact.kind == "onetone":
            one_route.ctrl.interactive.begin_line_pick("one")

    sub = one_route.ctrl.bus.subscribe(InteractiveChangedPayload, replace_input)
    receipt = _context(
        _picker_command(one_route, link, context, "set_threshold", threshold=0.1)
    )
    sub.unsubscribe()
    assert receipt["context_id"] == context["context_id"]
    assert receipt["kind"] == "onetone" and receipt["closed"] is True
    assert receipt["can_undo"] is False and receipt["commands"] == []
    assert _state(receipt)["threshold"] == 0.1
    successor = _context(one_route.request(link, "interactive.read"))
    assert (
        successor["kind"] == "line" and successor["context_id"] != receipt["context_id"]
    )


def test_failed_finish_keeps_native_input_and_undo_editable(one_route: RouteHarness):
    link = one_route.client()
    entry = copy.deepcopy(one_route.ctrl.state.spectrums["one"])
    entry.flux_int = entry.flux_half
    one_route.ctrl.state.put_spectrum(entry)
    context = _open_picker(one_route, link, "one", "line")
    assert _picker_command(one_route, link, context, "set_conjugate", enabled=True)[
        "ok"
    ]
    versions = one_route.ctrl.state.version.snapshot()
    _error(
        _picker_command(one_route, link, context, "finish"), "precondition_failed", None
    )
    live = _context(one_route.request(link, "interactive.read"))
    assert live["context_id"] == context["context_id"] and live["closed"] is False
    assert live["can_undo"] is True
    assert one_route.ctrl.state.version.snapshot() == versions
    assert _picker_command(one_route, link, context, "undo")["ok"]


def test_joint_png_read_does_not_refresh_other_clients_and_apply_self_write(
    joint_route: RouteHarness,
):
    link, other = joint_route.client(), joint_route.client()
    _read_sources(joint_route, link)
    context = _context(joint_route.request(link, "selection.interactive.open"))
    _image(_context(joint_route.request(other, "interactive.read")))
    _error(
        _joint_command(joint_route, other, context, "clear"),
        "precondition_failed",
        "stale_version",
    )
    _read_sources(joint_route, other)
    assert _joint_command(joint_route, link, context, "clear")["ok"]
    assert _joint_command(joint_route, link, context, "apply")["ok"]
    _error(
        _joint_command(joint_route, other, context, "undo"),
        "precondition_failed",
        "stale_version",
    )
    assert _joint_command(joint_route, link, context, "undo")["ok"]
    assert _joint_command(joint_route, link, context, "apply")["ok"]
    assert np.all(joint_route.ctrl.state.selection.selected)


def test_detector_setting_reports_actual_point_changes(two_route: RouteHarness):
    link = two_route.client()
    context = _open_picker(two_route, link, "two", "twotone")
    native = two_route.ctrl.interactive.current_twotone_pick()
    assert native is not None
    before = native.session.snapshot()
    reply = _picker_command(two_route, link, context, "set_settings", threshold=20)
    after = native.session.snapshot()
    view = project_twotone_pick(native.plugin.inputs, after, previous=before)
    assert view.removed_points.shape[0] > 0
    assert _changes(reply) == {
        "mask_added": 0,
        "mask_removed": 0,
        "points_added": view.added_points.shape[0],
        "points_removed": view.removed_points.shape[0],
    }
    assert _state(_context(reply))["point_count"] == view.result.dev_values.size


def test_distance_changes_kept_count_without_rewriting_input_brush_mask(
    joint_route: RouteHarness,
):
    # Same-x groups are preserved natively. Use a nearby different-x point to
    # observe downsampling rather than assuming duplicates must disappear.
    nearby = copy.deepcopy(joint_route.ctrl.state.spectrums["b"])
    nearby.points["fluxs"] = np.array([0.51])
    nearby.points["dev_values"] = np.array([0.51])
    nearby.points["freqs"] = np.array([4.51])
    joint_route.ctrl.state.put_spectrum(nearby)
    link = joint_route.client()
    _read_sources(joint_route, link)
    context = _context(joint_route.request(link, "selection.interactive.open"))
    native = joint_route.ctrl.interactive.current_cross_selection()
    assert native is not None
    before = native.session.snapshot()
    reply = _joint_command(
        joint_route, link, context, "set_min_distance", min_distance=0.05
    )
    after = native.session.snapshot()
    view = project_cross_selection(native.plugin.inputs, after, previous=before)
    assert view.removed_points.shape[0] > 0
    assert _changes(reply) == {
        "points_added": view.added_points.shape[0],
        "points_removed": view.removed_points.shape[0],
    }
    state = _state(_context(reply))
    assert state["selected"] == before.selected.tolist()
    assert state["selected_count"] == np.count_nonzero(view.result.selected)
    count = state["selected_count"]
    assert isinstance(count, int)
    assert count < len(before.selected)
    np.testing.assert_array_equal(
        _image(_context(reply)),
        _native_image(
            make_cross_selection_figure(
                native.plugin.inputs, after, previous=before, show_changes=True
            )
        ),
    )
