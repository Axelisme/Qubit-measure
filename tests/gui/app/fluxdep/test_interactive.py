"""Fluxdep context ownership and terminal publication through Controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import (
    InteractiveChangedPayload,
    SpectrumChangedPayload,
)
from zcu_tools.gui.app.fluxdep.interactive import LinePickContext
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpectrumEntry
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError


@pytest.fixture
def controller():
    state = FluxDepState()
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.asarray(
        np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30)), dtype=np.complex128
    )
    state.put_spectrum(
        SpectrumEntry(
            name="sample",
            spec_type="OneTone",
            raw={
                "signals": signals,
                "dev_values": devs,
                "freqs": freqs,
                "fluxs": devs.copy(),
            },
            points={
                "dev_values": np.empty(0),
                "freqs": np.empty(0),
                "fluxs": np.empty(0),
            },
        )
    )
    ctrl = Controller(state)
    ctrl.set_active_spectrum("sample")
    yield ctrl
    ctrl.interactive.dispose()


def test_inspect_identity_and_facts_follow_committed_session(controller: Controller):
    owner = controller.interactive
    facts: list[InteractiveChangedPayload] = []
    subscription = controller.bus.subscribe(InteractiveChangedPayload, facts.append)
    try:
        assert owner.inspect() is None
        assert not facts
        context = owner.begin_line_pick("sample")
        active = owner.inspect()
        assert active is not None
        assert active.context is context
        assert active.kind == "line"
        assert active.spectrum_name == "sample"
        assert active.context_id > 0
        assert [fact.phase for fact in facts] == ["opened"]

        assert owner.begin_line_pick("sample") is context
        assert owner.inspect() == active
        assert [fact.phase for fact in facts] == ["opened"]
        context.plugin.actions.swap.execute(context.session, None)
        context.session.undo()
        assert [fact.phase for fact in facts] == ["opened", "updated", "updated"]

        with pytest.raises(InvalidInputError, match="unknown"):
            context.plugin.execute_command(context.session, "unknown", {})
        assert len(facts) == 3
        owner.cancel()
        assert owner.inspect() is None
        assert facts[-1].phase == "closed"
        assert {fact.context_id for fact in facts} == {active.context_id}
        owner.cancel()
        assert len(facts) == 4

        replacement = owner.begin_line_pick("sample")
        new = owner.inspect()
        assert new is not None and new.context is replacement
        assert new.context_id > active.context_id
        with pytest.raises(FailedPreconditionError, match="closed"):
            context.plugin.actions.swap.execute(context.session, None)
        assert facts[-1].phase == "opened"
    finally:
        subscription.unsubscribe()


def test_inspect_exposes_onetone_and_selection_without_consuming_undo(
    controller: Controller, cross_controller: Controller
):
    controller.set_alignment("sample", 0.0, 2.0)
    one = controller.interactive.begin_onetone_pick("sample")
    active = controller.interactive.inspect()
    assert active is not None and active.context is one
    assert active.kind == "onetone" and active.spectrum_name == "sample"
    one.plugin.execute_command(one.session, "set_threshold", {"threshold": 0.5})
    assert one.session.can_undo()
    assert controller.interactive.inspect() == active
    assert one.session.can_undo()
    controller.interactive.finish_onetone_pick()
    assert controller.interactive.inspect() is None

    selection = cross_controller.interactive.begin_cross_selection()
    joint = cross_controller.interactive.inspect()
    assert joint is not None and joint.context is selection
    assert joint.kind == "selection" and joint.spectrum_name is None
    selection.plugin.clear.execute(selection.session, None)
    cross_controller.interactive.apply_cross_selection()
    assert cross_controller.interactive.inspect() == joint
    assert selection.session.can_undo()
    cross_controller.set_points("b", np.array([0.8]), np.array([4.8]))
    assert cross_controller.interactive.inspect() is None


def test_alignment_updates_stop_after_context_retirement(controller: Controller):
    owner = controller.interactive
    context = owner.begin_line_pick("sample")
    facts: list[InteractiveChangedPayload] = []
    subscription = controller.bus.subscribe(InteractiveChangedPayload, facts.append)
    deliveries: list[Callable[[], None]] = []

    def submit(compute, on_done, on_error):
        value = compute()
        deliveries.append(lambda: on_done(value))

    context.plugin.bind_background(submit)
    try:
        context.plugin.start_alignment(context.session)
        assert context.plugin.alignment_busy
        assert facts[-1].phase == "updated"
        owner.cancel()
        count = len(facts)
        deliveries.pop()()
        assert len(facts) == count
        assert owner.inspect() is None
    finally:
        subscription.unsubscribe()


def test_inspect_tracks_kind_switch_and_terminal_facts(twotone_controller):
    owner = twotone_controller.interactive
    line = owner.begin_line_pick("two")
    old = owner.inspect()
    assert old is not None
    facts: list[InteractiveChangedPayload] = []
    subscription = twotone_controller.bus.subscribe(
        InteractiveChangedPayload, facts.append
    )
    try:
        picked = owner.begin_twotone_pick("two")
        active = owner.inspect()
        assert active is not None and active.context is picked
        assert active.kind == "twotone"
        assert active.context_id > old.context_id
        assert [fact.phase for fact in facts] == ["closed", "opened"]
        with pytest.raises(FailedPreconditionError):
            line.session.ensure_input_open()
        owner.finish_twotone_pick()
        assert owner.inspect() is None
        assert facts[-1].phase == "closed"
        assert facts[-1].context_id == active.context_id
        assert twotone_controller.state.spectrums["two"].points_completed
    finally:
        subscription.unsubscribe()


@pytest.mark.parametrize("kind", ["line", "onetone", "twotone", "selection"])
def test_closed_callback_context_is_not_overwritten(
    controller: Controller, kind: Literal["line", "onetone", "twotone", "selection"]
):
    if kind == "twotone":
        controller.state.spectrums["sample"].spec_type = "TwoTone"
    controller.set_alignment("sample", -1.0, 1.0)
    controller.set_points("sample", np.array([-0.5, 0.5]), np.array([4.2, 4.3]))
    owner = controller.interactive
    old = (
        owner.begin_line_pick("sample")
        if kind == "selection"
        else owner.begin_cross_selection()
    )
    reentered: list[LinePickContext] = []
    facts: list[InteractiveChangedPayload] = []

    def on_fact(fact: InteractiveChangedPayload) -> None:
        facts.append(fact)
        if fact.phase == "closed" and not reentered:
            reentered.append(owner.begin_line_pick("sample"))

    begin: Callable[[], object] = {
        "line": lambda: owner.begin_line_pick("sample"),
        "onetone": lambda: owner.begin_onetone_pick("sample"),
        "twotone": lambda: owner.begin_twotone_pick("sample"),
        "selection": owner.begin_cross_selection,
    }[kind]
    subscription = controller.bus.subscribe(InteractiveChangedPayload, on_fact)
    try:
        with pytest.raises(FailedPreconditionError, match="retirement"):
            begin()
        active = owner.inspect()
        assert active is not None and active.context is reentered[0]
        with pytest.raises(FailedPreconditionError, match="closed"):
            old.session.ensure_input_open()
        current = reentered[0]
        current.plugin.actions.swap.execute(current.session, None)
        assert [fact.phase for fact in facts] == ["closed", "opened", "updated"]
        assert facts[-1].context_id == active.context_id
        owner.cancel()
        assert owner.inspect() is None
        assert facts[-1].phase == "closed"
        assert facts[-1].context_id == active.context_id
        with pytest.raises(FailedPreconditionError, match="closed"):
            current.plugin.actions.swap.execute(current.session, None)
    finally:
        subscription.unsubscribe()


def test_dispose_rejects_begin_from_closed_callback(controller: Controller):
    owner = controller.interactive
    old = owner.begin_line_pick("sample")
    facts: list[InteractiveChangedPayload] = []

    def on_fact(fact: InteractiveChangedPayload) -> None:
        facts.append(fact)
        if fact.phase == "closed":
            with pytest.raises(FailedPreconditionError, match="disposed"):
                owner.begin_line_pick("sample")
            owner.dispose()

    subscription = controller.bus.subscribe(InteractiveChangedPayload, on_fact)
    try:
        owner.dispose()
        assert owner.inspect() is None
        assert [fact.phase for fact in facts] == ["closed"]
        with pytest.raises(FailedPreconditionError, match="closed"):
            old.plugin.actions.swap.execute(old.session, None)
        owner.dispose()
        assert len(facts) == 1
    finally:
        subscription.unsubscribe()


def test_begin_reuses_session_and_finish_publishes_alignment(controller: Controller):
    owner = controller.interactive
    ctx = owner.begin_line_pick("sample")
    assert owner.begin_line_pick("sample") is ctx
    assert ctx.session.snapshot().magnitude_only
    ctx.plugin.execute_command(
        ctx.session, "move_line", {"role": "half", "position": 0.5}
    )
    ctx.plugin.execute_command(
        ctx.session, "move_line", {"role": "integer", "position": 2.0}
    )
    changes: list[str] = []
    subscription = controller.bus.subscribe(
        SpectrumChangedPayload, lambda event: changes.append(event.name)
    )
    result = owner.finish_line_pick()
    subscription.unsubscribe()
    assert result.flux_period == 3.0
    entry = controller.state.spectrums["sample"]
    assert entry.aligned
    assert (entry.flux_half, entry.flux_int) == (0.5, 2.0)
    np.testing.assert_allclose(
        entry.raw["fluxs"], (entry.raw["dev_values"] - 0.5) / 3.0 + 0.5
    )
    assert changes == ["sample"]
    assert owner.current_line_pick() is None
    with pytest.raises(FailedPreconditionError):
        ctx.plugin.actions.swap.execute(ctx.session, None)


def test_equal_seed_finish_failure_keeps_context_editable(controller):
    entry = controller.state.spectrums["sample"]
    controller.state.put_spectrum(
        replace(entry, alignment_seeded=True, flux_half=0.0, flux_int=0.0)
    )
    ctx = controller.interactive.begin_line_pick("sample")
    with pytest.raises(FailedPreconditionError, match="separat"):
        controller.interactive.finish_line_pick()
    assert controller.interactive.current_line_pick() is ctx
    assert not controller.state.spectrums["sample"].aligned
    ctx.plugin.actions.move.execute(ctx.session, ("half", 1.0))
    assert controller.interactive.finish_line_pick().flux_period == 2.0


@pytest.mark.parametrize("change", ["switch", "remove", "reload", "alignment"])
def test_spectrum_change_closes_old_input(controller, change):
    ctx = controller.interactive.begin_line_pick("sample")
    if change == "switch":
        controller.set_active_spectrum(None)
    elif change == "remove":
        controller.remove_spectrum("sample")
    elif change == "reload":
        controller.state.put_spectrum(replace(controller.state.spectrums["sample"]))
        from zcu_tools.gui.app.fluxdep.event_bus import SpectrumAddedPayload

        controller.bus.emit(SpectrumAddedPayload(name="sample"))
    else:
        controller.set_alignment("sample", 0.0, 2.0)
    assert controller.interactive.current_line_pick() is None
    with pytest.raises(FailedPreconditionError):
        ctx.plugin.actions.swap.execute(ctx.session, None)


def test_unknown_inactive_disposed_and_cancel_contract(controller):
    owner = controller.interactive
    with pytest.raises(InvalidInputError):
        owner.begin_line_pick("missing")
    controller.set_active_spectrum(None)
    with pytest.raises(FailedPreconditionError):
        owner.begin_line_pick("sample")
    controller.set_active_spectrum("sample")
    old = owner.begin_line_pick("sample")
    owner.cancel()
    new = owner.begin_line_pick("sample")
    assert new.session is not old.session
    assert not controller.state.spectrums["sample"].aligned
    owner.dispose()
    owner.dispose()
    with pytest.raises(FailedPreconditionError):
        owner.begin_line_pick("sample")


def test_cancelled_worker_cannot_publish_into_replacement_context(controller):
    owner = controller.interactive
    old = owner.begin_line_pick("sample")
    deliveries: list[Callable[[], None]] = []

    def submit(compute, on_done, on_error):
        value = compute()
        deliveries.append(lambda: on_done(value))

    old.plugin.bind_background(submit)
    old.plugin.start_alignment(old.session)
    owner.cancel()
    new = owner.begin_line_pick("sample")
    before = new.session.snapshot()
    deliveries.pop()()
    assert new.session.snapshot() == before
    assert not controller.state.spectrums["sample"].aligned


def test_twotone_reuse_kind_switch_and_terminal_publication(twotone_controller):
    from zcu_tools.analysis.fluxdep.twotone import analyze_twotone_pick

    ctrl = twotone_controller
    owner = ctrl.interactive
    line = owner.begin_line_pick("two")
    context = owner.begin_twotone_pick("two")
    assert owner.current_line_pick() is None
    assert owner.begin_twotone_pick("two") is context
    with pytest.raises(FailedPreconditionError):
        line.session.undo()
    expected = analyze_twotone_pick(context.plugin.inputs, context.session.snapshot())
    assert expected.dev_values.size > 0
    changes: list[str] = []
    subscription = ctrl.bus.subscribe(
        SpectrumChangedPayload, lambda event: changes.append(event.name)
    )
    try:
        result = owner.finish_twotone_pick()
    finally:
        subscription.unsubscribe()
    np.testing.assert_array_equal(result.dev_values, expected.dev_values)
    entry = ctrl.state.spectrums["two"]
    assert entry.points_completed
    np.testing.assert_array_equal(entry.points["dev_values"], expected.dev_values)
    np.testing.assert_array_equal(entry.points["freqs"], expected.freqs)
    np.testing.assert_allclose(
        entry.points["fluxs"],
        (entry.points["dev_values"] - entry.flux_half) / entry.flux_period + 0.5,
    )
    assert changes == ["two"]
    assert owner.current_twotone_pick() is None
    with pytest.raises(FailedPreconditionError):
        context.plugin.clear.execute(context.session, None)
    with pytest.raises(FailedPreconditionError):
        owner.finish_twotone_pick()


def test_invalid_twotone_finish_keeps_context_editable(twotone_controller):
    owner = twotone_controller.interactive
    context = owner.begin_twotone_pick("two")
    context.session.commit(
        lambda state: replace(state, mask=np.ones((1, 1), dtype=bool))
    )
    with pytest.raises(ValueError, match="mask"):
        owner.finish_twotone_pick()
    assert owner.current_twotone_pick() is context
    assert not twotone_controller.state.spectrums["two"].points_completed
    context.session.undo()
    context.plugin.clear.execute(context.session, None)
    assert owner.finish_twotone_pick().dev_values.size == 0


def test_line_begin_closes_previous_twotone_input(twotone_controller):
    owner = twotone_controller.interactive
    old = owner.begin_twotone_pick("two")
    line = owner.begin_line_pick("two")
    assert owner.current_line_pick() is line
    assert owner.current_twotone_pick() is None
    with pytest.raises(FailedPreconditionError):
        old.plugin.clear.execute(old.session, None)


@pytest.mark.parametrize(
    "condition", ["missing", "inactive", "unaligned", "type", "disposed"]
)
def test_twotone_begin_prerequisites(twotone_controller, condition):
    ctrl = twotone_controller
    name = "two"
    error = FailedPreconditionError
    if condition == "missing":
        name = "missing"
        error = InvalidInputError
    elif condition == "inactive":
        ctrl.set_active_spectrum(None)
    elif condition == "unaligned":
        ctrl.state.put_spectrum(replace(ctrl.state.spectrums[name], aligned=False))
    elif condition == "type":
        ctrl.state.put_spectrum(
            replace(ctrl.state.spectrums[name], spec_type="OneTone")
        )
    else:
        ctrl.interactive.dispose()
    with pytest.raises(error):
        ctrl.interactive.begin_twotone_pick(name)


@pytest.mark.parametrize(
    "change", ["switch", "remove", "reload", "alignment", "cancel", "dispose"]
)
def test_twotone_invalidation_closes_input_without_points(twotone_controller, change):
    from zcu_tools.gui.app.fluxdep.event_bus import SpectrumAddedPayload

    ctrl = twotone_controller
    owner = ctrl.interactive
    context = owner.begin_twotone_pick("two")
    if change == "switch":
        ctrl.set_active_spectrum(None)
    elif change == "remove":
        ctrl.remove_spectrum("two")
    elif change == "reload":
        ctrl.state.put_spectrum(replace(ctrl.state.spectrums["two"]))
        ctrl.bus.emit(SpectrumAddedPayload(name="two"))
    elif change == "alignment":
        ctrl.set_alignment("two", 0.1, 0.6)
    elif change == "cancel":
        owner.cancel()
    else:
        owner.dispose()
    assert owner.current_twotone_pick() is None
    with pytest.raises(FailedPreconditionError):
        context.plugin.clear.execute(context.session, None)
    if "two" in ctrl.state.spectrums:
        assert not ctrl.state.spectrums["two"].points_completed


def test_twotone_publication_failure_does_not_reopen_input(twotone_controller):
    from zcu_tools.gui.app.fluxdep.interactive import (
        FluxDepInteractiveOwner,
        FluxDepInteractivePorts,
    )
    from zcu_tools.gui.session.adapters.manual_owner_scheduler import (
        ManualOwnerScheduler,
    )

    ctrl = twotone_controller

    def fail_points(name, devs, freqs):
        raise OSError("publication unavailable")

    owner = FluxDepInteractiveOwner(
        ctrl.state,
        ctrl.bus,
        ManualOwnerScheduler(),
        ports=FluxDepInteractivePorts(
            background=None,
            publish_alignment=ctrl.set_alignment,
            publish_points=fail_points,
            derive_pointcloud=ctrl.derive_pointcloud,
            publish_selection=ctrl.set_selection,
        ),
    )
    try:
        context = owner.begin_twotone_pick("two")
        with pytest.raises(OSError, match="publication unavailable"):
            owner.finish_twotone_pick()
        assert owner.current_twotone_pick() is None
        with pytest.raises(FailedPreconditionError):
            context.plugin.clear.execute(context.session, None)
    finally:
        owner.dispose()


def test_onetone_reuse_kind_switch_and_terminal_publication(onetone_controller):
    ctrl = onetone_controller
    owner = ctrl.interactive
    line = owner.begin_line_pick("one")
    context = owner.begin_onetone_pick("one")
    assert owner.current_line_pick() is None
    assert owner.begin_onetone_pick("one") is context
    with pytest.raises(FailedPreconditionError):
        line.plugin.actions.swap.execute(line.session, None)
    context.plugin.execute_command(context.session, "set_threshold", {"threshold": 0.1})
    from zcu_tools.analysis.fluxdep.onetone import analyze_onetone_pick

    expected = analyze_onetone_pick(context.plugin.inputs, context.session.snapshot())
    assert expected.dev_values.size == 2
    assert np.all(np.diff(expected.dev_values) < 0)
    changes: list[str] = []
    subscription = ctrl.bus.subscribe(
        SpectrumChangedPayload, lambda event: changes.append(event.name)
    )
    try:
        result = owner.finish_onetone_pick()
    finally:
        subscription.unsubscribe()
    np.testing.assert_array_equal(result.dev_values, expected.dev_values)
    np.testing.assert_array_equal(result.freqs, expected.freqs)
    entry = ctrl.state.spectrums["one"]
    assert entry.points_completed
    np.testing.assert_array_equal(
        entry.points["dev_values"], np.sort(expected.dev_values)
    )
    np.testing.assert_allclose(
        entry.points["fluxs"],
        (entry.points["dev_values"] - entry.flux_half) / entry.flux_period + 0.5,
    )
    np.testing.assert_array_equal(
        entry.points["freqs"], expected.freqs[np.argsort(expected.dev_values)]
    )
    assert changes == ["one"]
    assert owner.current_onetone_pick() is None
    with pytest.raises(FailedPreconditionError):
        context.plugin.set_threshold.execute(context.session, 1.0)
    with pytest.raises(FailedPreconditionError):
        owner.finish_onetone_pick()


def test_invalid_onetone_finish_keeps_context_editable(onetone_controller):
    from zcu_tools.analysis.fluxdep.onetone import OneTonePickState

    owner = onetone_controller.interactive
    context = owner.begin_onetone_pick("one")
    size = context.plugin.inputs.spectrum.dev_values.size
    context.session.commit(
        lambda state: OneTonePickState(threshold=state.threshold, peak_indices=(size,))
    )
    with pytest.raises(
        ValueError, match="peak_indices must be within the captured device axis"
    ):
        owner.finish_onetone_pick()
    assert owner.current_onetone_pick() is context
    assert not onetone_controller.state.spectrums["one"].points_completed
    context.plugin.set_threshold.execute(context.session, 0.1)
    assert owner.finish_onetone_pick().dev_values.size == 2


def test_line_begin_closes_previous_onetone_input(onetone_controller):
    owner = onetone_controller.interactive
    old = owner.begin_onetone_pick("one")
    line = owner.begin_line_pick("one")
    assert owner.current_line_pick() is line
    assert owner.current_onetone_pick() is None
    with pytest.raises(FailedPreconditionError):
        old.plugin.set_threshold.execute(old.session, 1.0)
    assert not onetone_controller.state.spectrums["one"].points_completed


@pytest.mark.parametrize(
    "condition", ["missing", "inactive", "unaligned", "type", "disposed"]
)
def test_onetone_begin_prerequisites(onetone_controller, condition):
    ctrl = onetone_controller
    name = "one"
    error = FailedPreconditionError
    if condition == "missing":
        name = "missing"
        error = InvalidInputError
    elif condition == "inactive":
        ctrl.set_active_spectrum(None)
    elif condition == "unaligned":
        ctrl.state.put_spectrum(replace(ctrl.state.spectrums[name], aligned=False))
    elif condition == "type":
        ctrl.state.put_spectrum(
            replace(ctrl.state.spectrums[name], spec_type="TwoTone")
        )
    else:
        ctrl.interactive.dispose()
    with pytest.raises(error):
        ctrl.interactive.begin_onetone_pick(name)


@pytest.mark.parametrize(
    "change", ["switch", "remove", "reload", "alignment", "cancel", "dispose"]
)
def test_onetone_invalidation_closes_input_without_points(onetone_controller, change):
    ctrl = onetone_controller
    owner = ctrl.interactive
    context = owner.begin_onetone_pick("one")
    if change == "switch":
        ctrl.set_active_spectrum(None)
    elif change == "remove":
        ctrl.remove_spectrum("one")
    elif change == "reload":
        ctrl.state.put_spectrum(replace(ctrl.state.spectrums["one"]))
        from zcu_tools.gui.app.fluxdep.event_bus import SpectrumAddedPayload

        ctrl.bus.emit(SpectrumAddedPayload(name="one"))
    elif change == "alignment":
        ctrl.set_alignment("one", 0.1, 0.6)
    elif change == "cancel":
        owner.cancel()
    else:
        owner.dispose()
    assert owner.current_onetone_pick() is None
    with pytest.raises(FailedPreconditionError):
        context.plugin.set_threshold.execute(context.session, 1.0)
    if "one" in ctrl.state.spectrums:
        assert not ctrl.state.spectrums["one"].points_completed


def test_onetone_publication_failure_does_not_reopen_input(onetone_controller):
    from zcu_tools.gui.app.fluxdep.interactive import (
        FluxDepInteractiveOwner,
        FluxDepInteractivePorts,
    )
    from zcu_tools.gui.session.adapters.manual_owner_scheduler import (
        ManualOwnerScheduler,
    )

    ctrl = onetone_controller

    def fail_points(name, devs, freqs):
        raise OSError("publication unavailable")

    owner = FluxDepInteractiveOwner(
        ctrl.state,
        ctrl.bus,
        ManualOwnerScheduler(),
        ports=FluxDepInteractivePorts(
            background=None,
            publish_alignment=ctrl.set_alignment,
            publish_points=fail_points,
            derive_pointcloud=ctrl.derive_pointcloud,
            publish_selection=ctrl.set_selection,
        ),
    )
    try:
        context = owner.begin_onetone_pick("one")
        with pytest.raises(OSError, match="publication unavailable"):
            owner.finish_onetone_pick()
        assert owner.current_onetone_pick() is None
        with pytest.raises(FailedPreconditionError):
            context.plugin.set_threshold.execute(context.session, 1.0)
        assert not ctrl.state.spectrums["one"].points_completed
    finally:
        owner.dispose()
