"""Fluxdep context ownership and terminal publication through Controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import SpectrumChangedPayload
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
    assert entry.points_selected
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
    assert not onetone_controller.state.spectrums["one"].points_selected
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
    assert not onetone_controller.state.spectrums["one"].points_selected


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
        assert not ctrl.state.spectrums["one"].points_selected


def test_onetone_publication_failure_does_not_reopen_input(onetone_controller):
    from zcu_tools.gui.app.fluxdep.interactive import FluxDepInteractiveOwner
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
        background=None,
        publish_alignment=ctrl.set_alignment,
        publish_points=fail_points,
    )
    try:
        context = owner.begin_onetone_pick("one")
        with pytest.raises(OSError, match="publication unavailable"):
            owner.finish_onetone_pick()
        assert owner.current_onetone_pick() is None
        with pytest.raises(FailedPreconditionError):
            context.plugin.set_threshold.execute(context.session, 1.0)
        assert not ctrl.state.spectrums["one"].points_selected
    finally:
        owner.dispose()
