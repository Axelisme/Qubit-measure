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


def test_begin_reuses_session_and_finish_publishes_alignment(controller):
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
    unsubscribe = controller.bus.subscribe(
        SpectrumChangedPayload, lambda event: changes.append(event.name)
    )
    result = owner.finish_line_pick()
    unsubscribe()
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
