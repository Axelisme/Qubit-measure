"""Shared line-picking behavior through actions, commands and owner delivery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickInputs,
    FluxPickState,
    analyze_flux_pick,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.interactive.flux_pick import SharedFluxPickPlugin
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


@pytest.fixture
def plugin():
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.asarray(
        np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30)), dtype=np.complex128
    )
    inputs = FluxPickInputs(signals, devs, freqs)
    return SharedFluxPickPlugin(
        inputs,
        FluxPickState(flux_half=0.0, flux_int=2.0),
        build_result=lambda state: analyze_flux_pick(inputs, state),
    )


def test_actions_commands_and_undo_share_one_state(plugin):
    session = plugin.open(ManualOwnerScheduler())
    plugin.actions.move.execute(session, ("half", 0.5))
    after_gui = session.snapshot()
    plugin.execute_command(session, "move_line", {"role": "integer", "position": 3.0})
    assert session.snapshot().flux_half == after_gui.flux_half
    assert session.snapshot().flux_int == 3.0
    assert session.undo() == after_gui
    with pytest.raises(FailedPreconditionError):
        session.undo()
    plugin.execute_command(session, "set_conjugate", {"enabled": True})
    plugin.execute_command(session, "move_line", {"role": "half", "position": 1.0})
    assert session.snapshot().flux_int == 2.5
    plugin.execute_command(session, "swap_lines", {})
    result = plugin.finish(session)
    assert (result.flux_half, result.flux_int, result.flux_period) == (2.5, 1.0, 3.0)
    with pytest.raises(FailedPreconditionError):
        plugin.execute_command(session, "swap_lines", {})


@pytest.mark.parametrize(
    "params",
    [
        {"role": "unknown", "position": 1.0},
        {"role": "half", "position": 2.0},
        {"role": "half", "position": float("nan")},
        {"role": "half", "position": True},
    ],
)
def test_invalid_move_preserves_state_and_undo(plugin, params):
    session = plugin.open(ManualOwnerScheduler())
    start = session.snapshot()
    plugin.actions.move.execute(session, ("half", 0.5))
    committed = session.snapshot()
    with pytest.raises(InvalidInputError):
        plugin.execute_command(session, "move_line", params)
    assert session.snapshot() == committed
    assert session.undo() == start


def test_unbound_alignment_fails_without_leaving_busy(plugin):
    session = plugin.open(ManualOwnerScheduler())
    start = session.snapshot()
    with pytest.raises(FailedPreconditionError, match="background"):
        plugin.start_alignment(session)
    assert plugin.alignment_busy is False
    assert session.snapshot() == start


def test_single_flight_owner_delivery_and_terminal_late_result(plugin):
    owner = ManualOwnerScheduler()
    session = plugin.open(owner)
    deliveries: list[Callable[[], None]] = []
    computed_values: list[object] = []

    def submit(compute, on_done, on_error):
        value = compute()
        computed_values.append(value)
        deliveries.append(lambda: on_done(value))

    plugin.bind_background(submit)
    plugin.start_alignment(session)
    assert plugin.alignment_busy
    with pytest.raises(FailedPreconditionError, match="already"):
        plugin.start_alignment(session)
    start = session.snapshot()
    assert start == plugin.seed
    calculated = computed_values.pop()
    assert isinstance(calculated, FluxPickState)
    calculated_positions = (calculated.flux_half, calculated.flux_int)
    assert calculated_positions != (start.flux_half, start.flux_int)

    plugin.actions.move.execute(session, ("half", 1.0))
    plugin.execute_command(session, "set_conjugate", {"enabled": True})
    session.commit(lambda state: replace(state, magnitude_only=True))
    latest = session.snapshot()
    assert latest == FluxPickState(
        flux_half=1.0, flux_int=2.0, conjugate=True, magnitude_only=True
    )
    assert plugin.alignment_busy
    owner.post(deliveries.pop())
    assert session.snapshot() == latest
    owner.pump_all()
    aligned = session.snapshot()
    assert (aligned.flux_half, aligned.flux_int) == calculated_positions
    assert aligned.conjugate is latest.conjugate
    assert aligned.magnitude_only is latest.magnitude_only
    assert not plugin.alignment_busy
    assert session.undo() == latest
    plugin.start_alignment(session)
    plugin.finish(session)
    committed = session.snapshot()
    owner.post(deliveries.pop())
    owner.pump_all()
    assert session.snapshot() == committed
    assert not plugin.alignment_busy


def test_submission_failure_recovers_input(plugin):
    session = plugin.open(ManualOwnerScheduler())

    def fail_submission(compute, on_done, on_error):
        raise RuntimeError("pool unavailable")

    plugin.bind_background(fail_submission)
    with pytest.raises(RuntimeError, match="pool unavailable"):
        plugin.start_alignment(session)
    assert not plugin.alignment_busy
    assert plugin.info()["alignment_error"] == "pool unavailable"
    plugin.actions.move.execute(session, ("half", 1.0))
    assert session.snapshot().flux_half == 1.0
