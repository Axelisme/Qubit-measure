"""One-tone commands, results and undo through public plugin/Session seams."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs
from zcu_tools.analysis.fluxdep.onetone import (
    OneToneInputs,
    analyze_onetone_pick,
    detect_peaks,
)
from zcu_tools.gui.app.fluxdep.onetone import OneTonePickPlugin
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


@pytest.fixture
def plugin(onetone_controller):
    raw = onetone_controller.state.spectrums["one"].raw
    return OneTonePickPlugin(
        OneToneInputs(FluxPickInputs(raw["signals"], raw["dev_values"], raw["freqs"]))
    )


def test_threshold_action_command_result_and_full_undo(plugin):
    session = plugin.open(ManualOwnerScheduler())
    seed = session.snapshot()
    low = plugin.set_threshold.execute(session, 0.1)
    assert low.threshold == 0.1
    assert low.peak_indices == tuple(
        int(i) for i in detect_peaks(plugin.inputs.smoothed, 0.1)
    )
    low_points = analyze_onetone_pick(plugin.inputs, low)
    np.testing.assert_allclose(np.sort(low_points.dev_values), [0.25, 0.75], atol=0.05)
    np.testing.assert_array_equal(
        low_points.freqs,
        np.full(2, plugin.inputs.spectrum.freqs[plugin.inputs.max_freq_index]),
    )
    plugin.execute_command(session, "set_threshold", {"threshold": 5.0})
    high = session.snapshot()
    assert high.threshold == 5.0
    assert high.peak_indices != low.peak_indices
    assert session.undo() == low
    with pytest.raises(FailedPreconditionError):
        session.undo()
    assert seed.threshold == 1.0
    result = plugin.finish(session)
    np.testing.assert_array_equal(result.dev_values, low_points.dev_values)
    np.testing.assert_array_equal(result.freqs, low_points.freqs)
    with pytest.raises(FailedPreconditionError):
        plugin.set_threshold.execute(session, 1.0)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"threshold": True},
        {"threshold": None},
        {"threshold": "1"},
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"threshold": -0.1},
        {"threshold": 5.1},
        {"threshold": 1.0, "extra": 1},
    ],
)
def test_invalid_threshold_does_not_consume_undo(plugin, params):
    session = plugin.open(ManualOwnerScheduler())
    seed = session.snapshot()
    committed = plugin.set_threshold.execute(session, 0.1)
    with pytest.raises(InvalidInputError):
        plugin.execute_command(session, "set_threshold", params)
    assert session.snapshot() == committed
    assert session.undo() == seed


def test_invalid_typed_action_keeps_state_and_history(plugin):
    session = plugin.open(ManualOwnerScheduler())
    seed = session.snapshot()
    committed = plugin.set_threshold.execute(session, 0.1)
    with pytest.raises(InvalidInputError):
        plugin.set_threshold.execute(session, 5.1)
    assert session.snapshot() == committed
    assert session.undo() == seed


def test_empty_result_and_captured_inputs(plugin, onetone_controller):
    session = plugin.open(ManualOwnerScheduler())
    low = plugin.set_threshold.execute(session, 0.1)
    before = analyze_onetone_pick(plugin.inputs, low)
    onetone_controller.state.spectrums["one"].raw["signals"][:] = 0
    after = analyze_onetone_pick(plugin.inputs, session.snapshot())
    np.testing.assert_array_equal(after.dev_values, before.dev_values)
    plugin.execute_command(session, "set_threshold", {"threshold": 5.0})
    empty = plugin.finish(session)
    assert empty.dev_values.size == 0
    assert empty.freqs.size == 0
