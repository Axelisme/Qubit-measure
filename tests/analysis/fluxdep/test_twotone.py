"""Captured TwoTone data and detached numerical result contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs
from zcu_tools.analysis.fluxdep.twotone import (
    TwoToneInputs,
    TwoTonePickChange,
    analyze_twotone_pick,
    fill_twotone_state,
    make_twotone_state,
    project_twotone_pick,
)


def test_inputs_capture_readonly_finite_matching_spectrum() -> None:
    devs = np.linspace(-1.0, 1.0, 8)
    freqs = np.linspace(4.0, 6.0, 40)
    profile = np.exp(-(((freqs - 5.0) / 0.1) ** 2))
    signals = np.asarray(
        np.tile(1.0 + 1j * profile, (devs.size, 1)), dtype=np.complex128
    )
    inputs = TwoToneInputs(FluxPickInputs(signals, devs, freqs))
    state = make_twotone_state(inputs)
    expected = analyze_twotone_pick(inputs, state)
    signals[:] = 0
    devs[:] = 99
    freqs[:] = 99
    actual = analyze_twotone_pick(inputs, state)
    np.testing.assert_array_equal(actual.dev_values, expected.dev_values)
    np.testing.assert_array_equal(actual.freqs, expected.freqs)
    assert actual.dev_values.size > 0
    assert not inputs.spectrum.signals.flags.writeable
    assert not inputs.spectrum.dev_values.flags.writeable
    assert not inputs.spectrum.freqs.flags.writeable
    state.mask[:] = False
    assert analyze_twotone_pick(inputs, state).dev_values.size == 0
    assert make_twotone_state(inputs).mask.all()


@pytest.mark.parametrize("invalid", ["shape", "nan", "inf"])
def test_inputs_reject_invalid_raw_data(invalid: str) -> None:
    devs = np.linspace(-1.0, 1.0, 8)
    freqs = np.linspace(4.0, 6.0, 40)
    signals = np.ones((devs.size, freqs.size), dtype=np.complex128)
    if invalid == "shape":
        signals = signals[:, :-1]
    else:
        signals[0, 0] = float(invalid)
    with pytest.raises(ValueError, match="signals"):
        TwoToneInputs(FluxPickInputs(signals, devs, freqs))


def test_projection_reports_both_sides_of_equal_count_position_change() -> None:
    devs = np.linspace(-1.0, 1.0, 8)
    freqs = np.linspace(4.0, 6.0, 100)
    profile = np.exp(-(((freqs - 4.4) / 0.08) ** 2)) + np.exp(
        -(((freqs - 5.6) / 0.08) ** 2)
    )
    signals = np.asarray(
        np.tile(1.0 + 1j * profile, (devs.size, 1)), dtype=np.complex128
    )
    inputs = TwoToneInputs(FluxPickInputs(signals, devs, freqs))
    seed = make_twotone_state(inputs, smooth_method="gaussian")
    before_mask = np.tile(freqs < 5.0, (devs.size, 1))
    after_mask = np.tile(freqs > 5.0, (devs.size, 1))
    state = replace(
        seed,
        mask=after_mask,
        last_change=TwoTonePickChange(
            before_mask, seed.threshold, seed.sigma, seed.smooth_method
        ),
    )
    view = project_twotone_pick(inputs, state)
    assert view.result.dev_values.size == devs.size
    assert view.added_points.shape == view.removed_points.shape == (devs.size, 2)
    assert np.all(view.added_points[:, 1] > 5.0)
    assert np.all(view.removed_points[:, 1] < 5.0)
    assert view.mask_added == view.mask_removed == after_mask.sum()
    np.testing.assert_array_equal(state.mask, after_mask)


def test_explicit_previous_snapshot_projects_inverse_undo_change() -> None:
    devs = np.linspace(-1.0, 1.0, 8)
    freqs = np.linspace(4.0, 6.0, 40)
    profile = np.exp(-(((freqs - 5.0) / 0.1) ** 2))
    signals = np.asarray(
        np.tile(1.0 + 1j * profile, (devs.size, 1)), dtype=np.complex128
    )
    inputs = TwoToneInputs(FluxPickInputs(signals, devs, freqs))
    restored = make_twotone_state(inputs)
    cleared = fill_twotone_state(inputs, restored, select=False)
    view = project_twotone_pick(inputs, restored, previous=cleared)
    assert view.state.last_change is None
    assert view.result.dev_values.size > 0
    assert view.added_points.shape[0] == view.result.dev_values.size
    assert view.removed_points.shape == (0, 2)
    assert view.mask_added == restored.mask.size
    assert view.mask_removed == 0
    assert view.stroke_vertices == ()
