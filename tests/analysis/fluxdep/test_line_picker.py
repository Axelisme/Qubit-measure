from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep import (
    FluxPickInputs,
    FluxPickState,
    TwoLinePicker,
    align_lines,
    find_best_mirror_position,
    fold_initial_lines,
    mirror_loss_at,
    move_line,
    swap_lines,
)


def _spectrum(n_dev: int = 60, n_freq: int = 30):
    devs = np.linspace(-5.0, 5.0, n_dev).astype(np.float64)
    freqs = np.linspace(4.0, 5.0, n_freq).astype(np.float64)
    sig = np.zeros((n_dev, n_freq), dtype=np.complex128)
    sig += np.exp(-(devs[:, None] ** 2) / (2 * 1.0**2))
    return sig, devs, freqs


def _make_picker(**kwargs) -> TwoLinePicker:
    sig, devs, freqs = _spectrum()
    fig = Figure()
    FigureCanvasAgg(fig)
    return TwoLinePicker(fig, sig, devs, freqs, **kwargs)


@pytest.mark.parametrize(
    ("half", "integer", "message"),
    [(float("nan"), 2.0, "flux_half"), (0.0, float("inf"), "flux_int")],
)
def test_flux_pick_state_rejects_nonfinite_positions(
    half: float, integer: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FluxPickState(flux_half=half, flux_int=integer)


def test_flux_pick_inputs_own_read_only_data_with_axis_separation() -> None:
    signals, devs, freqs = _spectrum()
    expected = complex(signals[0, 0])

    inputs = FluxPickInputs(signals, devs, freqs)
    signals[0, 0] += 5.0

    assert inputs.signals[0, 0] == expected
    assert not inputs.signals.flags.writeable
    assert inputs.min_distance == pytest.approx(0.1)


def test_flux_pick_inputs_reject_mismatched_signal_shape() -> None:
    signals, devs, freqs = _spectrum()
    with pytest.raises(ValueError, match="signals shape"):
        FluxPickInputs(signals[:-1], devs, freqs)


def test_mirror_loss_at_uses_valid_rows_of_symmetric_spectrum() -> None:
    signals, devs, _freqs = _spectrum()
    real = np.abs(signals)

    loss, mean = mirror_loss_at(devs, real, 0.0)

    assert loss.shape == real.shape
    assert mean == pytest.approx(0.0, abs=1e-10)


def test_align_lines_returns_pure_candidate_near_symmetric_spectrum_center() -> None:
    signals, devs, _freqs = _spectrum()
    state = FluxPickState(flux_half=0.1, flux_int=-0.1)

    candidate = align_lines(state, devs, np.abs(signals))

    assert abs(candidate.flux_half) < 0.1
    assert abs(candidate.flux_int) < 0.1
    assert (state.flux_half, state.flux_int) == (0.1, -0.1)


def test_swap_lines_preserves_settings_and_original_state() -> None:
    committed = FluxPickState(
        flux_half=1.0, flux_int=-2.0, conjugate=True, magnitude_only=True
    )

    candidate = swap_lines(committed)

    assert (candidate.flux_half, candidate.flux_int) == (-2.0, 1.0)
    assert candidate.conjugate and candidate.magnitude_only
    assert (committed.flux_half, committed.flux_int) == (1.0, -2.0)


def test_move_line_clamps_preview_without_mutating_committed_state() -> None:
    committed = FluxPickState(flux_half=0.0, flux_int=2.0)

    candidate = move_line(committed, "half", 1.95, min_distance=0.1)

    assert (candidate.flux_half, candidate.flux_int) == pytest.approx((1.9, 2.0))
    assert (committed.flux_half, committed.flux_int) == (0.0, 2.0)


def test_move_line_conjugate_preserves_separation() -> None:
    committed = FluxPickState(flux_half=0.0, flux_int=2.0, conjugate=True)

    candidate = move_line(committed, "integer", 3.0, min_distance=0.1)

    assert (candidate.flux_half, candidate.flux_int) == (1.0, 3.0)


@pytest.mark.parametrize(
    ("role", "position", "min_distance", "message"),
    [
        ("unknown", 1.0, 0.1, "role"),
        ("half", float("nan"), 0.1, "position"),
        ("half", 1.0, -0.1, "min_distance"),
    ],
)
def test_move_line_rejects_invalid_action(
    role: str, position: float, min_distance: float, message: str
) -> None:
    committed = FluxPickState(flux_half=0.0, flux_int=2.0)
    with pytest.raises(ValueError, match=message):
        move_line(committed, role, position, min_distance=min_distance)  # type: ignore[arg-type]
    assert (committed.flux_half, committed.flux_int) == (0.0, 2.0)


def test_fold_initial_lines_defaults() -> None:
    _sig, devs, _freqs = _spectrum()
    half, integer = fold_initial_lines(devs, None, None)
    assert devs[0] <= half <= devs[-1]
    assert devs[0] <= integer <= devs[-1]


def test_find_best_mirror_position_prefers_symmetric_center() -> None:
    n = 51
    devs = np.linspace(-5.0, 5.0, n, dtype=np.float64)
    center = float(devs[n // 2])
    col = np.abs(devs).reshape(n, 1).astype(np.float64)
    pos = find_best_mirror_position(devs, col, center, search_width=1.0)
    assert pos == pytest.approx(center, abs=0.5 * (devs[1] - devs[0]))


def test_picker_drag_swap_and_compute_apply() -> None:
    picker = _make_picker()
    half0, int0 = picker.positions()
    picker.on_press(half0)
    picker.on_move(half0 + 0.5)
    half1, int1 = picker.positions()
    assert half1 != half0
    assert int1 == int0

    picker.swap()
    assert picker.positions() == (int1, half1)

    before = picker.positions()
    aligned = picker.compute_aligned_positions()
    assert picker.positions() == before
    picker.apply_positions(*aligned)
    assert picker.positions() == aligned


def test_picker_magnitude_toggle() -> None:
    picker = _make_picker()
    assert picker.magnitude_only is False
    picker.set_magnitude_only(True)
    assert picker.magnitude_only is True


def test_picker_explicit_pick_hooks() -> None:
    picker = _make_picker()
    assert picker.selected_role is None
    picker.pick_half()
    assert picker.selected_role == "half"
    picker.pick_integer()
    assert picker.selected_role == "integer"
    picker.clear_selection()
    assert picker.selected_role is None


def test_picker_identifies_main_axes() -> None:
    signals, devs, freqs = _spectrum()
    figure = Figure()
    FigureCanvasAgg(figure)
    picker = TwoLinePicker(figure, signals, devs, freqs)
    main_axes, loss_axes = figure.axes

    assert picker.is_main_axes(main_axes)
    assert not picker.is_main_axes(loss_axes)
    assert not picker.is_main_axes(None)
