"""Direct electrical-delay branch search, alias alignment, and input contracts."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.analysis.fitting.resonance import (
    HangerModel,
    TransmissionModel,
    find_edelay_branch,
    fit_edelay,
)

from ._resonance_truth import (
    hanger_truth,
    homophasal_like_grid,
    transmission_truth,
)


@pytest.mark.parametrize("model", [HangerModel, TransmissionModel])
@pytest.mark.parametrize("descending", [False, True])
def test_fit_edelay_recovers_large_delay_branch_on_nonuniform_grid(
    model: type[HangerModel] | type[TransmissionModel],
    descending: bool,
) -> None:
    freq, Ql, edelay = 5549.0, 740.0, 11.299
    freqs = homophasal_like_grid(freq, Ql, 15.0, 301)
    if descending:
        freqs = freqs[::-1]
    if model is HangerModel:
        signals = hanger_truth(
            freqs,
            freq=freq,
            Ql=Ql,
            Qc_abs=1100.0,
            phi=0.12,
            a0=1.25 * np.exp(0.31j),
            edelay=edelay,
            bg_amp_slope=0.004,
        )
    else:
        signals = transmission_truth(
            freqs,
            freq=freq,
            Ql=Ql,
            a0=1.25 * np.exp(0.31j),
            edelay=edelay,
            bg_amp_slope=0.004,
        )

    np.testing.assert_allclose(fit_edelay(freqs, signals), edelay, atol=3e-3)


def test_fit_edelay_preserves_uniform_grid_alias_class() -> None:
    freq, Ql, edelay = 5549.0, 740.0, 11.299
    freqs = np.linspace(freq - 15.0, freq + 15.0, 301)
    signals = hanger_truth(
        freqs,
        freq=freq,
        Ql=Ql,
        Qc_abs=1100.0,
        phi=0.12,
        a0=1.25 * np.exp(0.31j),
        edelay=edelay,
        bg_amp_slope=0.004,
    )

    estimated = fit_edelay(freqs, signals)
    alias_period = 1.0 / float(freqs[1] - freqs[0])
    alias_error = (estimated - edelay + 0.5 * alias_period) % alias_period
    alias_error -= 0.5 * alias_period

    assert abs(alias_error) < 3e-3
    assert abs(estimated) < 0.5 * alias_period


def test_related_uniform_traces_share_alias_across_branch_cut() -> None:
    edelay = 4.997
    freqs = np.linspace(5000.0, 5030.0, 301)
    signals = np.asarray(
        [
            hanger_truth(
                freqs,
                freq=freq,
                Ql=740.0,
                Qc_abs=1100.0,
                phi=0.12,
                a0=1.25 * np.exp(0.31j),
                edelay=edelay,
                bg_amp_slope=0.004,
            )
            for freq in (5008.0, 5013.0)
        ]
    )

    branch_seed = find_edelay_branch(freqs, signals)
    estimates = np.asarray(
        [fit_edelay(freqs, row, branch_seed=branch_seed) for row in signals]
    )
    alias_period = 1.0 / float(freqs[1] - freqs[0])
    alias_errors = (estimates - edelay + 0.5 * alias_period) % alias_period
    alias_errors -= 0.5 * alias_period

    np.testing.assert_allclose(alias_errors, 0.0, atol=3e-3)
    assert np.ptp(estimates) < 1e-3


def test_related_four_point_uniform_traces_align_refined_aliases() -> None:
    edelay = 0.02
    freqs = np.linspace(5000.0, 5030.0, 4)
    signals = np.asarray(
        [
            hanger_truth(
                freqs,
                freq=freq,
                Ql=740.0,
                Qc_abs=1100.0,
                phi=0.12,
                a0=1.25 * np.exp(0.31j),
                edelay=edelay,
                bg_amp_slope=0.0,
            )
            for freq in (5008.0, 5013.0)
        ]
    )

    branch_seed = find_edelay_branch(freqs, signals)
    estimates = np.asarray(
        [fit_edelay(freqs, row, branch_seed=branch_seed) for row in signals]
    )
    alias_period = 1.0 / float(freqs[1] - freqs[0])
    mean_error = (np.mean(estimates) - edelay + 0.5 * alias_period) % alias_period
    mean_error -= 0.5 * alias_period

    assert np.ptp(estimates) < 0.5 * alias_period
    assert abs(mean_error) < 3e-3


@pytest.mark.parametrize("search_radius", [1.5, 20.0])
def test_fit_edelay_recovers_large_delay_with_bounded_search(
    search_radius: float,
) -> None:
    freq, Ql, edelay = 5549.0, 740.0, 11.299
    freqs = homophasal_like_grid(freq, Ql, 15.0, 301)
    signals = hanger_truth(
        freqs,
        freq=freq,
        Ql=Ql,
        Qc_abs=1100.0,
        phi=0.12,
        a0=1.25 * np.exp(0.31j),
        edelay=edelay,
        bg_amp_slope=0.004,
    )
    estimated = fit_edelay(
        freqs,
        signals,
        search_radius=search_radius,
        max_search_radius=20.0,
    )

    assert estimated == pytest.approx(edelay, abs=3e-3)


@pytest.mark.parametrize(
    ("search_radius", "max_search_radius", "error"),
    [
        (1.5, None, "search boundary"),
        (1.5, 3.0, "search boundary"),
        (1e6, None, "resource limit"),
    ],
)
def test_fit_edelay_rejects_insufficient_or_oversized_search(
    search_radius: float,
    max_search_radius: float | None,
    error: str,
) -> None:
    freqs = homophasal_like_grid(5549.0, 740.0, 15.0, 301)
    signals = hanger_truth(
        freqs,
        freq=5549.0,
        Ql=740.0,
        Qc_abs=1100.0,
        phi=0.12,
        a0=1.25 * np.exp(0.31j),
        edelay=11.299,
        bg_amp_slope=0.004,
    )

    with pytest.raises(ValueError, match=error):
        fit_edelay(
            freqs,
            signals,
            search_radius=search_radius,
            max_search_radius=max_search_radius,
        )


def test_find_edelay_branch_warns_for_near_tied_nonuniform_peaks() -> None:
    # Almost-uniform sampling leaves adjacent alias branches nearly indistinguishable.
    freqs = np.linspace(5000.0, 5030.0, 301) + 1e-5 * np.sin(np.arange(301))
    edelay = 0.2
    signals = np.exp(-1j * 2.0 * np.pi * freqs * edelay)

    with pytest.warns(RuntimeWarning, match="branch search is ambiguous"):
        estimated = find_edelay_branch(freqs, signals)

    assert estimated == pytest.approx(edelay, abs=1e-8)


def test_fit_edelay_rejects_nonmonotonic_frequency_grid() -> None:
    freqs = np.asarray([5000.0, 5000.2, 5000.1, 5000.3])
    signals = np.exp(-1j * 2.0 * np.pi * freqs * 0.2)

    with pytest.raises(ValueError, match="strictly monotonic"):
        fit_edelay(freqs, signals)


def test_fit_edelay_rejects_branch_seed_with_search_radius() -> None:
    freqs = np.linspace(4990.0, 5010.0, 101)
    signals = np.exp(-1j * 2.0 * np.pi * freqs * 0.2)

    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_edelay(freqs, signals, branch_seed=0.2, search_radius=5.0)


def test_fit_edelay_rejects_branch_seed_with_max_search_radius() -> None:
    freqs = np.linspace(4990.0, 5010.0, 101)
    signals = np.exp(-1j * 2.0 * np.pi * freqs * 0.2)

    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_edelay(freqs, signals, branch_seed=0.2, max_search_radius=5.0)
