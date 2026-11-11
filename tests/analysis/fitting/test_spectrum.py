"""Candidate selection and isolated numerical spectrum fitting contracts."""

import numpy as np
import pytest
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.analysis.fitting import detect_peaks
from zcu_tools.analysis.fitting.spectrum import SpectrumFitOptions, fit_spectrum


@pytest.mark.parametrize(
    "expected, center, width, interval",
    [(5040.0, 5040.0, 6.0, (5010.0, 5100.0)), (None, 4980.0, 8.0, (4900.0, 5010.0))],
)
def test_selected_peak_excludes_its_neighbour(
    expected: float | None, center: float, width: float, interval: tuple[float, float]
) -> None:
    freqs = np.linspace(4900.0, 5100.0, 801)
    # Independent Lorentzian truths: centers 4980/5040, FWHM 8/6 MHz.
    observed = 1 / (1 + ((freqs - 4980) / 4) ** 2)
    observed += 0.6 / (1 + ((freqs - 5040) / 3) ** 2)
    original = observed.copy()

    fit = fit_spectrum(
        freqs, observed, SpectrumFitOptions(expected_center_mhz=expected)
    )

    assert fit.freq == pytest.approx(center, abs=0.02)
    assert fit.fwhm == pytest.approx(width, rel=0.03)
    assert fit.selected_peak is not None
    assert fit.selected_peak.center == pytest.approx(center, abs=0.25)
    assert [peak.center for peak in fit.peaks] == pytest.approx(
        [4980.0, 5040.0], abs=0.25
    )
    assert fit.freqs.min() == interval[0]
    assert fit.freqs.max() == interval[1]
    assert fit.freqs.shape == fit.observed.shape == fit.fitted.shape
    assert np.isfinite([fit.freq_err, fit.fwhm_err]).all()
    np.testing.assert_array_equal(observed, original)


@pytest.mark.parametrize("isolate", [True, False])
def test_signed_sinc_recovers_center_width_and_negative_sidelobes(
    isolate: bool,
) -> None:
    freqs = np.linspace(4900.0, 5100.0, 401)
    observed = 0.1 + 2 * np.sinc((freqs - 5000) / 5)

    fit = fit_spectrum(
        freqs,
        observed,
        SpectrumFitOptions(model_type="sinc", min_prominence=0.5),
        isolate_peaks=isolate,
    )

    assert fit.freq == pytest.approx(5000.0, abs=0.01)
    assert fit.fwhm == pytest.approx(6.0335, rel=0.01)
    np.testing.assert_allclose(fit.fitted, fit.observed, atol=1e-6)
    assert fit.fitted.min() < 0


def test_supplied_empty_candidates_allow_legacy_whole_window_fit() -> None:
    freqs = np.linspace(4900.0, 5100.0, 401)
    observed = 0.1 + 1 / (1 + ((freqs - 5000) / 4) ** 2)

    fit = fit_spectrum(
        freqs, observed, SpectrumFitOptions(), peaks=(), isolate_peaks=False
    )

    assert fit.freq == pytest.approx(5000.0, abs=0.01)
    assert fit.fwhm == pytest.approx(8.0, rel=0.01)
    assert fit.peaks == ()
    assert fit.selected_peak is None
    np.testing.assert_array_equal(fit.freqs, freqs)


def test_no_candidate_distinguishes_flat_signal_from_narrow_peak() -> None:
    freqs = np.linspace(4900.0, 5100.0, 401)
    with pytest.raises(AnalysisFailed, match="No candidate.*prominence and SNR"):
        fit_spectrum(freqs, np.zeros_like(freqs), SpectrumFitOptions())

    spike = np.zeros_like(freqs)
    spike[200] = 1.0
    with pytest.raises(
        AnalysisFailed,
        match=r"min_width_samples=3 samples: 5000 MHz .*observable width 1 samples",
    ):
        fit_spectrum(freqs, spike, SpectrumFitOptions(min_width_samples=3.0))


def test_supplied_neighbours_with_too_few_isolated_samples_fail() -> None:
    freqs = np.arange(11.0)
    observed = np.array([0.0, 0, 1, 0, 0, 3, 0, 0, 2, 0, 0])
    peaks = tuple(detect_peaks(freqs, observed, min_snr=0, min_width_samples=0))

    with pytest.raises(AnalysisFailed, match="Fewer than six samples"):
        fit_spectrum(freqs, observed, SpectrumFitOptions(), peaks=peaks)
