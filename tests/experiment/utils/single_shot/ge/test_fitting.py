"""Paired histogram calibration through the shared fitting/plotting interface."""

import numpy as np
import pytest
from zcu_tools.analysis.fitting.singleshot import calc_population_pdf
from zcu_tools.experiment.utils.single_shot.ge import singleshot_ge_analysis
from zcu_tools.experiment.utils.single_shot.ge.base import fitting_ge_and_plot
from zcu_tools.plotting.plots import NonPresentingHost, Plots


@pytest.mark.parametrize(
    ("length_ratio", "align_t1", "logscale"),
    [(0.0, True, False), (0.4, True, True), (0.4, False, False)],
)
def test_histogram_calibration_retains_centers_populations_and_diagnostics(
    length_ratio, align_t1, logscale
):
    xs = np.linspace(-3.0, 3.0, 10001)
    quantiles = (np.arange(2000) + 0.5) / 2000
    ground_pdf = calc_population_pdf(xs, -1.0, 1.0, 0.25, 1.0, 0.0, 0.5, length_ratio)
    excited_pdf = calc_population_pdf(xs, -1.0, 1.0, 0.25, 0.0, 1.0, 0.5, length_ratio)
    ground = np.interp(quantiles, np.cumsum(ground_pdf) / ground_pdf.sum(), xs)
    excited = np.interp(quantiles, np.cumsum(excited_pdf) / excited_pdf.sum(), xs)
    quadratures = 0.1 * np.sin(np.arange(2000))
    signals = np.asarray([ground + 1j * quadratures, excited + 1j * quadratures])
    plots = Plots(NonPresentingHost())

    fidelity, populations, result = fitting_ge_and_plot(
        signals,
        lambda *_: {"theta": 0.0},
        numbins=80,
        length_ratio=length_ratio,
        init_p0_g=1.0,
        init_p0_e=0.0,
        avg_p=0.5,
        align_t1=align_t1,
        logscale=logscale,
        plots=plots,
    )
    figures = plots.finish(present=False)
    plots.release()

    assert 0.7 < fidelity <= 1.0
    np.testing.assert_allclose(populations, [[1.0, 0.0], [0.0, 1.0]], atol=0.03)
    assert result["g_center"].real == pytest.approx(-1.0, abs=0.03)
    assert result["e_center"].real == pytest.approx(1.0, abs=0.03)
    assert result["s"] == pytest.approx(0.25, abs=0.03)
    assert result["length_ratio_g"] == pytest.approx(length_ratio, abs=0.05)
    assert result["length_ratio_e"] == pytest.approx(length_ratio, abs=0.05)
    assert set(result["fit_quality"]) == {"joint", "ground", "excited"}
    assert set(figures) == {"fit"}
    figure = figures["fit"]
    assert len(figure.axes) == 4
    assert figure.axes[1].get_yscale() == ("log" if logscale else "linear")
    labels = {line.get_label() for line in figure.axes[1].lines}
    assert {"total", "ground", "excited", "other"} <= labels
    assert ("ideal total" in labels) == (length_ratio != 0.0)


@pytest.mark.parametrize(
    "selection", [{"backend": "pca"}, {"backend": "center"}, {"angle": 0.0}]
)
def test_ge_analysis_custom_names_coexist_with_default_owned_figures(selection):
    from scipy.special import ndtri

    quantiles = (np.arange(1000) + 0.5) / 1000
    noise = 0.25 * ndtri(quantiles)
    quadrature = 0.1j * np.sin(np.arange(1000))
    signals = np.asarray([-1 + noise + quadrature, 1 + noise + quadrature])
    plots = Plots(NonPresentingHost())
    other_plots = Plots(NonPresentingHost())
    try:
        baseline = singleshot_ge_analysis(
            signals, plots=plots, length_ratio=0.0, **selection
        )
        original = plots["fit"]
        for name in ("fit_0", "fit_2"):
            fitted = singleshot_ge_analysis(
                signals, plots=plots, figure_name=name, length_ratio=0.0, **selection
            )
            assert fitted[0] == pytest.approx(baseline[0])
            np.testing.assert_allclose(fitted[1], baseline[1])
            assert fitted[2]["g_center"] == pytest.approx(baseline[2]["g_center"])
            assert len(plots[name].axes) == 4
            with pytest.raises(ValueError, match="owned by another collection"):
                other_plots.adopt(name, plots[name])
        assert plots["fit"] is original
        assert set(plots.finish(present=False)) == {"fit", "fit_0", "fit_2"}
    finally:
        plots.finish(present=False)
        plots.release()
        other_plots.finish(present=False)
        other_plots.release()
