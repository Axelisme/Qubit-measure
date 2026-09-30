"""OneTone FluxDep analysis uses the caller's result and named native figure."""

from io import BytesIO

import numpy as np
from zcu_tools.experiment.v2.onetone.flux_dep import (
    FluxDepAnalysis,
    FluxDepAnalyzeOptions,
    FluxDepExp,
    FluxDepResult,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots


def test_analyze_committed_flux_lines_keeps_numeric_result_and_native_pick() -> None:
    values = np.linspace(-0.5, 0.5, 9)
    freqs = np.linspace(4.8, 5.4, 7)
    signals = np.asarray(
        np.sin(values[:, None] * 7 + freqs[None, :] * 9)
        + 1j * np.cos(values[:, None] * 3 - freqs[None, :] * 7),
        dtype=np.complex128,
    )
    source = FluxDepResult(values, freqs, signals)
    before = signals.copy()
    plots = Plots(NonPresentingHost())

    analysis = FluxDepExp().analyze(
        source,
        FluxDepAnalyzeOptions(-0.2, 0.3, magnitude_only=True),
        plots=plots,
    )

    assert analysis == FluxDepAnalysis(-0.2, 0.3, 1.0)
    assert list(plots) == ["pick"]
    plots.finish()
    plots.release()
    png = BytesIO()
    plots["pick"].savefig(png, format="png")
    assert png.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_array_equal(signals, before)
