"""v2 SampleTable contract tests for the fluxdep sample-point visualizer.

``FreqFluxDependVisualizer.plot_sample_points`` must resolve coordinates only
through the shared v2 validator/resolver seam: explicit ``flux``, row-frame
derivation and caller-declared fallback frames all plot at their resolved
flux; unresolved rows and legacy coordinate columns fail fast before any
trace is added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from plotly.graph_objects import Figure, Scatter
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.notebook.analysis.fluxdep.utils import FreqFluxDependVisualizer
from zcu_tools.resources.sample_table import SampleFluxFrame, SampleTableV2Error


def _sample_points_trace(fig: Figure) -> Scatter:
    # plot_sample_points adds exactly one marker scatter.
    traces = []
    for trace in fig.data:
        assert isinstance(trace, Scatter)
        if trace.mode == "markers":
            traces.append(trace)
    assert len(traces) == 1
    return traces[0]


def _coordinates(values: object) -> NDArray[np.float64]:
    assert isinstance(values, (list, tuple, np.ndarray))
    assert not np.iscomplexobj(values)
    return np.asarray(values, dtype=np.float64)


def _trace_count(fig: Figure) -> int:
    data = fig.data
    assert isinstance(data, (list, tuple))
    return len(data)


def _make_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dev_value": 0.0,
                "dev_unit": "A",
                "Freq (MHz)": 4000.0,
                "T1 (us)": 10.0,
                **row,
            }
            for row in rows
        ]
    )


def test_explicit_flux_rows_plot_at_resolved_flux() -> None:
    table = _make_table(
        [
            {"flux": 0.3, "comment": "a"},
            {"flux": 0.7, "comment": "b"},
        ]
    )
    vis = FreqFluxDependVisualizer()
    vis.plot_sample_points(table)

    trace = _sample_points_trace(vis.fig)
    np.testing.assert_allclose(_coordinates(trace.x), [0.3, 0.7])
    # Freq (MHz) -> GHz display conversion is preserved.
    np.testing.assert_allclose(_coordinates(trace.y), [4.0, 4.0])
    assert vis.xlimits == [0.3, 0.7]


def test_explicit_flux_precedes_row_frame() -> None:
    table = _make_table(
        [
            {
                "flux": 0.25,
                "flux_int": 0.0,
                "flux_period": 1.0,
                "dev_value": 0.9,  # row frame would give flux 0.9
            }
        ]
    )
    vis = FreqFluxDependVisualizer()
    vis.plot_sample_points(table)

    trace = _sample_points_trace(vis.fig)
    np.testing.assert_allclose(_coordinates(trace.x), [0.25])


def test_row_frame_derived_rows_plot_at_derived_flux() -> None:
    table = _make_table(
        [
            {"flux_int": 0.0, "flux_period": 2.0, "dev_value": 1.0},  # flux 0.5
            {"flux_int": 0.0, "flux_period": 2.0, "dev_value": -1.0},  # flux -0.5
        ]
    )
    vis = FreqFluxDependVisualizer()
    vis.plot_sample_points(table)

    trace = _sample_points_trace(vis.fig)
    np.testing.assert_allclose(_coordinates(trace.x), [0.5, -0.5])


def test_fallback_frame_rows_plot_at_fallback_flux() -> None:
    table = _make_table(
        [
            {"dev_value": 0.5},  # fallback flux 0.5
            {"dev_value": -0.5},  # fallback flux -0.5
        ]
    )
    vis = FreqFluxDependVisualizer()
    vis.plot_sample_points(
        table, fallback_frame=SampleFluxFrame("A", flux_int=0.0, flux_period=1.0)
    )

    trace = _sample_points_trace(vis.fig)
    np.testing.assert_allclose(_coordinates(trace.x), [0.5, -0.5])


def test_mixed_provenance_rows_plot_together() -> None:
    table = _make_table(
        [
            {"flux": 0.25, "comment": "explicit"},
            {"flux_int": 0.0, "flux_period": 1.0, "dev_value": 0.5},  # row-frame
            {"dev_value": 0.75, "comment": "fallback"},  # fallback 0.75
        ]
    )
    vis = FreqFluxDependVisualizer()
    vis.plot_sample_points(
        table, fallback_frame=SampleFluxFrame("A", flux_int=0.0, flux_period=1.0)
    )

    trace = _sample_points_trace(vis.fig)
    np.testing.assert_allclose(_coordinates(trace.x), [0.25, 0.5, 0.75])
    assert vis.xlimits == [0.25, 0.75]


def test_hover_labels_exclude_coordinate_and_freq_columns() -> None:
    table = _make_table(
        [
            {
                "flux": 0.3,
                "flux_int": 0.0,
                "flux_period": 1.0,
                "dev_value": 0.3,
                "dev_unit": "A",
                "T1 (us)": 11.0,
                "comment": "hello",
            }
        ]
    )
    vis = FreqFluxDependVisualizer()
    vis.plot_sample_points(table)

    trace = _sample_points_trace(vis.fig)
    hovertext = trace.hovertext
    assert isinstance(hovertext, (list, tuple, np.ndarray))
    label = np.asarray(hovertext)[0]
    assert isinstance(label, str)
    assert "dev_value" not in label
    assert "dev_unit" not in label
    assert "flux_int" not in label
    assert "flux_period" not in label
    assert "Freq (MHz)" not in label
    assert "T1 (us)=11.0" in label
    assert "comment=hello" in label


def test_unresolved_rows_fail_with_indexes() -> None:
    table = _make_table([{"flux": 0.3}, {"comment": "no frame"}])
    vis = FreqFluxDependVisualizer()

    with pytest.raises(SampleTableV2Error, match="row\\(s\\) \\[1\\]"):
        vis.plot_sample_points(table)
    # nothing was plotted before the failure
    assert _trace_count(vis.fig) == 0


def test_fallback_frame_unit_mismatch_is_unresolved() -> None:
    table = _make_table([{"dev_value": 1.0}])  # dev_unit "A"
    vis = FreqFluxDependVisualizer()

    with pytest.raises(SampleTableV2Error, match="row\\(s\\) \\[0\\]"):
        vis.plot_sample_points(
            table, fallback_frame=SampleFluxFrame("V", flux_int=0.0, flux_period=1.0)
        )


def test_legacy_coordinate_column_fails_before_analysis() -> None:
    table = pd.DataFrame(
        {
            "calibrated mA": [0.1, 0.2],
            "Freq (MHz)": [4000.0, 4100.0],
        }
    )
    vis = FreqFluxDependVisualizer()

    with pytest.raises(SampleTableV2Error, match="calibrated mA"):
        vis.plot_sample_points(table)
    assert _trace_count(vis.fig) == 0


def test_empty_table_fails_before_analysis() -> None:
    table = pd.DataFrame(
        {"dev_value": pd.Series([], dtype=float), "dev_unit": pd.Series([], dtype=str)}
    )
    vis = FreqFluxDependVisualizer()

    with pytest.raises(SampleTableV2Error, match="empty"):
        vis.plot_sample_points(table)
    assert _trace_count(vis.fig) == 0


def test_mixed_unit_rows_plot_at_resolved_flux() -> None:
    """A and V rows plot side by side at their own frame-resolved flux.

    The visualizer resolves each row through its own frame and never compares
    cross-unit ``dev_value``: the A row (1.0 A) and the V row (3.0 V) both
    resolve to flux 0.5 even though their device values are far apart.
    """
    table = _make_table(
        [
            # flux = (1.0 - 0.0) / 2.0 = 0.5
            {"dev_unit": "A", "flux_int": 0.0, "flux_period": 2.0, "dev_value": 1.0},
            # flux = (3.0 - 2.0) / 2.0 = 0.5; direct dev_value comparison with
            # the A row would pick neither, so only resolved flux may be used
            {"dev_unit": "V", "flux_int": 2.0, "flux_period": 2.0, "dev_value": 3.0},
        ]
    )
    vis = FreqFluxDependVisualizer()
    vis.plot_sample_points(table)

    trace = _sample_points_trace(vis.fig)
    np.testing.assert_allclose(_coordinates(trace.x), [0.5, 0.5])
    np.testing.assert_allclose(_coordinates(trace.y), [4.0, 4.0])


def test_plot_md_fluent_chain_smoke() -> None:
    """The notebook_md/analysis/plot.md fluent chain must bind end to end.

    Mirrors the "Flux dependence" cell of the manual plot notebook with
    synthetic arrays (no result files, no hardware): every fluent call resolves
    to a real visualizer method, sample rows without explicit flux or a row
    frame resolve through the caller-declared fallback frame, and the
    dev-value secondary x axis is produced.
    """
    flxs = np.linspace(0.5, 1.0, 100)
    energies = np.zeros((len(flxs), 10))
    frame = SampleFluxFrame("A", flux_int=0.0, flux_period=1.0)
    dev_values = frame.dev_value_from_flux(flxs)

    v_allows: TransitionDict = {
        "transitions": [(0, 1), (1, 4), (4, 7), (7, 9), (1, 9)],
        "sample_f": 9.58464 / 2,
        "r_f": 7.0,
    }

    # v2 rows with no explicit flux and no row frame: resolved via fallback.
    freqs_df = pd.DataFrame(
        {
            "dev_value": [0.5, 0.8],
            "dev_unit": ["A", "A"],
            "Freq (MHz)": [4000.0, 4100.0],
            "T1 (us)": [10.0, 12.0],
        }
    )

    fig = (
        FreqFluxDependVisualizer()
        .plot_simulation_lines(flxs, energies, v_allows)
        .plot_sample_points(freqs_df, fallback_frame=frame)
        .plot_constant_freq(v_allows["r_f"], "r_f")
        .plot_constant_freq(v_allows["sample_f"], "sample_f")
        .plot_constant_freq(2 * v_allows["sample_f"] - v_allows["r_f"], "mirror r_f")
        .add_dev_values_ticks(flxs, dev_values)
        .get_figure()
    )

    # sample-point markers plotted at fallback-resolved flux, MHz->GHz display
    markers = _sample_points_trace(fig)
    np.testing.assert_allclose(_coordinates(markers.x), [0.5, 0.8])
    np.testing.assert_allclose(_coordinates(markers.y), [4.0, 4.1])

    # dev-value secondary x axis ticks overlaid on the flux axis
    xaxis2 = fig.layout.xaxis2
    assert xaxis2 is not None
    assert xaxis2.side == "top"
    assert xaxis2.overlaying == "x"
    assert isinstance(xaxis2.ticktext, (list, tuple))
    tickvals = _coordinates(xaxis2.tickvals)
    assert len(tickvals) == len(xaxis2.ticktext)
    n = flxs.shape[0]
    tick_indices = np.unique(np.round(np.linspace(0, n - 1, 12)).astype(int))
    np.testing.assert_allclose(tickvals, flxs[tick_indices])
    assert list(xaxis2.ticktext) == [f"{v:.1e}" for v in dev_values[tick_indices]]

    # three constant-frequency reference lines (added as layout shapes)
    assert len(fig.layout.shapes) == 3
