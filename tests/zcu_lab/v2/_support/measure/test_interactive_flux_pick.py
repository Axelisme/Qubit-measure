"""Measure flux plugin contract through both production adapters."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Literal

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep.line_state import fold_initial_lines
from zcu_tools.experiment.records import RunRecord
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext, RoleBinding
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import ComponentSchema, RoleSpec
from zcu_tools.resources.entry.schema import PointDocument

from tests.resources.entry.fakes import registry_state
from tests.resources.entry.view_support import registered_model
from zcu_lab.v2._support.measure.flux_pick_plugin import (
    FluxPickPlugin,
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_lab.v2._support.measure.interactive_flux_pick import (
    FluxPickParams,
    FluxPickResult,
)
from zcu_lab.v2.onetone.flux_dep.core import FluxDepCfg, FluxDepResult
from zcu_lab.v2.onetone.flux_dep.gui import OneToneFluxDepAdapter
from zcu_lab.v2.twotone.fluxdep.core import FreqFluxCfg, FreqFluxResult
from zcu_lab.v2.twotone.fluxdep.gui import FluxDepAdapter as TwoToneFluxDepAdapter

FluxParameters = Callable[[YamlValue, YamlValue], ParameterContext]
FluxEntrypoint = Literal["factory", "twotone"]


@pytest.fixture
def flux_parameters() -> Generator[FluxParameters]:
    class Qubit(ComponentSchema):
        flux_half: YamlValue = None
        flux_int: YamlValue = None

    before = registry_state()
    with registered_model("test/flux-picker-qubit", Qubit) as kind:

        def make(half: YamlValue, integer: YamlValue) -> ParameterContext:
            point = PointDocument.model_validate(
                {
                    "format": "zcu.parameter-container",
                    "format_version": "1.0",
                    "general": {"created_at": "2026-10-04T00:00:00Z"},
                    "components": {
                        "Q1": {"kind": kind, "flux_half": half, "flux_int": integer}
                    },
                }
            )
            return ParameterContext(
                point=point,
                library=LibrarySnapshot(
                    ModuleLibraryDocument(
                        format="zcu.module-library",
                        format_version="1.0",
                    )
                ),
                target=SavedPoint(
                    entry_id="00000000-0000-0000-0000-000000000001", point="working"
                ),
                roles=RoleBinding("Q1", {"qubit": RoleSpec(kind)}, {"qubit": "Q1"}),
            )

        yield make
    assert registry_state() == before


@pytest.fixture
def flux_plots() -> Generator[Plots]:
    plots = Plots(NonPresentingHost())
    try:
        yield plots
    finally:
        plots.finish()
        plots.release()


def _current_plugin(
    entrypoint: FluxEntrypoint, parameters: ParameterContext | None, plots: Plots
) -> FluxPickPlugin:
    values = np.linspace(-5.0, 5.0, 60, dtype=np.float64)
    freqs = np.linspace(4.0, 5.0, 30, dtype=np.float64)
    signals = np.asarray(
        np.exp(-(values[:, None] ** 2)) * np.ones((1, 30)), dtype=np.complex128
    )
    result = FreqFluxResult(values, freqs, signals)
    if entrypoint == "factory":
        return make_flux_pick_plugin(
            AnalyzeRequest(
                run_result=result,
                analyze_params=FluxPickParams(),
                parameters=parameters,
            ),
            force_magnitude=True,
            plots=plots,
            result_builder=render_flux_pick,
        )
    return TwoToneFluxDepAdapter().make_interactive_plugin(
        AnalyzeRequest(
            run_result=RunRecord[FreqFluxCfg, FreqFluxResult](cfg=None, result=result),
            analyze_params=FluxPickParams(),
            parameters=parameters,
        ),
        plots=plots,
    )


@pytest.mark.parametrize("entrypoint", ["factory", "twotone"])
def test_current_request_without_parameters_builds_and_finishes_picker(
    entrypoint: FluxEntrypoint, flux_plots: Plots
) -> None:
    plugin = _current_plugin(entrypoint, None, flux_plots)
    session = plugin.open(ManualOwnerScheduler())
    try:
        state = session.snapshot()
        assert state.flux_half == pytest.approx(0.0)
        assert state.flux_int > state.flux_half
        assert state.magnitude_only is (entrypoint == "factory")
        plugin.execute_command(session, "move_line", {"role": "half", "position": 0.5})
        result = plugin.finish(session)
        assert result.flx_half == pytest.approx(0.5)
        assert result.flx_period > 0.0
        assert tuple(flux_plots) == ("pick",)
    finally:
        session.dispose()


@pytest.mark.parametrize("entrypoint", ["factory", "twotone"])
@pytest.mark.parametrize("half", [None, 0.5])
def test_current_request_uses_fixed_parameter_calibration(
    entrypoint: FluxEntrypoint,
    half: float | None,
    flux_parameters: FluxParameters,
    flux_plots: Plots,
) -> None:
    parameters = flux_parameters(half, 2.0)
    expected_half, expected_int = fold_initial_lines(
        np.linspace(-5.0, 5.0, 60, dtype=np.float64), half, 2.0
    )
    plugin = _current_plugin(entrypoint, parameters, flux_plots)
    session = plugin.open(ManualOwnerScheduler())
    try:
        state = session.snapshot()
        assert state.flux_half == pytest.approx(expected_half)
        assert state.flux_int == pytest.approx(expected_int)
        assert plugin.finish(session).flx_period == pytest.approx(
            2 * abs(expected_int - expected_half)
        )
    finally:
        session.dispose()


@pytest.mark.parametrize("entrypoint", ["factory", "twotone"])
@pytest.mark.parametrize(
    ("half", "error", "message"),
    [
        (True, TypeError, "real scalar"),
        ("bad", TypeError, "real scalar"),
        (float("nan"), ValueError, "finite"),
    ],
)
def test_current_request_propagates_invalid_parameter_calibration(
    entrypoint: FluxEntrypoint,
    half: YamlValue,
    error: type[Exception],
    message: str,
    flux_parameters: FluxParameters,
    flux_plots: Plots,
) -> None:
    with pytest.raises(error, match=message):
        _current_plugin(entrypoint, flux_parameters(half, 2.0), flux_plots)


def _request(
    parameters: ParameterContext | None = None,
) -> AnalyzeRequest[FreqFluxResult, FluxPickParams]:
    devs = np.linspace(-5.0, 5.0, 60, dtype=np.float64)
    freqs = np.linspace(4.0, 5.0, 30, dtype=np.float64)
    signals = np.asarray(
        np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30)), dtype=np.complex128
    )
    return AnalyzeRequest(
        run_result=FreqFluxResult(values=devs, freqs=freqs, signals=signals),
        analyze_params=FluxPickParams(),
        parameters=parameters,
    )


def test_plugin_typed_actions_and_commands_share_committed_state():
    plots = Plots(NonPresentingHost())
    plugin = make_flux_pick_plugin(
        _request(), force_magnitude=True, plots=plots, result_builder=render_flux_pick
    )
    session = plugin.open(ManualOwnerScheduler())
    start = session.snapshot()
    assert start.magnitude_only is True
    assert start.conjugate is False

    plugin.actions.move.execute(session, ("half", start.flux_half + 0.6))
    after_gui = session.snapshot()
    assert after_gui.flux_half == pytest.approx(start.flux_half + 0.6)
    assert after_gui.flux_int == start.flux_int

    plugin.execute_command(session, "set_conjugate", {"enabled": True})
    plugin.execute_command(
        session, "move_line", {"role": "half", "position": after_gui.flux_half + 0.4}
    )
    after_remote = session.snapshot()
    assert after_remote.flux_half - after_gui.flux_half == pytest.approx(0.4)
    assert after_remote.flux_int - after_gui.flux_int == pytest.approx(0.4)
    with pytest.raises(InvalidInputError, match="role"):
        plugin.execute_command(session, "move_line", {"role": "bad", "position": 1.0})
    assert session.snapshot() == after_remote
    plugin.execute_command(session, "swap_lines", {})
    swapped = session.snapshot()
    assert (swapped.flux_half, swapped.flux_int) == (
        after_remote.flux_int,
        after_remote.flux_half,
    )
    result = plugin.finish(session)
    assert isinstance(result, FluxPickResult)
    assert result.flx_half == swapped.flux_half
    assert result.flx_period == 2 * abs(swapped.flux_int - swapped.flux_half)
    assert tuple(plots) == ("pick",)
    assert isinstance(plots["pick"], Figure)
    plots.finish()
    plots.release()


def test_equal_seed_cannot_finish_until_a_valid_line_is_committed(
    flux_parameters: FluxParameters,
) -> None:
    parameters = flux_parameters(0.0, 0.0)
    plots = Plots(NonPresentingHost())
    plugin = make_flux_pick_plugin(
        _request(parameters),
        force_magnitude=True,
        plots=plots,
        result_builder=render_flux_pick,
    )
    session = plugin.open(ManualOwnerScheduler())
    with pytest.raises(FailedPreconditionError, match="separat"):
        plugin.finish(session)
    assert session.snapshot().flux_half == session.snapshot().flux_int
    plugin.execute_command(session, "move_line", {"role": "half", "position": 1.0})
    assert plugin.finish(session).flx_period > 0.0
    assert tuple(plots) == ("pick",)
    plots.finish()
    plots.release()


@pytest.mark.parametrize(
    ("adapter_type", "magnitude_only"),
    [(OneToneFluxDepAdapter, True), (TwoToneFluxDepAdapter, False)],
)
def test_both_adapters_seed_projection_and_publish_named_result_figure(
    adapter_type: type[OneToneFluxDepAdapter] | type[TwoToneFluxDepAdapter],
    magnitude_only: bool,
    flux_parameters: FluxParameters,
) -> None:
    parameters = flux_parameters(0.0, 2.0)
    plots = Plots(NonPresentingHost())
    bare = _request(parameters).run_result
    if adapter_type is OneToneFluxDepAdapter:
        plugin = OneToneFluxDepAdapter().make_interactive_plugin(
            AnalyzeRequest(
                run_result=RunRecord[FluxDepCfg, FluxDepResult](
                    cfg=None,
                    result=FluxDepResult(bare.values, bare.freqs, bare.signals),
                ),
                analyze_params=FluxPickParams(),
                parameters=parameters,
            ),
            plots=plots,
        )
    else:
        plugin = TwoToneFluxDepAdapter().make_interactive_plugin(
            AnalyzeRequest(
                run_result=RunRecord[FreqFluxCfg, FreqFluxResult](
                    cfg=None, result=bare
                ),
                analyze_params=FluxPickParams(),
                parameters=parameters,
            ),
            plots=plots,
        )
    assert isinstance(plugin, FluxPickPlugin)
    session = plugin.open(ManualOwnerScheduler())
    state = session.snapshot()
    assert state.magnitude_only is magnitude_only
    assert state.flux_half == pytest.approx(0.0)
    assert state.flux_int == pytest.approx(2.0)
    assert [command.name for command in plugin.commands] == [
        "move_line",
        "set_conjugate",
        "swap_lines",
        "auto_align",
    ]
    result = plugin.finish(session)
    assert tuple(plots) == ("pick",)
    assert isinstance(plots["pick"], Figure)
    assert result.to_summary_dict() == {
        "flx_half": state.flux_half,
        "flx_int": state.flux_int,
        "flx_period": 2 * abs(state.flux_int - state.flux_half),
    }
    plots.finish()
    plots.release()
