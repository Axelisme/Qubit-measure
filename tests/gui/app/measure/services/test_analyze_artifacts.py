"""Typed analysis and flux-picker adapter results retained as saveable artifacts."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickAnalysis,
    FluxPickInputs,
    FluxPickState,
)
from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest
from zcu_tools.gui.app.measure.artifacts import SaveStatus
from zcu_tools.gui.app.measure.interactive_flux_pick import (
    FluxPickParams,
    FluxPickResult,
)
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.interactive import PluginDefinition
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import ArtifactKey

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._components import QubitSchema
from tests.gui.app.measure._flux_adapter import (
    SpectrumAdapter,
    SpectrumCfg,
    SpectrumCore,
    SpectrumResult,
)
from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    PaneParams,
    PaneSummary,
    native_save_session,
)
from tests.gui.app.measure.services._analyze_support import (
    make_plots,
    make_service,
    make_state,
)


@pytest.fixture
def native_session(tmp_path: Path, qapp: QApplication) -> Generator[NativeSaveSession]:
    with native_save_session(tmp_path, qapp) as session:
        yield session


def _flux_parameters(original: ParameterContext | None = None) -> ParameterContext:
    qubit = QubitSchema(kind="qubit/fluxonium", flux_half=-0.2, flux_int=0.3)
    if original is None:
        parameters = parameter_capture({"Q1": qubit})
    else:
        parameters = ParameterContext(
            point=original.document().model_copy(update={"components": {"Q1": qubit}}),
            library=original.library,
            target=original.target,
            roles=None,
            storage=original.storage,
        )
    return parameters.bind_aliases((AliasRow("qubit", "Q1"),), {})


def _start_onetone_plugin(
    state: State,
    source: RunRecord[SpectrumCfg, SpectrumResult],
    plots: Plots,
    handles: OperationHandles,
    *,
    writeback: MagicMock | None = None,
    tab_id: str = "tab1",
) -> tuple[AnalyzeService, PluginDefinition[Any, Any], int]:
    adapter = state.get_tab(tab_id).adapter
    assert isinstance(adapter, SpectrumAdapter)
    plugin = adapter.make_interactive_plugin(
        AnalyzeRequest(
            source,
            FluxPickParams(),
            state.get_tab(tab_id).run.parameters,
        ),
        plots=plots,
    )
    service, _bg = make_service(state, EventBus(), handles=handles, writeback=writeback)
    token = service.start_plugin(
        tab_id,
        plugin,
        ManualOwnerScheduler(),
        analyze_params_instance=FluxPickParams(),
        plots=plots,
    )
    return service, plugin, token


def test_typed_gui_analysis_publishes_typed_result_and_saveable_named_fit(
    native_session: NativeSaveSession,
) -> None:
    state = native_session.state
    tab = state.get_tab("tab")

    class NamedAdapter(AnalysisSavingAdapter):
        def analyze(self, req, *, plots):
            _, ax = plots.subplots("fit")
            ax.plot(req.run_result.result.frequency, req.run_result.result.signal.real)
            return PaneSummary(mean=req.analyze_params.gain)

    tab.adapter = NamedAdapter()
    times = np.linspace(0, 100, 101)
    signals = (0.2 + 0.8 * np.exp(-times / 20)).astype(np.complex128)
    source = RunRecord[SavingCfg, SavingResult](
        cfg=None, result=SavingResult(times, signals)
    )
    state.update_tab_result(
        "tab",
        replace(
            source,
            metadata=native_session.metadata.model_copy(
                update={"experiment": SavingCore.AXES_SPEC.tag}
            ),
        ),
        parameters=_flux_parameters(tab.run.parameters),
    )
    handles = OperationHandles()
    service, bg = make_service(state, EventBus(), handles=handles)
    plots = make_plots()
    token = service.start_analyze("tab", PaneParams(gain=3), plots=plots)
    assert bg.last_work is not None and bg.last_on_done is not None
    bg.last_on_done(bg.last_work())

    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"
    pane = state.get_tab("tab").analysis
    assert isinstance(pane.result, PaneSummary)
    assert pane.result.mean == 3.0
    assert pane.plots is plots
    assert tuple(plots) == ("fit",)
    artifacts = [
        artifact
        for section in state.get_artifact_snapshots("tab")
        for artifact in section.artifacts
        if artifact.ref == ArtifactRef("analysis", "results")
    ]
    assert len(artifacts) == 1 and artifacts[0].status is SaveStatus.NOT_SAVED
    assert [member.key for member in artifacts[0].members] == [
        ArtifactKey("analysis", "results", "figure", "fit"),
        ArtifactKey("analysis", "results", "analysis", "analysis"),
    ]
    assert [output.format for output in artifacts[0].members[0].outputs] == [
        "png",
        "png",
    ]
    assert [output.format for output in artifacts[0].members[1].outputs] == ["json"]
    assert all(
        output.status == "reserved"
        for member in artifacts[0].members
        for output in member.outputs
    )
    output = BytesIO()
    plots["fit"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")
    plots.release()


def test_onetone_gui_done_uses_frontend_result_and_retains_named_pick(
    native_session: NativeSaveSession,
) -> None:
    state = native_session.state
    tab = state.get_tab("tab")
    tab.adapter = SpectrumAdapter()
    values = np.linspace(-0.5, 0.5, 9)
    freqs = np.linspace(4.8, 5.4, 7)
    signals = np.asarray(
        np.sin(values[:, None] * 7 + freqs[None, :] * 9)
        + 1j * np.cos(values[:, None] * 3 - freqs[None, :] * 7),
        dtype=np.complex128,
    )
    source = RunRecord(cfg=None, result=SpectrumResult(values, freqs, signals))
    state.update_tab_result(
        "tab",
        replace(
            source,
            metadata=native_session.metadata.model_copy(
                update={"experiment": SpectrumCore.AXES_SPEC.tag}
            ),
        ),
        parameters=_flux_parameters(tab.run.parameters),
    )
    state.set_context(replace(state.session_env, parameters=tab.run.parameters))
    plots = make_plots()
    handles = OperationHandles()
    writeback = MagicMock()
    service, plugin, token = _start_onetone_plugin(
        state, source, plots, handles, writeback=writeback, tab_id="tab"
    )
    active = service.get_interactive("tab")
    assert active is not None
    assert tuple(plots) == ()
    plugin.execute_command(active.session, "swap_lines", {})
    committed = active.session.snapshot()
    assert service.finish_plugin("tab") is True
    assert service.get_interactive("tab") is None
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "finished"
    pane = state.get_tab("tab").analysis
    assert isinstance(pane.result, FluxPickResult)
    assert pane.result.flx_half == committed.flux_half
    assert pane.result.flx_int == committed.flux_int
    assert pane.result.flx_period == pytest.approx(1.0)
    writeback_items = writeback.create_draft.call_args.args[0]
    assert {item.target_name: item.proposed_value for item in writeback_items} == {
        "$qubit.flux_half": committed.flux_half,
        "$qubit.flux_int": committed.flux_int,
        "$qubit.flux_period": pane.result.flx_period,
    }
    assert pane.plots is plots
    assert tuple(plots) == ("pick",)
    np.testing.assert_allclose(
        np.asarray(plots["pick"].axes[0].lines[0].get_xdata(), dtype=np.float64),
        [committed.flux_half],
    )
    artifacts = [
        artifact
        for section in state.get_artifact_snapshots("tab")
        for artifact in section.artifacts
        if artifact.ref == ArtifactRef("analysis", "results")
    ]
    assert len(artifacts) == 1 and artifacts[0].status is SaveStatus.NOT_SAVED
    assert [member.key for member in artifacts[0].members] == [
        ArtifactKey("analysis", "results", "figure", "pick"),
        ArtifactKey("analysis", "results", "analysis", "analysis"),
    ]
    assert [output.format for output in artifacts[0].members[0].outputs] == [
        "png",
        "png",
    ]
    assert all(
        output.status == "reserved"
        for member in artifacts[0].members
        for output in member.outputs
    )
    plots.release()
    output = BytesIO()
    plots["pick"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")


def test_onetone_gui_cancel_retains_prior_analysis_and_no_new_pick(qapp) -> None:
    state = make_state()
    tab = state.get_tab("tab1")
    tab.adapter = SpectrumAdapter()
    previous = object()
    previous_plots = make_plots()
    old_figure, _ = previous_plots.subplots("old")
    previous_plots.finish()
    values = np.linspace(-0.5, 0.5, 9)
    freqs = np.linspace(4.8, 5.4, 7)
    source = RunRecord(
        cfg=None,
        result=SpectrumResult(values, freqs, np.ones((9, 7), dtype=np.complex128)),
    )
    state.update_tab_result("tab1", source)
    state.update_tab_analyze("tab1", previous, previous_plots)
    plots = make_plots()
    handles = OperationHandles()
    service, _plugin, token = _start_onetone_plugin(state, source, plots, handles)
    assert service.cancel_interactive("tab1") is True
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "cancelled"
    assert service.get_interactive("tab1") is None
    assert state.get_tab("tab1").analysis.result is previous
    assert state.get_tab("tab1").analysis.plots is previous_plots
    assert tuple(plots) == ()
    output = BytesIO()
    old_figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG")
    previous_plots.release()


def test_onetone_gui_failed_frontend_analysis_settles_without_replacing_old_pane(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = make_state()
    tab = state.get_tab("tab1")
    tab.adapter = SpectrumAdapter()
    source = RunRecord(
        cfg=None,
        result=SpectrumResult(
            np.linspace(-0.5, 0.5, 9),
            np.linspace(4.8, 5.4, 7),
            np.ones((9, 7), dtype=np.complex128),
        ),
    )
    state.update_tab_result("tab1", source)
    previous = object()
    previous_plots = make_plots()
    old_figure, _ = previous_plots.subplots("old")
    previous_plots.finish()
    state.update_tab_analyze("tab1", previous, previous_plots)
    parameters = _flux_parameters()
    state.update_tab_result("tab1", source, parameters=parameters)
    state.set_context(replace(state.session_env, parameters=parameters))
    state.update_tab_analyze("tab1", previous, previous_plots)
    plots = make_plots()

    def fail_kernel(
        inputs: FluxPickInputs, committed: FluxPickState
    ) -> FluxPickAnalysis:
        np.testing.assert_array_equal(inputs.signals, source.result.signals)
        assert committed.flux_half == pytest.approx(-0.2)
        raise RuntimeError("frontend analysis failed")

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.flux_pick_plugin.analyze_flux_pick",
        fail_kernel,
    )
    handles = OperationHandles()
    service, _plugin, token = _start_onetone_plugin(state, source, plots, handles)
    assert service.finish_plugin("tab1") is True
    outcome = handles.poll(token)
    assert outcome is not None and outcome.status == "failed"
    assert service.get_interactive("tab1") is None
    assert state.get_tab("tab1").analysis.result is previous
    assert state.get_tab("tab1").analysis.plots is previous_plots
    assert tuple(plots) == ()
    image = BytesIO()
    old_figure.savefig(image, format="png")
    assert image.getvalue().startswith(b"\x89PNG")
    previous_plots.release()
