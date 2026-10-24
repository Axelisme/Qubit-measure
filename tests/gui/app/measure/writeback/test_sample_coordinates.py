"""Proposal-declared sample coordinates through analysis service entry points."""

from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    PostAnalysis,
    PostAnalyzeRequest,
    PostWritebackRequest,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.writeback import (
    ModuleWriteback,
    ParameterWriteback,
    SampleWriteback,
    WritebackDraft,
    WritebackItem,
    WritebackService,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.interactive import PluginDefinition
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui.app.measure._native_save_support import (
    PaneParams,
    PaneSummary,
    native_save_session,
)
from tests.gui.app.measure._record_support import require_record
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@pytest.fixture
def registered_components() -> Generator[None]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        yield
        assert registry_state() == installed
    finally:
        restore_registry(before)


def sample_proposals(coordinate_device: str | None) -> list[WritebackItem]:
    return [
        ParameterWriteback("general.description", "Description", proposed_value="fit"),
        ModuleWriteback(
            "Q1.pulses.tuned",
            "Pulse",
            proposal=PulseSpec(
                ch=0, nqz=1, waveform=ConstWaveformCfg(length=0.1), freq=100.0, gain=0.2
            ),
        ),
        SampleWriteback(
            "Q1",
            "Sample",
            proposed_value={"freq_mhz": 1250.0, "dev_value": 999, "dev_unit": "V"},
            coordinate_device=coordinate_device,
        ),
    ]


def sample_form(ctx: FormContext) -> Form[SavingCfg]:
    del ctx
    return Form(SavingCfg, {})


def sample_analyze(
    req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams], *, plots: Plots
) -> PaneSummary:
    return PaneSummary()


def sample_post_analyze(
    req: PostAnalyzeRequest[
        RunRecord[SavingCfg, SavingResult], PaneSummary, PaneParams
    ],
    *,
    plots: Plots,
) -> PaneSummary:
    return PaneSummary()


def sample_adapter(coordinate_device: str | None):
    def writeback(
        req: WritebackRequest[RunRecord[SavingCfg, SavingResult], PaneSummary],
    ) -> list[WritebackItem]:
        return sample_proposals(coordinate_device)

    def post_writeback(
        req: PostWritebackRequest[
            RunRecord[SavingCfg, SavingResult], PaneSummary, PaneSummary
        ],
    ) -> list[WritebackItem]:
        return sample_proposals(coordinate_device)

    return Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=sample_form,
        run=SavingCore().run,
        guide=None,
        analysis=Analysis(
            fn=sample_analyze,
            options=lambda source, env: PaneParams(),
            writeback=writeback,
        ),
        post_analysis=PostAnalysis(
            fn=sample_post_analyze,
            options=lambda source, primary, env: PaneParams(),
            writeback=post_writeback,
        ),
    )


@pytest.mark.parametrize("source", ["device_at_run", None])
def test_sample_coordinates_follow_the_captured_component_source(
    tmp_path: Path,
    qapp: QApplication,
    registered_components: None,
    source: str | None,
) -> None:
    with native_save_session(tmp_path, qapp) as native:
        native.context.add_component("Q1", "fake/drive/a", {"source": source})
        native.context.set_alias_rows((AliasRow("target", "Q1"),))
        captured = native.context.read_context()
        assert captured.value("$target.source") == source
        native.context.set_alias_rows((AliasRow("target", "later_target"),))
        writeback = WritebackService(
            ContextControlFacet(context=native.context), arb_names=lambda: ()
        )
        draft = writeback.create_draft(
            [
                SampleWriteback(
                    "Q1",
                    "Sample",
                    {"freq_mhz": 1250.0},
                    coordinate_device="$target.source",
                )
            ],
            parameters=captured,
            experiment_name="synthetic",
            run_id="captured-run",
            device_snapshot={"device_at_run": {"value": 0.00125, "value_unit": "A"}},
        )
        try:
            if source is None:
                assert draft.preview() == []
                assert draft.unavailable_reasons == (
                    "Q1: Sample writeback requires a Run $target.source name",
                )
            else:
                samples = [
                    item
                    for item in draft.preview()
                    if isinstance(item, SampleWriteback)
                ]
                assert [item.proposed_value for item in samples] == [
                    {"freq_mhz": 1250.0, "dev_value": 0.00125, "dev_unit": "A"}
                ]
                assert draft.unavailable_reasons == ()
        finally:
            writeback.teardown_draft(draft)


@pytest.mark.parametrize("stage", ["fit", "interactive", "post"])
@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
def test_analysis_uses_only_the_proposal_coordinate_device(
    tmp_path: Path,
    qapp: QApplication,
    registered_components: None,
    stage: str,
    declared: bool,
) -> None:
    with native_save_session(tmp_path, qapp) as native:
        native.context.add_component("Q1", "fake/drive/a", {})
        tab = native.state.get_tab("tab")
        registry = Registry()
        registry.register(
            "sample", sample_adapter("$coordinate_source" if declared else None)
        )
        tab.adapter = registry.create("sample")
        native.context.set_alias_rows(
            (
                AliasRow("coordinate_source", "device_at_run"),
                AliasRow("flux_dev", "unrelated_device"),
            )
        )
        tab.run.parameters = native.context.read_context()
        tab.run.result = replace(
            require_record(tab.run),
            metadata=native.metadata.model_copy(
                update={
                    "device_snapshot": {
                        "device_at_run": {"value": 0.00125, "value_unit": "A"},
                        "unrelated_device": {"value": 42.0, "value_unit": "V"},
                    }
                }
            ),
        )
        writeback = WritebackService(
            ContextControlFacet(context=native.context), arb_names=lambda: ()
        )
        runner = OperationRunner(
            None, native.handles, None, native.background, native.bus
        )
        analyze = AnalyzeService(
            native.state, runner, native.bus, writeback, native.handles
        )
        if stage == "interactive":
            analyze.start_plugin(
                "tab",
                PluginDefinition(
                    "pick", 0, (), lambda _state: None, lambda _state: PaneSummary()
                ),
                ManualOwnerScheduler(),
                analyze_params_instance=PaneParams(),
                plots=Plots(NonPresentingHost()),
            )
        else:
            analyze.start_analyze("tab", PaneParams(), plots=Plots(NonPresentingHost()))
        if stage == "post":
            assert native.background.quiesce()
            post = PostAnalyzeService(
                native.state, runner, native.bus, native.handles, writeback
            )
            post.start_post_analyze(
                "tab", PaneParams(), plots=Plots(NonPresentingHost())
            )
        native.context.set_alias_rows((AliasRow("coordinate_source", "later_device"),))
        tab.run.parameters = native.context.read_context()
        tab.run.result = replace(
            require_record(tab.run),
            metadata=native.metadata.model_copy(update={"device_snapshot": {}}),
        )
        if stage == "interactive":
            assert analyze.finish_plugin("tab")
        else:
            assert native.background.quiesce()
        draft = (
            tab.post_analysis.writeback_draft
            if stage == "post"
            else tab.analysis.writeback_draft
        )
        assert isinstance(draft, WritebackDraft)
        try:
            items = draft.preview()
            assert [item.target_name for item in items[:2]] == [
                "general.description",
                "Q1.pulses.tuned",
            ]
            samples = [item for item in items if isinstance(item, SampleWriteback)]
            if declared:
                assert [item.proposed_value for item in samples] == [
                    {"freq_mhz": 1250.0, "dev_value": 0.00125, "dev_unit": "A"}
                ]
                assert draft.unavailable_reasons == ()
            else:
                assert samples == []
                assert draft.unavailable_reasons == (
                    "Q1: Sample writeback requires an adapter-declared device alias",
                )
        finally:
            writeback.teardown_draft(draft)
            primary = tab.analysis.writeback_draft
            if isinstance(primary, WritebackDraft) and primary is not draft:
                writeback.teardown_draft(primary)
