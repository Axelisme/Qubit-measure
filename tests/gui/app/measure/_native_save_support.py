"""Synchronous native Save through real state, writers and ledger."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.artifacts import ArtifactRef, SaveReport
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    PostAnalyzeRequest,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.adapter.types import (
    AnalyzeResultBase,
    PostAnalyzeResultBase,
)
from zcu_tools.gui.app.measure.artifacts import SaveControlFacet, SaveService
from zcu_tools.gui.app.measure.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import CommandOrigin, PointUseRequest
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui.app.measure._form_fakes import make_form

GUI_ORIGIN = CommandOrigin("gui")


@dataclass
class PaneParams:
    gain: float = 2.0


@dataclass
class PaneSummary(AnalyzeResultBase):
    mean: float = 3.0


@dataclass
class PostSummary(PostAnalyzeResultBase):
    correction: float = 0.5


class SavingAdapter(
    BaseAdapter[
        SavingCfg,
        RunRecord[SavingCfg, SavingResult],
        PaneSummary,
        PaneParams,
        PostSummary,
        PaneParams,
    ]
):
    """Real BaseAdapter mapping with an offline typed experiment declaration."""

    exp_cls = SavingCore

    @staticmethod
    def make_form(ctx: FormContext) -> Form[SavingCfg]:
        del ctx
        return Form(SavingCfg, {})

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=False, requires_soc=False
    )
    experiment_tag: ClassVar[str | None] = SavingCore.AXES_SPEC.tag

    def run(
        self,
        req: RunRequest,
        cfg: SavingCfg,
        *,
        context: RunContext,
    ) -> RunRecord[SavingCfg, SavingResult]:
        """Acquire a typed record using only the offline fake backend."""
        return RunRecord(cfg, SavingCore().run(cfg, context=context))


class AnalysisSavingAdapter(SavingAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT,
        post_analysis=True,
        load_data=True,
        requires_soc=False,
    )

    def analyze(
        self,
        req: AnalyzeRequest[RunRecord[SavingCfg, SavingResult], PaneParams],
        *,
        plots: Plots,
    ) -> PaneSummary:
        return PaneSummary()

    def get_post_analyze_params(
        self, analyze_result: PaneSummary, ctx: SessionEnv
    ) -> PaneParams:
        return PaneParams()

    def post_analyze(
        self,
        req: PostAnalyzeRequest[
            RunRecord[SavingCfg, SavingResult], PaneSummary, PaneParams
        ],
        *,
        plots: Plots,
    ) -> PostSummary:
        return PostSummary()


@dataclass
class NativeSaveSession:
    entry: ResultEntry
    context: ContextService
    state: State
    control: SaveControlFacet
    handles: OperationHandles
    bus: BaseEventBus
    metadata: RunMetadata
    completions: list[SaveDataFinishedPayload]
    artifact_completions: list[SaveArtifactsFinishedPayload]
    background: BackgroundRunner

    @property
    def native_data_path(self) -> Path:
        """Return this fixture's canonical native destination, even before Save."""
        experiment = self.state.get_tab("tab").adapter_name.replace("/", "_")
        run_name = f"{experiment}_{self.metadata.run_id}"
        return self.entry.database_path / "runs" / run_name / "data.h5"

    def capture_parameters(
        self, aliases: Mapping[str, str] | None
    ) -> ParameterContext | None:
        """Use current aliases, retaining EMPTY deferred reads without overrides."""
        if self.state.session_env.parameters is None and not aliases:
            return None
        return self.context.read_context(aliases=aliases)

    def save(self, *, origin: CommandOrigin = GUI_ORIGIN) -> SaveReport:
        """Observe native writes and State publication before the public call returns."""
        report = self.control.save_data("tab", comment="first note", origin=origin)
        assert isinstance(report, SaveReport)
        assert not self.state.is_tab_busy("tab")
        return report

    def save_artifacts(self, refs: tuple[ArtifactRef, ...] | None = None) -> SaveReport:
        """Observe the complete whole-artifact report from the synchronous command."""
        report = self.control.save_artifacts("tab", artifacts=refs, origin=GUI_ORIGIN)
        assert isinstance(report, SaveReport)
        assert not self.state.is_tab_busy("tab")
        return report


@contextmanager
def native_save_session(
    tmp_path: Path, qapp: QApplication
) -> Generator[NativeSaveSession]:
    entry = ResultEntry.create(
        "original", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    entry.new_point("p")
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    context.install_entry(entry)
    context_control = ContextControlFacet(context=context)
    context_control.use_point(PointUseRequest("p"))
    captured = context_control.read_context()
    storage = captured.capture_storage()
    cfg = SavingCfg()
    record = RunRecord(cfg, SavingCore().run(cfg, context=None))
    metadata = RunMetadata(
        run_id="20261008T000000Z-a1b2c3",
        experiment=SavingCore.AXES_SPEC.tag,
        started_at="2026-10-08T00:00:00+00:00",
        finished_at="2026-10-08T00:01:00+00:00",
        completion="complete",
        snapshot=RunSnapshot(
            entry_id=storage.entry_id, point=storage.point, roles=storage.roles
        ),
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
    )
    state.add_tab(
        "tab",
        Session(
            adapter_name="saving",
            adapter=AnalysisSavingAdapter(),
            cfg=make_form(SavingAdapter.make_form(FormContext(None)), state=state),
        ),
    )
    state.update_tab_result(
        "tab", replace(record, metadata=metadata), parameters=captured
    )
    handles = OperationHandles()
    background = BackgroundRunner()
    service = SaveService(state, bus)
    control = SaveControlFacet(
        state=state,
        guard=GuardService(state),
        save=service,
        context=context,
    )
    completions: list[SaveDataFinishedPayload] = []
    subscription = bus.subscribe(SaveDataFinishedPayload, completions.append)
    artifact_completions: list[SaveArtifactsFinishedPayload] = []
    batch_subscription = bus.subscribe(
        SaveArtifactsFinishedPayload, artifact_completions.append
    )
    try:
        yield NativeSaveSession(
            entry,
            context,
            state,
            control,
            handles,
            bus,
            metadata,
            completions,
            artifact_completions,
            background,
        )
    finally:
        subscription.unsubscribe()
        batch_subscription.unsubscribe()
        background.quiesce()
        background.deleteLater()
        qapp.processEvents()


def declare_panes(
    session: NativeSaveSession, names: tuple[str, ...] = ("fit", "residual")
) -> Plots:
    """Publish finished pane resources through the same State terminal commands."""
    plots = Plots(NonPresentingHost())
    for name in names:
        _, axes = plots.subplots(name)
        axes.plot([0, 1], [1, 2])
    plots.finish(present=False)
    session.state.update_tab_analyze(
        "tab", PaneSummary(), plots, analyze_params_instance=PaneParams(2.0)
    )
    session.state.update_tab_post_analyze(
        "tab", PostSummary(), None, post_analyze_params_instance=PaneParams(4.0)
    )
    return plots
