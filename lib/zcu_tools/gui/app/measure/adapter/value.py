"""Immutable experiment values consumed by the measure host."""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal

from typing_extensions import TypeVar

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import Container
from zcu_tools.experiment.artifacts import AnalysisDocument, AnalysisItem, DataItem
from zcu_tools.experiment.axes_spec import AxesSpec, GroupedAxesSpec
from zcu_tools.experiment.native_persistence import RestoredRun, restore_run
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import ArtifactKey

from ..writeback import WritebackItem
from .hooks import (
    AnalyzeHook,
    FrontendHook,
    FrontendMount,
    PluginHook,
    PostAnalyzeHook,
    RunHook,
    SessionFactory,
)
from .types import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResult,
    PostAnalyzeRequest,
    PostWritebackRequest,
    WritebackRequest,
)


@dataclass(frozen=True, kw_only=True)
class _PrimaryStage[C: ConfigBase, R, O, A: AnalyzeResult | None](ABC):
    """Hide only interactive state S, retaining source/options/result linkage."""

    options: Callable[[RunRecord[C, R], SessionEnv], O] | None = None
    writeback: (
        Callable[[WritebackRequest[RunRecord[C, R], A]], Sequence[WritebackItem]] | None
    ) = None

    @property
    @abstractmethod
    def mode(self) -> AnalysisMode: ...

    def analyze(self, req: AnalyzeRequest[RunRecord[C, R], O], *, plots: Plots) -> A:
        raise NotImplementedError("Interactive analysis requires a session")

    def prepare_interactive(
        self,
        req: AnalyzeRequest[RunRecord[C, R], O],
        create_session: SessionFactory,
        *,
        plots: Plots,
    ) -> FrontendMount:
        raise NotImplementedError("Synchronous analysis has no interactive frontend")


@dataclass(frozen=True, kw_only=True)
class Analysis[C: ConfigBase, R, O, A: AnalyzeResult | None](_PrimaryStage[C, R, O, A]):
    """Synchronous analysis and optional source-dependent editable inputs.

    options runs once when preparing a new source, never on analysis clicks or
    schema queries. None means an empty form, not a missing analysis capability.
    fn receives the user's edited object in its historical request. writeback
    proposes items only after successful adoption; failures remain isolated by
    the writeback owner.
    """

    fn: AnalyzeHook[C, R, O, A]

    @property
    def mode(self) -> AnalysisMode:
        return AnalysisMode.FIT

    def analyze(self, req: AnalyzeRequest[RunRecord[C, R], O], *, plots: Plots) -> A:
        return self.fn(req, plots=plots)


@dataclass(frozen=True, kw_only=True)
class Interactive[C: ConfigBase, R, O, A: AnalyzeResult | None, S](
    _PrimaryStage[C, R, O, A]
):
    """Pair a Qt-free plugin with its frontend without exposing S to Adapter.

    The host's generic factory opens and owns the session. Only this method
    connects its typed session to the matching frontend. The returned mount
    callback creates no state and leaves widget validation/cleanup to the UI.
    """

    plugin: PluginHook[C, R, O, S, A]
    frontend: FrontendHook[S, A]

    @property
    def mode(self) -> AnalysisMode:
        return AnalysisMode.INTERACTIVE

    def prepare_interactive(
        self,
        req: AnalyzeRequest[RunRecord[C, R], O],
        create_session: SessionFactory,
        *,
        plots: Plots,
    ) -> FrontendMount:
        plugin = self.plugin(req, plots=plots)
        session = create_session(plugin)
        return lambda env, finish, cancel: self.frontend(
            plugin, session, env, finish, cancel, plots=plots
        )


@dataclass(frozen=True, kw_only=True)
class PostAnalysis[
    C: ConfigBase,
    R,
    A: AnalyzeResult | None,
    O,
    P: AnalyzeResult | None,
]:
    """Independent second-stage inputs/result tied to a committed primary.

    options also receives the adopted primary. None gives an empty form.
    Failed execution does not replace the previous committed pane.
    """

    fn: PostAnalyzeHook[C, R, A, O, P]
    options: Callable[[RunRecord[C, R], A, SessionEnv], O] | None = None
    writeback: (
        Callable[[PostWritebackRequest[RunRecord[C, R], A, P]], Sequence[WritebackItem]]
        | None
    ) = None


T_Cfg = TypeVar("T_Cfg", bound=ConfigBase)
T_Result = TypeVar("T_Result")
T_Options = TypeVar("T_Options", default=None)
T_Analysis = TypeVar("T_Analysis", bound=AnalyzeResult | None, default=None)
T_PostOptions = TypeVar("T_PostOptions", default=None)
T_PostAnalysis = TypeVar("T_PostAnalysis", bound=AnalyzeResult | None, default=None)


@dataclass(frozen=True, kw_only=True, init=False)
class Adapter(
    Generic[T_Cfg, T_Result, T_Options, T_Analysis, T_PostOptions, T_PostAnalysis]
):
    """Caller-owned wiring, with no mutable per-tab state or construction I/O.

    spec owns native persistence and identity. make_form builds fresh captured
    input. run returns raw data; the operation host builds the RunRecord.
    guide is a Markdown path or None; guide() reads it on demand. The explicit
    initializer keeps that author input separate from the existing host method.
    """

    spec: AxesSpec[T_Result, T_Cfg] | GroupedAxesSpec[T_Result, T_Cfg]
    make_form: Callable[[FormContext], Form[T_Cfg]]
    run: RunHook[T_Cfg, T_Result]
    _guide: Path | None
    analysis: _PrimaryStage[T_Cfg, T_Result, T_Options, T_Analysis] | None
    post_analysis: (
        PostAnalysis[T_Cfg, T_Result, T_Analysis, T_PostOptions, T_PostAnalysis] | None
    )

    def __init__(
        self,
        *,
        spec: AxesSpec[T_Result, T_Cfg] | GroupedAxesSpec[T_Result, T_Cfg],
        make_form: Callable[[FormContext], Form[T_Cfg]],
        run: RunHook[T_Cfg, T_Result],
        guide: Path | None,
        analysis: _PrimaryStage[T_Cfg, T_Result, T_Options, T_Analysis] | None = None,
        post_analysis: PostAnalysis[
            T_Cfg, T_Result, T_Analysis, T_PostOptions, T_PostAnalysis
        ]
        | None = None,
    ) -> None:
        for name, value in (("spec", spec), ("make_form", make_form), ("run", run)):
            if value is None:
                raise TypeError(f"Adapter {name} is required")
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "make_form", make_form)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "_guide", guide)
        object.__setattr__(self, "analysis", analysis)
        object.__setattr__(self, "post_analysis", post_analysis)

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            analysis=AnalysisMode.NONE if self.analysis is None else self.analysis.mode,
            post_analysis=self.post_analysis is not None,
            load_data=True,
        )

    @property
    def experiment_tag(self) -> str:
        return self.spec.tag

    def guide(self) -> AdapterGuide:
        if self._guide is None:
            return AdapterGuide(markdown="(no guide written yet)", resource_dir=None)
        return AdapterGuide(
            markdown=self._guide.read_text(encoding="utf-8"),
            resource_dir=self._guide.parent,
        )

    def restore(self, stored: Container) -> RestoredRun[RunRecord[T_Cfg, T_Result]]:
        return restore_run(stored, spec=self.spec)

    def get_analyze_params(
        self, source: RunRecord[T_Cfg, T_Result], env: SessionEnv
    ) -> T_Options | None:
        if self.analysis is None or self.analysis.options is None:
            return None
        return self.analysis.options(source, env)

    def analyze(
        self,
        req: AnalyzeRequest[RunRecord[T_Cfg, T_Result], T_Options],
        *,
        plots: Plots,
    ) -> T_Analysis:
        if self.analysis is None:
            raise NotImplementedError("No analysis declared")
        return self.analysis.analyze(req, plots=plots)

    def make_interactive_frontend(
        self,
        req: AnalyzeRequest[RunRecord[T_Cfg, T_Result], T_Options],
        create_session: SessionFactory,
        *,
        plots: Plots,
    ) -> FrontendMount:
        if self.analysis is None:
            raise NotImplementedError("No analysis declared")
        return self.analysis.prepare_interactive(req, create_session, plots=plots)

    def get_post_analyze_params(
        self, source: RunRecord[T_Cfg, T_Result], primary: T_Analysis, env: SessionEnv
    ) -> T_PostOptions | None:
        if self.post_analysis is None or self.post_analysis.options is None:
            return None
        return self.post_analysis.options(source, primary, env)

    def post_analyze(
        self,
        req: PostAnalyzeRequest[RunRecord[T_Cfg, T_Result], T_Analysis, T_PostOptions],
        *,
        plots: Plots,
    ) -> T_PostAnalysis:
        if self.post_analysis is None:
            raise NotImplementedError("No post-analysis declared")
        return self.post_analysis.fn(req, plots=plots)

    def get_writeback_items(
        self, req: WritebackRequest[RunRecord[T_Cfg, T_Result], T_Analysis]
    ) -> Sequence[WritebackItem]:
        if self.analysis is None or self.analysis.writeback is None:
            return []
        return self.analysis.writeback(req)

    def get_post_writeback_items(
        self,
        req: PostWritebackRequest[
            RunRecord[T_Cfg, T_Result], T_Analysis, T_PostAnalysis
        ],
    ) -> Sequence[WritebackItem]:
        if self.post_analysis is None or self.post_analysis.writeback is None:
            return []
        return self.post_analysis.writeback(req)

    def analysis_item(
        self,
        key: ArtifactKey,
        result: object,
        parameters: object | None,
        *,
        pane: Literal["analysis", "post_analysis"],
    ) -> AnalysisItem:
        """Hand over one pane's historical inputs and summary under its assigned key.

        result is None for figures-only analysis, producing an empty summary,
        or implements AnalyzeResult.to_summary_dict. parameters is the
        dataclass instance used for that result, or None for no inputs, never
        the next edited draft. Unsupported input/result types raise TypeError;
        invalid/non-finite JSON raises ValueError. The datafile model isolates
        containers once; this projection neither reads live context nor writes files.
        """
        if result is not None and not isinstance(result, AnalyzeResult):
            raise TypeError("Analysis result must implement to_summary_dict")
        if parameters is None:
            inputs: dict[str, object] = {}
        elif dataclasses.is_dataclass(parameters) and not isinstance(parameters, type):
            inputs = {
                field.name: getattr(parameters, field.name)
                for field in dataclasses.fields(parameters)
            }
        else:
            raise TypeError(
                "Historical analysis parameters must be a dataclass or None"
            )
        document = AnalysisDocument.model_validate(
            {
                "pane": pane,
                "parameters": inputs,
                "summary": {} if result is None else result.to_summary_dict(),
            }
        )
        return AnalysisItem(key, document)

    def data_item(self, key: ArtifactKey, result: object, *, comment: str) -> DataItem:
        """Hand over a typed RunRecord under the framework-assigned data key.

        The envelope carries metadata while cfg and Result stay adapter-owned.
        The declared spec supplies the mapping. Preparation failures,
        including missing cfg, become failed data outputs. A non-record
        raises TypeError, which Save isolates the same way. No paths, format
        selection, metadata capture or I/O occurs. Arrays remain shared; do not
        mutate the record while saving.
        """
        if not isinstance(result, RunRecord):
            raise TypeError("Native saving requires a RunRecord")
        return DataItem.from_run(key, result, spec=self.spec, comment=comment)
