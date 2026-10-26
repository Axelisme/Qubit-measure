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
from zcu_tools.resources.entry import ArtifactKey, SaveContext

from ..summary import summarize_analysis
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
    PostAnalyzeRequest,
    PostWritebackRequest,
    WritebackRequest,
)


@dataclass(frozen=True, kw_only=True)
class _PrimaryStage[C: ConfigBase, R, O, A](ABC):
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
class Analysis[C: ConfigBase, R, O, A](_PrimaryStage[C, R, O, A]):
    """Frozen synchronous primary analysis declaration.

    fn receives AnalyzeRequest with the retained RunRecord, edited options and
    fixed parameters, plus keyword-only plots. It returns a dataclass instance,
    pydantic model or Mapping, or None for figures-only output. Callback failures
    propagate; the host adopts results and preserves previous output on failure.

    options, inherited from the primary stage, is an optional (source, env)
    generator. The host calls it once when preparing a successful Run/Load
    source and retains the returned editable object for form/schema generation.
    None means no generator and an empty form, not no analysis capability.
    The value does not cache or rerun it on clicks or schema queries.

    writeback, also inherited, receives WritebackRequest after successful
    adoption and returns proposal items, not applied resource changes. None
    means no proposals. Generator and writeback failures propagate unchanged;
    the preparation and writeback owners isolate them from committed output.
    """

    fn: AnalyzeHook[C, R, O, A]

    @property
    def mode(self) -> AnalysisMode:
        """Return FIT, identifying synchronous primary analysis."""
        return AnalysisMode.FIT

    def analyze(self, req: AnalyzeRequest[RunRecord[C, R], O], *, plots: Plots) -> A:
        """Call fn with retained request inputs and plots; propagate its failures.

        Return its numeric result or None for figures-only output. This call
        neither commits the result nor generates options or writeback proposals.
        """
        return self.fn(req, plots=plots)


@dataclass(frozen=True, kw_only=True)
class Interactive[C: ConfigBase, R, O, A, S](_PrimaryStage[C, R, O, A]):
    """Frozen interactive primary declaration, with state S hidden from Adapter.

    plugin receives the retained AnalyzeRequest and keyword-only plots and
    returns PluginDefinition[S, A]. Its terminal result may be None for
    figures-only output. frontend receives that plugin, its matching Session[S],
    the background environment and finish/cancel callbacks, plus plots. It
    returns an object for the UI to validate, mount and clean up.

    options, inherited from the primary stage, is an optional (source, env)
    generator. The host invokes it once when preparing a successful Run/Load
    source and retains the editable object for form/schema generation. None
    gives an empty form without disabling interactive analysis.

    writeback, also inherited, receives WritebackRequest after successful result
    adoption and returns proposal items without applying them. None means no
    proposals. All callbacks propagate failures to their caller; preparation
    and writeback isolation belong to their respective host owners.

    The host's generic SessionFactory opens and owns the session, including
    cleanup if mounting fails. This value only pairs plugin/session/frontend;
    it does not settle operations, dispose sessions or own widgets.
    """

    plugin: PluginHook[C, R, O, S, A]
    frontend: FrontendHook[S, A]

    @property
    def mode(self) -> AnalysisMode:
        """Return INTERACTIVE, identifying deferred primary analysis."""
        return AnalysisMode.INTERACTIVE

    def prepare_interactive(
        self,
        req: AnalyzeRequest[RunRecord[C, R], O],
        create_session: SessionFactory,
        *,
        plots: Plots,
    ) -> FrontendMount:
        """Prepare a plugin/session pair now and return its deferred UI mount.

        req retains source, edited options and fixed parameters; plots belongs
        to this operation. Invoke plugin, then create_session, exactly once.
        The returned (env, finish, cancel) callback invokes frontend with that
        pair on the UI owner thread. It does not create another session.

        Plugin/factory failures propagate now; frontend failures propagate when
        mounted. There is no rollback here. The host owns session disposal and
        operation settlement, including cleanup after a failed frontend call.
        """
        plugin = self.plugin(req, plots=plots)
        session = create_session(plugin)
        return lambda env, finish, cancel: self.frontend(
            plugin, session, env, finish, cancel, plots=plots
        )


@dataclass(frozen=True, kw_only=True)
class PostAnalysis[
    C: ConfigBase,
    R,
    A,
    O,
    P,
]:
    """Frozen second-stage declaration tied to an adopted primary result.

    fn receives PostAnalyzeRequest with the retained RunRecord, adopted primary,
    edited post options and fixed parameters, plus keyword-only plots. It returns
    a dataclass instance, pydantic model or Mapping, or None for figures-only
    output. Callback failures propagate; the host preserves the previous
    committed pane when execution fails.

    options is an optional (source, primary, env) generator. The host invokes it
    once when preparing the form after the corresponding primary succeeds.
    Retain its returned editable object and use its type for form/schema
    generation. None means an empty form, not no post-analysis capability.

    writeback receives PostWritebackRequest after successful adoption and returns
    proposal items without applying them. None means no proposals. Generator and
    writeback failures propagate to the preparation/writeback owners, which
    isolate failures from committed output. This value performs no caching.
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
T_Analysis = TypeVar("T_Analysis", default=None)
T_PostOptions = TypeVar("T_PostOptions", default=None)
T_PostAnalysis = TypeVar("T_PostAnalysis", default=None)
T_Input = TypeVar("T_Input", bound=ConfigBase, default=T_Cfg)


@dataclass(frozen=True, kw_only=True, init=False)
class Adapter(
    Generic[
        T_Cfg, T_Result, T_Options, T_Analysis, T_PostOptions, T_PostAnalysis, T_Input
    ]
):
    """Frozen caller-owned wiring, without per-tab state or construction I/O.

    spec declares native persistence and experiment identity. make_form accepts
    FormContext and builds a fresh Form of adapter-owned input. run(inp, *,
    context) adapts that input and returns RunRecord with the actual experiment
    cfg and result. The host attaches evidence without replacing cfg or result.
    Form/run callback failures propagate to the calling owner.
    save_stem receives SaveContext and returns an exact filename stem; the host
    adds no prefix or run_id. The default joins the dotted adapter name,
    ordered role values with hyphens and the last six run_id characters.
    Invalid stems fail SaveLayout validation; callback failures propagate.

    guide is a required Markdown Path or None, read only by guide(). analysis
    is Analysis, Interactive or None, selecting one primary mode or none.
    post_analysis is PostAnalysis or None, independently enabling the second
    stage. The host owns preparation, retained options, result adoption, failure
    isolation and all session/widget lifetimes.
    """

    spec: AxesSpec[T_Result, T_Cfg] | GroupedAxesSpec[T_Result, T_Cfg]
    make_form: Callable[[FormContext], Form[T_Input]]
    run: RunHook[T_Input, T_Cfg, T_Result]
    _guide: Path | None
    analysis: _PrimaryStage[T_Cfg, T_Result, T_Options, T_Analysis] | None
    post_analysis: (
        PostAnalysis[T_Cfg, T_Result, T_Analysis, T_PostOptions, T_PostAnalysis] | None
    )

    def __init__(
        self,
        *,
        spec: AxesSpec[T_Result, T_Cfg] | GroupedAxesSpec[T_Result, T_Cfg],
        make_form: Callable[[FormContext], Form[T_Input]],
        run: RunHook[T_Input, T_Cfg, T_Result],
        guide: Path | None,
        analysis: _PrimaryStage[T_Cfg, T_Result, T_Options, T_Analysis] | None = None,
        post_analysis: PostAnalysis[
            T_Cfg, T_Result, T_Analysis, T_PostOptions, T_PostAnalysis
        ]
        | None = None,
    ) -> None:
        """Store the keyword-only declarations without invoking callbacks or I/O.

        See the class docstring for each input. None for spec, make_form or run
        raises TypeError. guide must be supplied but may be None; both analysis
        stages default to None. No guide existence or callback result is checked.
        """
        for name, value in (("spec", spec), ("make_form", make_form), ("run", run)):
            if value is None:
                raise TypeError(f"Adapter {name} is required")
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "make_form", make_form)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "_guide", guide)
        object.__setattr__(self, "analysis", analysis)
        object.__setattr__(self, "post_analysis", post_analysis)

    def save_stem(self, context: SaveContext) -> str:
        """Return this adapter's spec.save_stem(context) unchanged.

        context contains the full adapter path and historical run facts. The
        default uses dotted name, hyphen-joined roles and run_id tail. Customize
        by supplying a spec with another save_stem callable. Callback failures
        propagate; SaveLayout validates the returned stem before writing.
        """
        return self.spec.save_stem(context)

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Derive primary mode and post presence without executing callbacks.

        Every value requires SoC for Run and permits a Load attempt. load_data
        does not promise compatibility with any particular stored Container.
        """
        return AdapterCapabilities(
            analysis=AnalysisMode.NONE if self.analysis is None else self.analysis.mode,
            post_analysis=self.post_analysis is not None,
            load_data=True,
        )

    @property
    def experiment_tag(self) -> str:
        """Return spec.tag as the persistence identity, without constructing core."""
        return self.spec.tag

    def guide(self) -> AdapterGuide:
        """Read the declared UTF-8 Markdown and return text/resource directory.

        A None guide returns an unwritten-guide notice and resource_dir=None.
        Each call reads afresh. File I/O and decoding failures propagate; no
        images are read and no fallback replaces a failed declared guide.
        """
        if self._guide is None:
            return AdapterGuide(markdown="(no guide written yet)", resource_dir=None)
        return AdapterGuide(
            markdown=self._guide.read_text(encoding="utf-8"),
            resource_dir=self._guide.parent,
        )

    def restore(self, stored: Container) -> RestoredRun[RunRecord[T_Cfg, T_Result]]:
        """Restore an already-read Container with this spec, without I/O.

        Return the typed record and optional cfg_issue. Invalid metadata/tag,
        variables or cfg envelope raise ValueError/TypeError; spec callback
        failures propagate. Incompatible typed cfg emits CfgUnavailableWarning
        and returns cfg=None plus cfg_issue, retaining valid data and metadata.
        No live configuration substitutes for historical values.
        """
        return restore_run(stored, spec=self.spec)

    def get_analyze_params(
        self, source: RunRecord[T_Cfg, T_Result], env: SessionEnv
    ) -> T_Options | None:
        """Generate primary inputs from retained source and captured session env.

        Return the options object, or None when no primary/generator is declared.
        The host calls once while preparing each successful Run/Load source,
        retains the object for schema queries and passes edited inputs to analysis.
        This method does not cache. Generator failures propagate unchanged.
        """
        if self.analysis is None or self.analysis.options is None:
            return None
        return self.analysis.options(source, env)

    def analyze(
        self,
        req: AnalyzeRequest[RunRecord[T_Cfg, T_Result], T_Options],
        *,
        plots: Plots,
    ) -> T_Analysis:
        """Run synchronous primary analysis on req using the operation's plots.

        req contains retained source, edited options and fixed parameters.
        Return the callback result, including None for figures-only output.
        Raise NotImplementedError for no primary or an Interactive declaration.
        Callback failures propagate; this method does not adopt results.
        """
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
        """Prepare interactive analysis and return a deferred UI mount callback.

        req carries retained source, edited options and fixed parameters; plots
        is the operation's plot host. create_session opens/owns the plugin's
        matching session. The returned (env, finish, cancel) callback creates
        the frontend, not another session. The UI validates/mounts its object.

        Raise NotImplementedError for no primary or synchronous Analysis.
        Plugin/factory failures propagate now, frontend failures at mount time.
        The host must dispose sessions and settle operations on every terminal
        path, including mount failure; this value performs no cleanup.
        """
        if self.analysis is None:
            raise NotImplementedError("No analysis declared")
        return self.analysis.prepare_interactive(req, create_session, plots=plots)

    def get_post_analyze_params(
        self, source: RunRecord[T_Cfg, T_Result], primary: T_Analysis, env: SessionEnv
    ) -> T_PostOptions | None:
        """Generate post inputs from retained source, adopted primary and env.

        Return the options object, or None when no post stage/generator exists.
        The host calls once when preparing the corresponding successful primary's
        form, retains the object for schema queries and passes edited inputs to
        post_analyze. No caching occurs; generator failures propagate unchanged.
        """
        if self.post_analysis is None or self.post_analysis.options is None:
            return None
        return self.post_analysis.options(source, primary, env)

    def post_analyze(
        self,
        req: PostAnalyzeRequest[RunRecord[T_Cfg, T_Result], T_Analysis, T_PostOptions],
        *,
        plots: Plots,
    ) -> T_PostAnalysis:
        """Run post fn with req's source, adopted primary and edited post inputs.

        plots belongs to this operation. Return the callback result, including
        None for figures-only output. Missing post_analysis raises
        NotImplementedError. Callback failures propagate without result adoption;
        the host owns preservation of the previous committed pane.
        """
        if self.post_analysis is None:
            raise NotImplementedError("No post-analysis declared")
        return self.post_analysis.fn(req, plots=plots)

    def get_writeback_items(
        self, req: WritebackRequest[RunRecord[T_Cfg, T_Result], T_Analysis]
    ) -> Sequence[WritebackItem]:
        """Propose primary writeback from req's adopted result and fixed context.

        Call after successful adoption. Return the callback's proposal sequence,
        or an empty list when no primary/writeback callback exists. No resource
        changes are applied here. Callback failures propagate to the writeback
        owner, which isolates them from the already committed analysis.
        """
        if self.analysis is None or self.analysis.writeback is None:
            return []
        return self.analysis.writeback(req)

    def get_post_writeback_items(
        self,
        req: PostWritebackRequest[
            RunRecord[T_Cfg, T_Result], T_Analysis, T_PostAnalysis
        ],
    ) -> Sequence[WritebackItem]:
        """Propose post writeback from req's source, primary, post result and ctx.

        Call after successful post adoption. Return the callback's proposal
        sequence, or an empty list when no post/writeback callback exists.
        No resource changes are applied. Callback failures propagate to the
        writeback owner, which isolates them from committed pane output.
        """
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

        result is the committed native result, or None for figures-only output.
        The framework converter retains declared fields and replaces non-finite
        numbers with null. Invalid-number paths belong to live RPC, not this
        historical document. parameters is the dataclass used for that result,
        or None for no inputs, never the next edited draft. Unsupported inputs
        or result fields raise TypeError with the converter's field path.
        This projection neither reads live context nor writes files.
        """
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
                "summary": summarize_analysis({} if result is None else result).summary,
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
