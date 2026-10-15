from __future__ import annotations

import dataclasses
from abc import ABC
from collections.abc import Callable, Sequence
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, cast

from zcu_tools.datafile import AnalysisDocument, StoredRun
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.native_persistence import (
    PreparedRunData,
    RestoredRun,
    prepare_run_data,
)
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    NoAnalyzeParams,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    PostWritebackRequest,
    RunRequest,
    SessionEnv,
    T_AnalyzeParams,
    T_AnalyzeResult,
    T_Cfg,
    T_PostAnalyzeResult,
    T_Result,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.types import AnalyzeResult
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.interactive import PluginDefinition, Session

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.ui.interactive_frontend import (
        InteractiveFrontend,
        InteractiveFrontendEnv,
    )
    from zcu_tools.plotting.plots import Plots

# Index of T_AnalyzeParams in BaseAdapter's generic parameter list
# (Cfg, Result, AnalyzeResult, AnalyzeParams).
_ANALYZE_PARAMS_GENERIC_INDEX = 3
_NO_GUIDE = AdapterGuide(markdown="(no guide written yet)", resource_dir=None)


def _analyze_params_generic_arg(cls: type) -> type:
    """Recover an adapter's analyze-params type from its declared 4th generic arg.

    Used when ``get_analyze_params`` is not overridden (its annotation is the
    unbound TypeVar): the concrete type is whatever the adapter wrote as
    ``BaseAdapter[..., AnalyzeParams]``. Falls back to ``NoAnalyzeParams`` when the
    arg is absent (PEP 696 default — adapters that omit the last two generic args)
    or not a plain class (e.g. still a TypeVar).
    """
    import typing

    for base in getattr(cls, "__orig_bases__", ()):
        args = typing.get_args(base)
        if len(args) > _ANALYZE_PARAMS_GENERIC_INDEX:
            arg = args[_ANALYZE_PARAMS_GENERIC_INDEX]
            if isinstance(arg, type):
                return arg
    return NoAnalyzeParams


def _can_construct_without_args(cls: type[Any]) -> bool:
    try:
        signature(cls).bind()
    except TypeError:
        return False
    except ValueError:
        return True
    return True


def _require_boolean_load_flag(value: object, adapter_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{adapter_name}.capabilities.load_data must be bool, "
            f"got {type(value).__name__}"
        )


def _validated_capabilities(value: object, adapter_name: str) -> AdapterCapabilities:
    # Plugin declarations can violate their static annotations at import time.
    if not isinstance(value, AdapterCapabilities):
        raise TypeError(
            f"{adapter_name}.capabilities must be an AdapterCapabilities instance"
        )
    _require_boolean_load_flag(value.load_data, adapter_name)
    return value


class BaseAdapter(ABC, Generic[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    """Shared implementation for experiment adapters.

    Concrete adapters subclass this and fill in the experiment-specific knowledge
    (``make_form`` and — when the experiment supports analysis —
    ``get_analyze_params``/``analyze``). Run delegation and persistence payload
    preparation are provided here once.

    The class is generic over the four experiment types. Adapters without analysis
    omit the last two parameters (PEP 696 defaults fill in ``NoAnalysisResult`` /
    ``NoAnalyzeParams``) and inherit the raising no-op analysis below; they declare
    ``capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)`` so the
    framework never routes analysis to them.

    Structurally satisfies ``zcu_tools.gui.app.measure.adapter.ExpAdapterProtocol``; the GUI
    holds adapters only through that generic-free Protocol.
    """

    exp_cls: ClassVar[type[Any]]
    # Native acquisition evidence is explicit, not inferred from filename policy.
    # Concrete adapters set the tag from their experiment's persistence spec.
    # See ExpAdapterProtocol.run for the associated RunRecord envelope contract.
    experiment_tag: ClassVar[str | None] = None
    # Most registered adapters use the canonical PersistableExperiment loader;
    # concrete no-load adapters declare ``load_data=False`` explicitly.
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(load_data=True)
    # Explicit UTF-8 Markdown source. None returns the unwritten-guide notice;
    # a declared path is read on each guide() call and I/O errors propagate.
    guide_path: ClassVar[Path | None] = None

    @staticmethod
    def make_form(ctx: FormContext) -> Form[T_Cfg]:
        """Build a fresh definition for captured Open/Reset sources.

        Concrete adapters provide this factory. No fixed definition fallback is
        supplied; missing factories raise NotImplementedError when used.
        """
        del ctx
        raise NotImplementedError("Adapters must implement make_form")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._validate_capability_contract()

    @classmethod
    def _is_method_implemented(cls, name: str) -> bool:
        return getattr(cls, name) is not getattr(BaseAdapter, name)

    @classmethod
    def _require_method(cls, name: str, reason: str, fix: str) -> None:
        if not cls._is_method_implemented(name):
            raise TypeError(
                f"{cls.__name__} {reason} but does not implement {name}(); {fix}."
            )

    @classmethod
    def _forbid_method(cls, name: str, reason: str, fix: str) -> None:
        if cls._is_method_implemented(name):
            raise TypeError(f"{cls.__name__} {reason} but implements {name}(); {fix}.")

    @classmethod
    def _validate_interactive_hooks(cls, analysis: AnalysisMode) -> None:
        if analysis is AnalysisMode.INTERACTIVE:
            for name in ("make_interactive_plugin", "make_interactive_frontend"):
                cls._require_method(
                    name,
                    "declares analysis=INTERACTIVE",
                    f"override {name}() or set analysis=AnalysisMode.FIT/NONE",
                )
        else:
            for name in ("make_interactive_plugin", "make_interactive_frontend"):
                cls._forbid_method(
                    name,
                    f"declares analysis={analysis.name}",
                    f"remove {name}() or set analysis=AnalysisMode.INTERACTIVE",
                )

    @classmethod
    def _validate_capability_contract(cls) -> None:
        """Fail fast when declared capabilities and implemented hooks disagree."""
        caps = _validated_capabilities(cls.capabilities, cls.__name__)
        analysis = caps.analysis

        if analysis is AnalysisMode.FIT:
            cls._require_method(
                "analyze",
                "declares analysis=FIT",
                "override analyze() or set analysis=AnalysisMode.NONE/INTERACTIVE",
            )
        elif analysis is AnalysisMode.INTERACTIVE:
            cls._forbid_method(
                "analyze",
                "declares analysis=INTERACTIVE",
                "remove analyze() or set analysis=AnalysisMode.FIT",
            )
        elif analysis is AnalysisMode.NONE:
            cls._forbid_method(
                "analyze",
                "declares analysis=NONE",
                "remove analyze() or set analysis=AnalysisMode.FIT",
            )
            cls._forbid_method(
                "get_analyze_params",
                "declares analysis=NONE",
                "remove get_analyze_params() or set analysis=AnalysisMode.FIT/INTERACTIVE",
            )
        else:
            raise TypeError(
                f"{cls.__name__} declares unsupported analysis capability {analysis!r}; "
                "use an AnalysisMode value"
            )

        cls._validate_interactive_hooks(analysis)

        if analysis is not AnalysisMode.NONE:
            params_cls = cls.analyze_params_cls()
            if not _can_construct_without_args(params_cls):
                cls._require_method(
                    "get_analyze_params",
                    f"declares analysis={analysis.name} with params "
                    f"{params_cls.__name__} that require values",
                    "override get_analyze_params() or give every param field a default",
                )

        if caps.post_analysis:
            if analysis is not AnalysisMode.FIT:
                raise TypeError(
                    f"{cls.__name__} declares post_analysis=True with analysis={analysis.name}; "
                    "post_analysis requires analysis=AnalysisMode.FIT"
                )
            cls._require_method(
                "get_post_analyze_params",
                "declares post_analysis=True",
                "override get_post_analyze_params() or set post_analysis=False",
            )
            cls._require_method(
                "post_analyze",
                "declares post_analysis=True",
                "override post_analyze() or set post_analysis=False",
            )
        else:
            cls._forbid_method(
                "get_post_analyze_params",
                "declares post_analysis=False",
                "remove get_post_analyze_params() or set post_analysis=True",
            )
            cls._forbid_method(
                "post_analyze",
                "declares post_analysis=False",
                "remove post_analyze() or set post_analysis=True",
            )
            cls._forbid_method(
                "get_post_writeback_items",
                "declares post_analysis=False",
                "remove get_post_writeback_items() or set post_analysis=True",
            )

        cls._validate_load_capability(enabled=caps.load_data)

    @classmethod
    def _validate_load_capability(cls, *, enabled: bool) -> None:
        """Require a declared loader, or forbid an override when loading is disabled."""
        # Loading is a capability gate, not a runtime probe. A concrete adapter
        # may provide its own loader, or it may opt into BaseAdapter.restore when the
        # experiment class can be constructed without arguments and exposes a
        # canonical callable loader. A disabled capability must not hide an
        # accidental adapter override.
        if enabled:
            if not cls._is_method_implemented("restore"):
                exp_cls = getattr(cls, "exp_cls", None)
                has_canonical_loader = isinstance(exp_cls, type) and (
                    _can_construct_without_args(exp_cls)
                    and callable(getattr(exp_cls, "from_stored", None))
                )
                if not has_canonical_loader:
                    raise TypeError(
                        f"{cls.__name__} declares load_data=True but has neither "
                        "a concrete restore() override nor a no-argument exp_cls with "
                        "a callable from_stored"
                    )
        else:
            cls._forbid_method(
                "restore",
                "declares load_data=False",
                "remove restore() or set load_data=True",
            )

    # -- experiment-specific contract (subclass must fill) -----------------

    @classmethod
    def guide(cls) -> AdapterGuide:
        """Read the declared UTF-8 Markdown guide without parsing its contents.

        guide_path names the source file; its parent is the resource directory
        for relative image references. Relative guide paths use the current
        working directory. None returns an unwritten-guide notice and no
        resource directory. Each call reads the file afresh. A declared missing
        file raises FileNotFoundError; other I/O and decoding errors propagate.
        Images are not read here.
        """
        if cls.guide_path is None:
            return _NO_GUIDE
        return AdapterGuide(
            markdown=cls.guide_path.read_text(encoding="utf-8"),
            resource_dir=cls.guide_path.parent,
        )

    def validate_run_request(self, req: RunRequest, cfg: T_Cfg) -> None:
        """Pure run preflight for adapter-specific constraints.

        GuardService calls this before opening an async run handle. The default is
        intentionally empty: cfg is the output already validated by this adapter's
        Form, not raw input. Adapters that need SoC-dependent checks override it
        and must not touch devices or mutate cfg/state. Validation failures
        propagate to GuardService's admission error translation.
        """
        del req, cfg

    # -- analysis (raising no-op default; override when analysis != NONE) --

    def get_analyze_params(self, result: T_Result, ctx: SessionEnv) -> T_AnalyzeParams:
        """Build the analyze parameter instance presented to the user.

        An adapter whose analyze-params are all-default-constructible (including
        ``NoAnalyzeParams``) inherits this default, which returns ``params_cls()``.
        An adapter whose params need values (a field without a default) must override;
        otherwise this Fast-Fails — a forgotten override, not a normal code path.
        Adapters with ``analysis=AnalysisMode.NONE`` are never routed here.
        """
        del result, ctx
        params_cls = type(self).analyze_params_cls()
        if _can_construct_without_args(params_cls):
            return params_cls()  # type: ignore[return-value]
        raise NotImplementedError(
            f"{type(self).__name__} declares analysis with params "
            f"{params_cls.__name__} that need values, but does not override "
            "get_analyze_params"
        )

    def analyze(
        self, req: AnalyzeRequest[T_Result, T_AnalyzeParams], *, plots: Plots
    ) -> T_AnalyzeResult:
        """Run analysis on a completed run result.

        Default raises — see ``get_analyze_params`` for the rationale.
        """
        del req, plots
        raise NotImplementedError(
            f"{type(self).__name__} declares analysis support but does not "
            "override analyze"
        )

    def make_interactive_plugin(
        self, req: AnalyzeRequest[T_Result, T_AnalyzeParams], *, plots: Plots
    ) -> PluginDefinition[Any, Any]:
        raise NotImplementedError("override make_interactive_plugin")

    def make_interactive_frontend(
        self,
        plugin: PluginDefinition[Any, Any],
        session: Session[Any],
        env: InteractiveFrontendEnv,
        request_finish: Callable[[], bool],
        request_cancel: Callable[[], bool],
        *,
        plots: Plots,
    ) -> InteractiveFrontend:
        raise NotImplementedError("override make_interactive_frontend")

    # -- post-analysis (raising no-op default; override when post_analysis) --
    #
    # Mirrors the primary analyze chain's param mechanism (dataclass + ParamMeta,
    # reflected by ``analyze_params_cls`` → describe/reconstruct), NOT a CfgSchema:
    # the post-analysis params flow through the exact same form/RPC plumbing as
    # the primary analyze params.

    @classmethod
    def post_analyze_params_cls(cls) -> type:
        """Return the post-analysis param dataclass type (static, no instance).

        Reflected from the concrete ``get_post_analyze_params`` return annotation,
        mirroring ``analyze_params_cls``. Falls back to ``NoAnalyzeParams`` when
        the return is not annotated.
        """
        import typing

        try:
            hints = typing.get_type_hints(cls.get_post_analyze_params)
        except Exception:
            return NoAnalyzeParams
        return hints.get("return", NoAnalyzeParams)

    def get_post_analyze_params(
        self, analyze_result: T_AnalyzeResult, ctx: SessionEnv
    ) -> Any:
        """Build the post-analysis param instance presented to the user.

        Default raises — an adapter declaring ``capabilities.post_analysis`` must
        override this (Fast-Fail guard; adapters without post-analysis are never
        routed here). Mirrors ``get_analyze_params`` but receives the primary
        analyze result (the post params may seed from the primary fit).
        """
        del analyze_result, ctx
        raise NotImplementedError(
            f"{type(self).__name__} declares post-analysis support but does not "
            "override get_post_analyze_params"
        )

    def post_analyze(
        self,
        req: PostAnalyzeRequest[T_Result, T_AnalyzeResult, Any],
        *,
        plots: Plots,
    ) -> PostAnalyzeResultBase:
        """Run a second analysis on top of the primary analyze result.

        Default raises — see ``get_post_analyze_params`` for the rationale. The
        request carries both the raw ``run_result`` and the primary
        ``analyze_result`` so the post-analysis can refine/recompute from either.
        """
        del req, plots
        raise NotImplementedError(
            f"{type(self).__name__} declares post-analysis support but does not "
            "override post_analyze"
        )

    # -- shared implementation (provided once) -----------------------------

    @classmethod
    def analyze_params_cls(cls) -> type:
        """Return the analyze-params dataclass type (static, no instance/result).

        Reflected from the concrete ``get_analyze_params`` return annotation, so
        agents can query the param schema without running an experiment. An adapter
        with tunable params overrides ``get_analyze_params`` with a concrete return
        annotation, which is read here directly. An adapter with no params does NOT
        override it (no boilerplate); its base annotation is the unbound
        ``T_AnalyzeParams`` TypeVar, so the type is instead recovered from the 4th
        generic arg (``BaseAdapter[..., NoAnalyzeParams]``), falling back to
        ``NoAnalyzeParams`` when absent (PEP 696 default) or unreadable.
        """
        import typing

        try:
            hints = typing.get_type_hints(cls.get_analyze_params)
        except Exception:
            return _analyze_params_generic_arg(cls)
        ret = hints.get("return", T_AnalyzeParams)
        if isinstance(ret, typing.TypeVar):
            # The default (un-overridden) get_analyze_params — recover the concrete
            # type from the class's declared 4th generic arg instead.
            return _analyze_params_generic_arg(cls)
        return ret

    def run(
        self,
        req: RunRequest,
        cfg: T_Cfg,
        *,
        context: RunContext,
    ) -> T_Result:
        """Pass this adapter's validated Form output to its core unchanged.

        req supplies admission inputs; context owns acquisition dependencies.
        No form rebuild, source capture or cfg validation happens here.
        Core execution failures propagate to the operation owner.
        """
        del req
        return cast(T_Result, self.exp_cls().run(cfg, context=context))

    def restore(self, stored: StoredRun) -> RestoredRun[T_Result]:
        """Restore already-read native data through the core's from_stored contract.

        stored contains validated generic data and original history, not current
        form inputs. Return the adapter-owned record and explicit cfg diagnostic.
        Missing from_stored or required constructor arguments raise
        NotImplementedError. Invalid data raises ValueError; core constructor
        failures propagate. No I/O, Labber fallback or history recapture occurs.
        """
        if not _can_construct_without_args(self.exp_cls):
            raise NotImplementedError(
                f"{type(self).__name__} does not support loading native result files"
            )
        exp = self.exp_cls()
        from_stored = getattr(exp, "from_stored", None)
        if not callable(from_stored):
            raise NotImplementedError(
                f"{type(self).__name__} does not support loading native result files"
            )
        try:
            return exp.from_stored(stored)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"{type(self).__name__} does not support loading native result files"
            ) from exc

    def get_writeback_items(
        self, req: WritebackRequest[T_Result, T_AnalyzeResult]
    ) -> Sequence[WritebackItem]:
        del req
        return []

    def get_post_writeback_items(
        self,
        req: PostWritebackRequest[T_Result, T_AnalyzeResult, T_PostAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Return optional proposals owned by a post-analysis pane.

        The default keeps post-analysis useful without a writeback surface. The
        workflow owns primary/post semantics; WritebackService only receives the
        returned items and never sees this request or a stage selector.
        """
        del req
        return []

    def capture_analysis_document(
        self,
        result: object,
        parameters: object | None,
        *,
        pane: Literal["analysis", "post_analysis"],
    ) -> AnalysisDocument:
        """Capture one completed pane's historical inputs and numeric summary.

        result must implement AnalyzeResult.to_summary_dict. parameters is the
        dataclass instance used for that result, or None for no inputs, never
        the next edited draft. Unsupported input/result types raise TypeError;
        invalid/non-finite JSON raises ValueError. The datafile model isolates
        containers once; this hook neither reads live context nor writes files.
        """
        if not isinstance(result, AnalyzeResult):
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
        return AnalysisDocument.model_validate(
            {"pane": pane, "parameters": inputs, "summary": result.to_summary_dict()}
        )

    def prepare_save_data(self, result: object, *, comment: str) -> PreparedRunData:
        """Map one completed RunRecord through this adapter's persistence spec.

        The record must contain this experiment's typed cfg and Result. Return
        the shared SI payload and cfg snapshot without paths, metadata capture
        or I/O. Invalid record/cfg/Result/comment and missing declarations raise
        the owning experiment's TypeError, ValueError or NotImplementedError.
        Arrays remain shared; do not mutate the record while saving.
        """
        if not isinstance(result, RunRecord):
            raise TypeError("Native saving requires a RunRecord")
        return prepare_run_data(
            result, spec=self.exp_cls().persistence_spec(), comment=comment
        )
