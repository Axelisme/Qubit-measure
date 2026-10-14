from __future__ import annotations

import dataclasses
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, cast

from zcu_tools.datafile import AnalysisDocument, RunMetadata
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.native_persistence import (
    PreparedRunData,
    load_run_with_metadata,
    prepare_run_data,
)
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterCfgDefinition,
    AdapterGuide,
    AnalysisMode,
    AnalyzeRequest,
    LoadDataRequest,
    NoAnalyzeParams,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    PostWritebackRequest,
    RunRequest,
    SaveDataRequest,
    SavePaths,
    SessionEnv,
    T_AnalyzeParams,
    T_AnalyzeResult,
    T_Cfg,
    T_PostAnalyzeResult,
    T_Result,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.lowering import validate_schema
from zcu_tools.gui.app.measure.adapter.types import AnalyzeResult
from zcu_tools.gui.cfg import CfgSchema
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
_NO_GUIDE = AdapterGuide(
    behavior="(no guide written yet)",
    expects_md="",
    expects_ml="",
    typical_writeback="",
    recommended="",
)


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
    (``form``, ``build_exp_cfg``, ``make_filename_stem``
    and — when the experiment supports analysis — ``get_analyze_params``/``analyze``).
    Everything else (run delegation, save-path policy,
    save) is provided here once.

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
    guide_text: ClassVar[AdapterGuide] = _NO_GUIDE

    # Concrete adapters publish one core-derived definition; the host owns state.
    form: Form[T_Cfg]

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
        # may provide its own loader, or it may opt into BaseAdapter.load when the
        # experiment class can be constructed without arguments and exposes a
        # canonical callable loader. A disabled capability must not hide an
        # accidental adapter override.
        if enabled:
            if not cls._is_method_implemented("load"):
                exp_cls = getattr(cls, "exp_cls", None)
                has_canonical_loader = isinstance(exp_cls, type) and (
                    _can_construct_without_args(exp_cls)
                    and callable(getattr(exp_cls, "load", None))
                )
                if not has_canonical_loader:
                    raise TypeError(
                        f"{cls.__name__} declares load_data=True but has neither "
                        "a concrete load() override nor a no-argument exp_cls with "
                        "a callable canonical loader"
                    )
        else:
            cls._forbid_method(
                "load",
                "declares load_data=False",
                "remove load() or set load_data=True",
            )

    # -- experiment-specific contract (subclass must fill) -----------------

    @classmethod
    def cfg_definition(cls) -> AdapterCfgDefinition:
        """Return a legacy independent schema definition; not used by tab forms.

        Adapters that still use make_default_cfg must declare this hook. The
        core-derived tab path uses form instead. Missing legacy hooks raise
        NotImplementedError rather than guessing a form/schema conversion.
        """
        raise NotImplementedError("Legacy cfg_definition is not declared")

    @classmethod
    def guide(cls) -> AdapterGuide:
        """Static human-facing orientation guide.

        Concrete adapters keep the guide prose local as ``guide_text``. The
        method remains the framework-facing protocol entry point so GUI/MCP
        consumers do not need to know how each adapter stores its prose.

        Honest default: an adapter that has not written prose says so plainly
        (Fast-Fail spirit — surface the gap, do not fake content).
        """
        return cls.guide_text

    def build_exp_cfg(
        self, raw_cfg: Mapping[str, ExperimentCfgValue], req: RunRequest
    ) -> T_Cfg:
        """Validate an accepted Form dump as this adapter's concrete core cfg.

        raw_cfg contains the already lowered values from TabFormResource.accept,
        not authored input or library references. req remains the Run request;
        this conversion needs no sources, assets or hardware. Return the type
        published by form.core_cfg. Core ValidationError propagates for invalid
        mappings. The mapping and request are not modified or reassembled.
        """
        del req
        return self.form.core_cfg.model_validate(raw_cfg)

    def validate_run_request(
        self, req: RunRequest, raw_cfg: Mapping[str, ExperimentCfgValue]
    ) -> None:
        """Pure run preflight for adapter-specific constraints.

        GuardService calls this before opening an async run handle. The default is
        intentionally empty because most adapters have no constraints beyond
        lowering + model construction; adapters that need SoC-dependent checks
        override it and must not touch devices or mutate cfg/state.
        """
        del req, raw_cfg

    @abstractmethod
    def make_filename_stem(self, ctx: SessionEnv) -> str:
        """Return the filename stem used by the default save path template."""

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

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        """Instantiate fresh defaults and validate their static input contract.

        ctx supplies the fixed parameter capture. Unbound role defaults may
        retain expressions and empty linked references for cfg Invalid
        publication; they are not runnable. Definition, source and malformed
        input failures propagate without fallback or context mutation.
        """
        schema = type(self).cfg_definition().instantiate(ctx)
        validate_schema(schema, ctx.parameters)
        return schema

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
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> T_Result:
        cfg = self.build_exp_cfg(raw_cfg, req)
        return cast(T_Result, self.exp_cls().run(cfg, context=context))

    def load(self, req: LoadDataRequest) -> tuple[object, RunMetadata]:
        """Read req.data_path as an exact native file using this core's spec.

        Return the typed cfg/Result record and its full historical metadata.
        req.parameters is the Load-time capture, not a replacement for file history.
        Missing persistence declarations or required constructor arguments raise
        NotImplementedError. Native validation raises ValueError; I/O and core
        constructor failures propagate. No Labber fallback or history recapture.
        """
        if not _can_construct_without_args(self.exp_cls):
            raise NotImplementedError(
                f"{type(self).__name__} does not support loading native result files"
            )
        exp = self.exp_cls()
        persistence_spec = getattr(exp, "persistence_spec", None)
        if not callable(persistence_spec):
            raise NotImplementedError(
                f"{type(self).__name__} does not support loading native result files"
            )
        try:
            spec = exp.persistence_spec()
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"{type(self).__name__} does not support loading native result files"
            ) from exc
        return load_run_with_metadata(Path(req.data_path), spec=spec)

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

    def make_default_save_paths(self, ctx: SessionEnv) -> SavePaths:
        """Default save path policy shared by most adapters."""
        if not ctx.database_path:
            raise RuntimeError("SessionEnv.database_path is required for save paths")
        if not ctx.result_dir:
            raise RuntimeError("SessionEnv.result_dir is required for save paths")
        if not ctx.active_label:
            raise RuntimeError("SessionEnv.active_label is required for save paths")

        stem = self.make_filename_stem(ctx)
        # ctx.database_path is already the dated data folder
        # (Database/chip/qub/YYYY/MM/Data_MMDD; derive_project_paths owns the date),
        # so join the filename directly — do NOT re-append the date here.
        data_dir = ctx.database_path
        image_dir = os.path.join(ctx.result_dir, "exps", ctx.active_label, "image")
        # Data filename carries the flux label (single_qubit.md:
        # '{stem}@{em.label}') so the same experiment at different flux points
        # stays distinct within a day's data folder.
        return SavePaths(
            data_path=os.path.join(data_dir, f"{stem}@{ctx.active_label}"),
            image_path=os.path.join(image_dir, f"{stem}.png"),
        )

    def make_save_paths(self, ctx: SessionEnv) -> SavePaths:
        return self.make_default_save_paths(ctx)

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

    def save(self, req: SaveDataRequest[T_Result]) -> None:
        self.exp_cls().save(req.run_result, Path(req.data_path), comment=req.comment)
