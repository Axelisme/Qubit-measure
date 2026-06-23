from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Generic

from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    InteractiveHost,
    InteractiveSession,
    NoAnalyzeParams,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    RunRequest,
    SaveDataRequest,
    SavePaths,
    T_AnalyzeParams,
    T_AnalyzeResult,
    T_Cfg,
    T_Result,
    WritebackItem,
    WritebackRequest,
    require_soc_handles,
)

# Index of T_AnalyzeParams in BaseAdapter's generic parameter list
# (Cfg, Result, AnalyzeResult, AnalyzeParams).
_ANALYZE_PARAMS_GENERIC_INDEX = 3


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


class BaseAdapter(ABC, Generic[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    """Shared implementation for experiment adapters.

    Concrete adapters subclass this and fill in the experiment-specific knowledge
    (``cfg_spec``, ``make_default_value``, ``build_exp_cfg``, ``make_filename_stem``
    and — when the experiment supports analysis — ``get_analyze_params``/``analyze``).
    Everything else (spec+value composition, run delegation, save-path policy,
    save) is provided here once.

    The class is generic over the four experiment types. Adapters without analysis
    omit the last two parameters (PEP 696 defaults fill in ``NoAnalysisResult`` /
    ``NoAnalyzeParams``) and inherit the raising no-op analysis below; they declare
    ``capabilities = AdapterCapabilities(analysis=AnalysisMode.NONE)`` so the
    framework never routes analysis to them.

    Structurally satisfies ``zcu_tools.gui.app.main.adapter.ExpAdapterProtocol``; the GUI
    holds adapters only through that generic-free Protocol.
    """

    exp_cls: ClassVar[type[Any]]
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities()

    # Experiment cfg dataclass used by the default build_exp_cfg. Adapters whose
    # raw → cfg mapping is the common "flat dict through ml.make_cfg" shape just
    # set this and inherit build_exp_cfg. Adapters with a bespoke mapping (e.g.
    # extra kwargs, hand-built cfg) override build_exp_cfg and leave this None.
    ExpCfg_cls: ClassVar[Any] = None

    # -- experiment-specific contract (subclass must fill) -----------------

    @classmethod
    @abstractmethod
    def cfg_spec(cls) -> CfgSectionSpec:
        """Return the static cfg spec tree (no context, no instance state).

        The spec is the structural contract (field names, types, choices) and
        must not read any context. Default *values* live in make_default_value.
        """

    @classmethod
    def guide(cls) -> AdapterGuide:
        """Static human-facing orientation guide. Override per adapter.

        Honest default: an adapter that has not written one says so plainly
        (Fast-Fail spirit — surface the gap, do not fake content).
        """
        return AdapterGuide(
            behavior="(no guide written yet)",
            expects_md="",
            expects_ml="",
            typical_writeback="",
            recommended="",
        )

    @abstractmethod
    def make_default_value(self, ctx: ExpContext) -> CfgSectionValue:
        """Build the default value tree, which may read ctx (md/ml/device)."""

    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T_Cfg:
        """Build experiment config from the flat GUI raw dict.

        Default delegates to ``ml.make_cfg(raw_cfg, ExpCfg_cls)``. An adapter must
        either set the ``ExpCfg_cls`` ClassVar or override this; the raise is a
        Fast-Fail guard against forgetting both (mirrors the analysis no-ops).
        """
        if self.ExpCfg_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set ExpCfg_cls or override build_exp_cfg"
            )
        return req.ml.make_cfg(raw_cfg, self.ExpCfg_cls)

    def validate_run_request(self, req: RunRequest, raw_cfg: dict[str, object]) -> None:
        """Pure run preflight for adapter-specific constraints.

        GuardService calls this before opening an async run handle. The default is
        intentionally empty because most adapters have no constraints beyond
        lowering + model construction; adapters that need SoC-dependent checks
        override it and must not touch devices or mutate cfg/state.
        """
        del req, raw_cfg

    @abstractmethod
    def make_filename_stem(self, ctx: ExpContext) -> str:
        """Return the filename stem used by the default save path template."""

    # -- analysis (raising no-op default; override when analysis != NONE) --

    def get_analyze_params(self, result: T_Result, ctx: ExpContext) -> T_AnalyzeParams:
        """Build the analyze parameter instance presented to the user.

        An adapter whose analysis takes tunable params overrides this. An adapter
        whose analyze is a look-at-the-curve render with NO params declares
        ``NoAnalyzeParams`` as its 4th generic arg and inherits this default, which
        returns the empty ``NoAnalyzeParams()`` (no override boilerplate). When the
        adapter declares a real (non-``NoAnalyzeParams``) param type but forgets the
        override, this Fast-Fails — a forgotten override, not a normal code path.
        Adapters with ``analysis=AnalysisMode.NONE`` are never routed here.
        """
        del result, ctx
        params_cls = type(self).analyze_params_cls()
        if params_cls is NoAnalyzeParams:
            return NoAnalyzeParams()  # type: ignore[return-value]
        raise NotImplementedError(
            f"{type(self).__name__} declares analysis support but does not "
            "override get_analyze_params"
        )

    def analyze(
        self, req: AnalyzeRequest[T_Result, T_AnalyzeParams]
    ) -> T_AnalyzeResult:
        """Run analysis on a completed run result.

        Default raises — see ``get_analyze_params`` for the rationale.
        """
        del req
        raise NotImplementedError(
            f"{type(self).__name__} declares analysis support but does not "
            "override analyze"
        )

    def setup_interactive_analysis(
        self,
        req: AnalyzeRequest[T_Result, T_AnalyzeParams],
        host: InteractiveHost,
    ) -> InteractiveSession:
        """Set up an interactive analysis on the host's figure and return the
        in-progress session (used only by ``analysis=AnalysisMode.INTERACTIVE``).

        Default raises — only INTERACTIVE adapters override it; FIT/NONE adapters
        are never routed here (Fast-Fail guard against a forgotten override).
        """
        del req, host
        raise NotImplementedError(
            f"{type(self).__name__} declares INTERACTIVE analysis but does not "
            "override setup_interactive_analysis"
        )

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
        self, analyze_result: T_AnalyzeResult, ctx: ExpContext
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
    ) -> PostAnalyzeResultBase:
        """Run a second analysis on top of the primary analyze result.

        Default raises — see ``get_post_analyze_params`` for the rationale. The
        request carries both the raw ``run_result`` and the primary
        ``analyze_result`` so the post-analysis can refine/recompute from either.
        """
        del req
        raise NotImplementedError(
            f"{type(self).__name__} declares post-analysis support but does not "
            "override post_analyze"
        )

    # -- shared implementation (provided once) -----------------------------

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        """Compose the static spec with context-derived default values.

        Validates the result: an adapter's ``make_default_value`` must return a
        structurally-complete, spec-compliant value tree (every field present,
        literals/types/choices valid) — a violation fast-fails here, pointing at
        the offending adapter rather than surfacing as a later lowering error.
        """
        schema = CfgSchema(spec=self.cfg_spec(), value=self.make_default_value(ctx))
        schema.validate(ctx.ml)
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

    def run(self, req: RunRequest, schema: CfgSchema) -> T_Result:
        raw_cfg = schema.to_raw_dict(req.md, req.ml)
        cfg = self.build_exp_cfg(raw_cfg, req)
        if self.capabilities.requires_soc:
            soc, soccfg = require_soc_handles(req)
            return self.exp_cls().run(soc, soccfg, cfg)
        return self.exp_cls().run(req.soc, req.soccfg, cfg)

    def get_writeback_items(
        self, req: WritebackRequest[T_Result, T_AnalyzeResult]
    ) -> Sequence[WritebackItem]:
        del req
        return []

    def make_default_save_paths(self, ctx: ExpContext) -> SavePaths:
        """Default save path policy shared by most adapters."""
        if not ctx.database_path:
            raise RuntimeError("ExpContext.database_path is required for save paths")
        if not ctx.result_dir:
            raise RuntimeError("ExpContext.result_dir is required for save paths")
        if not ctx.active_label:
            raise RuntimeError("ExpContext.active_label is required for save paths")

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

    def make_save_paths(self, ctx: ExpContext) -> SavePaths:
        return self.make_default_save_paths(ctx)

    def save(self, req: SaveDataRequest[T_Result]) -> None:
        self.exp_cls().save(filepath=req.data_path, result=req.run_result)
