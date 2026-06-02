from __future__ import annotations

import os
from abc import ABC, abstractmethod

from typing_extensions import Any, ClassVar, Generic, Sequence

from zcu_tools.gui.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalyzeRequest,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    NoAnalyzeParams,
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
    ``capabilities = AdapterCapabilities(supports_analysis=False)`` so the framework
    never routes analysis to them.

    Structurally satisfies ``zcu_tools.gui.adapter.ExpAdapterProtocol``; the GUI
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

    @abstractmethod
    def make_filename_stem(self, ctx: ExpContext) -> str:
        """Return the filename stem used by the default save path template."""

    # -- analysis (raising no-op default; override when supports_analysis) --

    def get_analyze_params(self, result: T_Result, ctx: ExpContext) -> T_AnalyzeParams:
        """Build the analyze parameter instance presented to the user.

        Default raises: an adapter that declares ``supports_analysis=True`` must
        override this. Adapters with ``supports_analysis=False`` are never routed
        here by the framework, so the raise is a Fast-Fail guard against a
        forgotten override, not a normal code path.
        """
        del result, ctx
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

    # -- shared implementation (provided once) -----------------------------

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        """Compose the static spec with context-derived default values."""
        return CfgSchema(spec=self.cfg_spec(), value=self.make_default_value(ctx))

    @classmethod
    def analyze_params_cls(cls) -> type:
        """Return the analyze-params dataclass type (static, no instance/result).

        Reflected from the concrete ``get_analyze_params`` return annotation, so
        agents can query the param schema without running an experiment. Falls
        back to ``NoAnalyzeParams`` when the return is not annotated.
        """
        import typing

        try:
            hints = typing.get_type_hints(cls.get_analyze_params)
        except Exception:
            return NoAnalyzeParams
        return hints.get("return", NoAnalyzeParams)

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
