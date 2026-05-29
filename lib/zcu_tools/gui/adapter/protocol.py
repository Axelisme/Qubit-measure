from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from matplotlib.figure import Figure
from typing_extensions import Any, ClassVar, Generic, Optional, Sequence

from .types import (
    AdapterCapabilities,
    AnalyzeRequest,
    AnalyzeResultBase,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ExpContext,
    RunRequest,
    SaveDataRequest,
    SavePaths,
    T_AnalyzeParams,
    T_AnalyzeResult,
    T_Cfg,
    T_Result,
    WritebackItem,
    WritebackRequest,
)
from .validation import require_soc_handles


@dataclass
class NoAnalyzeParams:
    pass


@dataclass
class NoAnalysisResult(AnalyzeResultBase):
    figure: Optional[Figure] = None


class AbsExpAdapter(ABC, Generic[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    exp_cls: ClassVar[type[Any]]
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities()

    @abstractmethod
    def cfg_spec(self) -> "CfgSectionSpec":
        """Return the static cfg spec tree (no context required).

        The spec is the structural contract (field names, types, choices) and
        must not read any context. Default *values* live in make_default_value.
        Keeping spec context-free lets agents query an adapter's shape without
        building a tab/context.
        """

    @abstractmethod
    def make_default_value(self, ctx: ExpContext) -> "CfgSectionValue":
        """Build the default value tree, which may read ctx (md/ml/device)."""

    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        """Compose the static spec with context-derived default values."""
        return CfgSchema(spec=self.cfg_spec(), value=self.make_default_value(ctx))

    @abstractmethod
    def build_exp_cfg(self, raw_cfg: dict[str, object], req: RunRequest) -> T_Cfg:
        """Build experiment config from raw GUI dict."""

    def run(self, req: RunRequest, schema: CfgSchema) -> T_Result:
        raw_cfg = schema.to_raw_dict(req)
        cfg = self.build_exp_cfg(raw_cfg, req)
        if self.capabilities.requires_soc:
            soc, soccfg = require_soc_handles(req)
            return self.exp_cls().run(soc, soccfg, cfg)
        return self.exp_cls().run(req.soc, req.soccfg, cfg)

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

    @abstractmethod
    def get_analyze_params(self, result: T_Result, ctx: ExpContext) -> T_AnalyzeParams:
        """Build the analyze parameter instance presented to the user."""

    @abstractmethod
    def analyze(
        self, req: AnalyzeRequest[T_Result, T_AnalyzeParams]
    ) -> T_AnalyzeResult:
        """Run analysis on a completed run result."""

    def get_writeback_items(
        self, req: WritebackRequest[T_Result, T_AnalyzeResult]
    ) -> Sequence[WritebackItem]:
        del req
        return []

    @abstractmethod
    def make_filename_stem(self, ctx: ExpContext) -> str:
        """Return the filename stem used by the default save path template."""

    def make_default_save_paths(self, ctx: ExpContext) -> SavePaths:
        """Default save path policy shared by most adapters."""
        from zcu_tools.utils.datasaver import get_datafolder_path

        if not ctx.database_path:
            raise RuntimeError("ExpContext.database_path is required for save paths")
        if not ctx.result_dir:
            raise RuntimeError("ExpContext.result_dir is required for save paths")
        if not ctx.active_label:
            raise RuntimeError("ExpContext.active_label is required for save paths")

        stem = self.make_filename_stem(ctx)
        data_dir = get_datafolder_path(ctx.database_path)
        image_dir = os.path.join(ctx.result_dir, "exps", ctx.active_label, "image")
        return SavePaths(
            data_path=os.path.join(data_dir, stem),
            image_path=os.path.join(image_dir, f"{stem}.png"),
        )

    def make_save_paths(self, ctx: ExpContext) -> SavePaths:
        return self.make_default_save_paths(ctx)

    def save(self, req: SaveDataRequest[T_Result]) -> None:
        self.exp_cls().save(filepath=req.data_path, result=req.run_result)


class NoAnalysisAdapterMixin(
    Generic[T_Cfg, T_Result],
    AbsExpAdapter[T_Cfg, T_Result, NoAnalysisResult, NoAnalyzeParams],
    ABC,
):
    """Mixin providing no-op analyze defaults for adapters without analysis."""

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=True, supports_analysis=False
    )

    def get_analyze_params(self, result: T_Result, ctx: ExpContext) -> NoAnalyzeParams:
        del result, ctx
        return NoAnalyzeParams()

    def analyze(
        self, req: AnalyzeRequest[T_Result, NoAnalyzeParams]
    ) -> NoAnalysisResult:
        del req
        return NoAnalysisResult()
