from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

from typing_extensions import Generic

from .types import (
    AnalyzeRequest,
    CfgSchema,
    ExpContext,
    RunRequest,
    SaveDataRequest,
    SavePaths,
    T_AnalyzeParams,
    T_AnalyzeResult,
    T_Result,
    WritebackItem,
    WritebackRequest,
)

if TYPE_CHECKING:
    from zcu_tools.experiment.cfg_model import ExpCfgModel


class AbsExpAdapter(ABC, Generic[T_Result, T_AnalyzeResult, T_AnalyzeParams]):
    exp_cls: Optional[type[Any]] = None

    @abstractmethod
    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        """Build a default CfgSchema from ctx."""

    @abstractmethod
    def build_exp_cfg(
        self, raw_cfg: dict[str, object], req: RunRequest
    ) -> "ExpCfgModel":
        """Convert lowered raw cfg into the concrete experiment config model."""

    def run(self, req: RunRequest, schema: CfgSchema) -> T_Result:
        raise NotImplementedError("Run method is not implemented for this adapter")

    @abstractmethod
    def get_analyze_params(self, result: T_Result, ctx: ExpContext) -> T_AnalyzeParams:
        """Return a dataclass instance with current analysis parameters."""

    @abstractmethod
    def analyze(
        self,
        req: AnalyzeRequest[T_Result, T_AnalyzeParams],
    ) -> T_AnalyzeResult:
        """Run analysis."""

    @abstractmethod
    def get_writeback_items(
        self, req: WritebackRequest[T_Result, T_AnalyzeResult]
    ) -> Sequence[WritebackItem]: ...

    @abstractmethod
    def make_filename_stem(self, ctx: ExpContext) -> str:
        """Return the filename stem used by the default save path template."""

    def make_default_save_paths(self, ctx: ExpContext) -> SavePaths:
        """Default save path policy shared by most adapters."""
        from zcu_tools.utils.datasaver import create_datafolder

        if not ctx.database_path:
            raise RuntimeError("ExpContext.database_path is required for save paths")
        if not ctx.result_dir:
            raise RuntimeError("ExpContext.result_dir is required for save paths")
        if not ctx.active_label:
            raise RuntimeError("ExpContext.active_label is required for save paths")

        stem = self.make_filename_stem(ctx)
        data_dir = create_datafolder(ctx.database_path)
        image_dir = os.path.join(ctx.result_dir, "exps", ctx.active_label, "image")
        os.makedirs(image_dir, exist_ok=True)
        return SavePaths(
            data_path=os.path.join(data_dir, stem),
            image_path=os.path.join(image_dir, f"{stem}.png"),
        )

    def make_save_paths(self, ctx: ExpContext) -> SavePaths:
        return self.make_default_save_paths(ctx)

    @abstractmethod
    def save(self, req: SaveDataRequest[T_Result]) -> None: ...
