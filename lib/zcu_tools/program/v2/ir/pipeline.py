from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .node import RootNode


@dataclass
class PipeLineConfig:
    enable_unroll_loop: bool = True
    enable_dead_write: bool = True
    enable_dead_label: bool = True
    max_loop_unroll_count: int = 8
    pmem_budget: int | None = None

    # Unified cycle cost model
    cost_default: int = 1
    cost_wmem: int = 4
    cost_dmem: int = 4
    cost_jump_flush: int = 4


# Global default configuration for quick debugging and toggling optimization options.
DEFAULT_PIPELINE_CONFIG = PipeLineConfig()


@dataclass
class PipeLineContext:
    config: PipeLineConfig = field(default_factory=PipeLineConfig)
    pass_stats: dict[str, int] = field(default_factory=dict)
    analysis_cache: dict[str, Any] = field(default_factory=dict)


class AbsPipeLinePass(ABC):
    @abstractmethod
    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode: ...


class PipeLine:
    def __init__(self, config: PipeLineConfig, passes: list[AbsPipeLinePass]):
        self.config = config
        self.passes = passes

    def __call__(self, ir: RootNode) -> tuple[RootNode, PipeLineContext]:
        ctx = PipeLineContext(config=self.config)
        for _pass in self.passes:
            ir = _pass.process(ir, ctx)
        return ir, ctx


def make_default_pipeline(pmem_capacity: int) -> PipeLine:
    from .passes import (
        DeadLabelEliminationPass,
        DeadWriteEliminationPass,
        TimedInstructionMergePass,
        UnrollSmallLoopPass,
        ZeroDelayDCEPass,
    )

    config = deepcopy(DEFAULT_PIPELINE_CONFIG)
    if config.pmem_budget is None:
        config.pmem_budget = int(0.8 * pmem_capacity)

    return PipeLine(
        config,
        [
            UnrollSmallLoopPass(),
            DeadWriteEliminationPass(),
            DeadLabelEliminationPass(),
            ZeroDelayDCEPass(),
            TimedInstructionMergePass(),
        ],
    )
