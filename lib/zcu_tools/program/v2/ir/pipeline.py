from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .node import RootNode


@dataclass
class PipeLineConfig:
    disable_all_opt: bool = False
    enable_unroll_loop: bool = True
    enable_dead_write: bool = True
    enable_dead_label: bool = True
    enable_zero_delay_dce: bool = True
    enable_timed_instruction_merge: bool = True
    pmem_budget: int | None = None

    # Hard cap on the unroll factor k. For register-driven loops k is also
    # rounded down to the nearest power of 2 (Phase 8D).
    max_unroll_factor: int = 8

    # Maximum number of REG_WR words allowed in the dispatch shift-add
    # sequence. If the body_words multiply cannot fit, register-driven
    # unroll falls back to no-unroll.
    max_dispatch_words: int = 8

    # Unified cycle cost model
    cost_default: int = 1
    cost_wmem: int = 4
    cost_dmem: int = 4
    cost_jump_flush: int = 1000


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
        if self.config.disable_all_opt:
            return ir, ctx
        for _pass in self.passes:
            ir = _pass.process(ir, ctx)
        return ir, ctx


def make_default_pipeline(pmem_capacity: int) -> PipeLine:
    from .passes import (
        DeadLabelEliminationPass,
        DeadWriteEliminationPass,
        TimedMergePass,
        UnrollSmallLoopPass,
        ZeroDelayDCEPass,
    )

    config = deepcopy(DEFAULT_PIPELINE_CONFIG)
    if config.pmem_budget is None:
        config.pmem_budget = int(0.8 * pmem_capacity)

    return PipeLine(
        config,
        [
            # Cleaning passes run before Unroll so that body_size estimates
            # (used for k selection and jump-table stride) reflect the true
            # post-cleanup word count:
            #   • ZeroDelayDCE   — removes TIME #0 words that inflate body_size
            #   • TimedMerge     — collapses adjacent TIME #N into one word
            #   • DeadWriteElim  — removes overwritten register writes
            # DeadLabelElimination runs after Unroll because unrolling
            # introduces new label references that must be visible first.
            ZeroDelayDCEPass(),
            TimedMergePass(),
            DeadWriteEliminationPass(),
            UnrollSmallLoopPass(),
            DeadLabelEliminationPass(),
        ],
    )
