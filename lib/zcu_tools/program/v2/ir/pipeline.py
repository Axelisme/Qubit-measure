from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .node import RootNode


@dataclass
class PipeLineConfig:
    enable_unroll_loop: bool = True
    enable_dead_write: bool = True
    enable_dead_label: bool = True
    max_loop_unroll_count: int = 8


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


def make_default_pipeline(config: PipeLineConfig, disable: bool = False) -> PipeLine:
    from .passes import (
        DeadLabelEliminationPass,
        DeadWriteEliminationPass,
        TimedInstructionMergePass,
        UnrollLoopPass,
        ZeroDelayDCEPass,
    )

    return PipeLine(
        config,
        (
            [
                pass_
                for enabled, pass_ in [
                    (config.enable_unroll_loop, UnrollLoopPass()),
                    (config.enable_dead_write, DeadWriteEliminationPass()),
                    (config.enable_dead_label, DeadLabelEliminationPass()),
                    (True, ZeroDelayDCEPass()),
                    (True, TimedInstructionMergePass()),
                ]
                if enabled
            ]
            if not disable
            else []
        ),
    )
