from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .node import RootNode


@dataclass
class PipeLineConfig:
    pass


@dataclass
class PipeLineContext:
    pass


class AbsPipeLinePass(ABC):
    @abstractmethod
    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode: ...


class PipeLine:
    def __init__(self, config: PipeLineConfig, passes: list[AbsPipeLinePass]):
        self.config = config
        self.passes = passes

    def __call__(self, ir: RootNode) -> tuple[RootNode, PipeLineContext]:
        ctx = PipeLineContext()
        for _pass in self.passes:
            ir = _pass.process(ir, ctx)
        return ir, ctx


def make_default_pipeline(config: PipeLineConfig, disable: bool = False) -> PipeLine:
    from .passes import (
        BranchCaseNormalizePass,
        ConstantLoopUnrollPass,
        IRStructureValidationPass,
        LabelDCEPass,
        LabelReferenceValidationPass,
        LoopInvariantHoistPass,
        PeepholePass,
        TimedInstructionMergePass,
        TimingSanityPass,
        ZeroDelayDCEPass,
    )

    return PipeLine(
        config,
        [
            IRStructureValidationPass(),
            BranchCaseNormalizePass(),
            ConstantLoopUnrollPass(),
            LoopInvariantHoistPass(),
            PeepholePass(),
            ZeroDelayDCEPass(),
            TimedInstructionMergePass(),
            TimingSanityPass(),
            LabelReferenceValidationPass(),
            LabelDCEPass(),
        ]
        if not disable
        else [],
    )
