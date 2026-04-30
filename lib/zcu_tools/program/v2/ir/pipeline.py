from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

from .node import IRNode


@dataclass
class PipeLineConfig:
    pass


@dataclass
class PipeLineContext:
    pass


class AbsPipeLinePass(ABC):
    @abstractmethod
    def process(self, ir: IRNode, ctx: PipeLineContext) -> IRNode: ...


class PipeLine:
    def __init__(self, config: PipeLineConfig, passes: list[AbsPipeLinePass]):
        self.config = config
        self.passes = passes

    def __call__(self, ir: IRNode) -> tuple[IRNode, PipeLineContext]:
        ctx = PipeLineContext()
        for _pass in self.passes:
            ir = _pass.process(ir, ctx)
        return ir, ctx


def make_default_pipeline(config: PipeLineConfig) -> PipeLine:
    return PipeLine(config, [])
