"""IR (Intermediate Representation) for program/v2 compilation pipeline.

This module provides the IR layer between Module and Macro, enabling
IR-level optimizations before emission to QICK macros.

Pipeline: Module.lower() → IR → PassPipeline → Emitter → prog.macro_list
"""

from .nodes import (
    IRMeta,
    IRNode,
    IRPulse,
    IRReadout,
    IRDelay,
    IRRegOp,
    IRReadDmem,
    IRCondJump,
    IRJump,
    IRLabel,
    IRNop,
    IRSeq,
    IRLoop,
    IRRegLoop,
    IRBranch,
)
from .pass_base import Pass, PassConfig, PassCtx

__all__ = [
    "IRMeta",
    "IRNode",
    "IRPulse",
    "IRReadout",
    "IRDelay",
    "IRRegOp",
    "IRReadDmem",
    "IRCondJump",
    "IRJump",
    "IRLabel",
    "IRNop",
    "IRSeq",
    "IRLoop",
    "IRRegLoop",
    "IRBranch",
    "Pass",
    "PassConfig",
    "PassCtx",
]
