"""IR (Intermediate Representation) for program/v2 compilation pipeline.

Pipeline: Module.ir_run(builder, t) → IRBuilder → PassPipeline → Emitter → prog.macro_list
"""

from .builder import IRBuilder
from .nodes import (
    IRBranch,
    IRCondJump,
    IRDelay,
    IRDelayAuto,
    IRJump,
    IRLabel,
    IRLoop,
    IRMeta,
    IRNode,
    IRNop,
    IRPulse,
    IRPulseWmemReg,
    IRReadDmem,
    IRReadout,
    IRRegLoop,
    IRRegOp,
    IRSendReadoutConfig,
    IRSeq,
    RegOp,
)
from .pass_base import Pass, PassConfig, PassCtx
from .passes import (
    EstimateDurations,
    FreshLabels,
    FuseAdjacentDelays,
    RemoveZeroDelays,
    ReorderPulseLikeByTime,
    UnrollShortLoops,
    ValidateInvariants,
    make_default_pipeline,
)

__all__ = [
    "IRBuilder",
    "IRMeta",
    "IRNode",
    "IRPulse",
    "IRPulseWmemReg",
    "IRSendReadoutConfig",
    "IRReadout",
    "IRDelay",
    "IRDelayAuto",
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
    "RegOp",
    "Pass",
    "PassConfig",
    "PassCtx",
    "FreshLabels",
    "EstimateDurations",
    "UnrollShortLoops",
    "RemoveZeroDelays",
    "FuseAdjacentDelays",
    "ReorderPulseLikeByTime",
    "ValidateInvariants",
    "make_default_pipeline",
]
