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
    IRSendReadoutConfig,
    IRReadDmem,
    IRReadout,
    IRRegLoop,
    IRRegOp,
    IRSeq,
    RegOp,
)
from .pass_base import Pass, PassConfig, PassCtx
from .passes import (
    AlignBranchDispatch,
    EstimateDurations,
    FreshLabels,
    FuseAdjacentDelays,
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
    "FuseAdjacentDelays",
    "AlignBranchDispatch",
    "ValidateInvariants",
    "make_default_pipeline",
]
