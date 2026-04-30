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
    IRPulseByReg,
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
    FlattenSeq,
    FreshLabels,
    FuseAdjacentDelays,
    RemoveZeroDelays,
    ReorderPulseLikeByTime,
    UnrollShortLoops,
    ValidateInvariants,
    make_default_pipeline,
)
from .emiter import Emitter
