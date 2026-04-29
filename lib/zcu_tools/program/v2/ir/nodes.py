"""IR node definitions for program/v2 compilation pipeline.

All IR nodes are frozen dataclasses to ensure immutability and enable
structural sharing. Composite nodes carry metadata about their subtree.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, Union, Literal
from enum import Enum
from qick.asm_v2 import QickParam


class RegOp(str, Enum):
    """Register operation codes."""
    ADD = "+"
    SUB = "-"
    AND = "&"
    OR = "|"
    XOR = "^"
    SL = "SL"  # shift left
    SR = "SR"  # shift right
    ASR = "ASR"  # arithmetic shift right


@dataclass(frozen=True)
class IRMeta:
    """Metadata attached to IR nodes.

    Preserved throughout passes to track source module, timing, and side effects.
    """
    source_module: str = ""  # hierarchical path for diagnostics
    duration: Optional[float] = None  # computed duration in us; None if contains QickParam

    extra: dict[str, Any] = field(default_factory=dict)  # scratch pad for passes


class IRNode(ABC):
    """Base class for all IR nodes. Frozen dataclasses ensure immutability."""
    meta: IRMeta


# ============================================================================
# Leaf IR Nodes (atomic operations)
# ============================================================================


@dataclass(frozen=True)
class IRPulse(IRNode):
    """Emit a single pulse on a channel."""
    ch: str
    pulse_name: str
    pre_delay: Union[float, QickParam]  # delay before pulse emission (us)
    post_delay: Union[float, QickParam]  # delay after pulse emission (us)
    advance: Union[float, QickParam]  # t increment after this module
    tag: Optional[str] = None
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRReadout(IRNode):
    """Trigger readout on channels."""
    ch: str
    ro_chs: Tuple[str, ...]  # readout channels to trigger
    pulse_name: str
    trig_offset: Union[float, QickParam]  # timing offset for trigger pulse (us)
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRDelay(IRNode):
    """Delay execution.

    Attributes:
        duration: delay duration in us (or float/QickParam expression)
        auto: if True, auto-aligns to max active channel timestamp before delay
        tag: optional barrier tag; tagged delays are not fused or moved by passes
    """
    duration: Union[float, QickParam, str]  # float/QickParam(us) or runtime reg(str)
    auto: bool = False
    gens: bool = True
    ros: bool = True
    tag: Optional[str] = None
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRSoftDelay(IRNode):
    """Timeline-only delay that does not emit any macro."""
    duration: Union[float, QickParam]
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRRegOp(IRNode):
    """Register arithmetic: dst = lhs <op> rhs."""
    dst: str  # destination register
    lhs: str  # left operand register
    op: RegOp  # operation
    rhs: Union[int, str, None]  # right operand: literal int, register, or None (copy)
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRReadDmem(IRNode):
    """Read value from DMEM (data memory)."""
    dst: str  # destination register
    addr: str  # address register or literal
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRCondJump(IRNode):
    """Conditional jump: if (arg1 <test> arg2) goto target."""
    target: str  # label to jump to
    arg1: str  # register to test
    test: str  # test code: "Z" (==0), "S" (<0), "N" (>0)
    op: Optional[str] = None  # optional operation: "-", "+"
    arg2: Union[int, str, None] = None  # operand for operation
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRJump(IRNode):
    """Unconditional jump."""
    target: str  # label to jump to
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRLabel(IRNode):
    """Label definition."""
    name: str
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRNop(IRNode):
    """No-operation instruction (used for padding)."""
    meta: IRMeta = field(default_factory=IRMeta)


# ============================================================================
# Composite IR Nodes (control flow)
# ============================================================================


@dataclass(frozen=True)
class IRSeq(IRNode):
    """Sequential composition of IR nodes."""
    body: Tuple[IRNode, ...] = field(default_factory=tuple)
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRLoop(IRNode):
    """Counted loop: repeat body n times.

    Translates to QICK OpenLoop/CloseLoop pair.
    """
    name: str
    n: int  # loop count (must be constant at IR level)
    body: IRNode
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRRegLoop(IRNode):
    """Register-driven loop: repeat body n_reg times.

    The loop count is stored in a runtime register.
    Translates to QICK OpenLoopReg/CloseLoopReg pair.
    """
    name: str
    n_reg: str  # register holding loop count
    body: IRNode
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRBranch(IRNode):
    """N-way dispatch based on register comparison.

    Emitter builds binary tree; pass adds NOP padding to align paths.

    Attributes:
        compare_reg: register to test
        arms: tuple of IR nodes for each branch
    """
    compare_reg: str
    arms: Tuple[IRNode, ...] = field(default_factory=tuple)
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRParallel(IRNode):
    """Emit all children from same start-t, then merge end-t by policy."""
    body: Tuple[IRNode, ...] = field(default_factory=tuple)
    end_policy: Literal["max", "index"] = "max"
    end_index: int = 0
    meta: IRMeta = field(default_factory=IRMeta)


# ============================================================================
# Type aliases for visitor pattern
# ============================================================================

IRLeaf = Union[
    IRPulse,
    IRReadout,
    IRDelay,
    IRSoftDelay,
    IRRegOp,
    IRReadDmem,
    IRCondJump,
    IRJump,
    IRLabel,
    IRNop,
]
IRComposite = Union[IRSeq, IRLoop, IRRegLoop, IRBranch, IRParallel]
