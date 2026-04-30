"""IR node definitions for program/v2 compilation pipeline.

All IR nodes are frozen dataclasses. Leaf nodes carry a single `t` field
meaning "relative to the current ref_t". Only IRDelay / IRDelayAuto advance
ref_t (mirroring QICK hardware semantics).
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from qick.asm_v2 import QickParam


class RegOp(str, Enum):
    """Register operation codes."""

    ADD = "+"
    SUB = "-"
    AND = "&"
    OR = "|"
    XOR = "^"
    SL = "SL"
    SR = "SR"
    ASR = "ASR"


@dataclass(frozen=True)
class IRMeta:
    """Metadata attached to IR nodes."""

    extra: Dict[str, Any] = field(default_factory=dict)


class IRNode(ABC):
    """Base class for all IR nodes."""

    meta: IRMeta


# ============================================================================
# Leaf IR Nodes
# ============================================================================


@dataclass(frozen=True)
class IRPulse(IRNode):
    """Emit a single pulse at time t relative to the current ref_t."""

    ch: int
    pulse_id: str
    t: Union[float, QickParam]
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRSendReadoutConfig(IRNode):
    """Send readout configuration at time t."""

    ch: int
    readout_id: str
    t: Union[float, QickParam]
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRReadout(IRNode):
    """Trigger readout; t is relative to current ref_t (trig_offset folded in by caller)."""

    ch: str
    ro_chs: Tuple[int, ...]
    t: Union[float, QickParam]
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRPulseByReg(IRNode):
    """Play pulse from runtime-computed wmem address register."""

    ch: int
    addr_reg: str
    t: Union[float, QickParam]
    flat_top_pulse: bool = False
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRDelay(IRNode):
    """Delay: advances ref_t by t."""

    t: Union[float, QickParam]
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRDelayAuto(IRNode):
    """HW-align then advance ref_t by t (usually 0).

    The `t` field has three valid forms:
      - float / int: static delay value lowered to ``prog.delay_auto``.
      - QickParam: parameterized delay lowered to ``prog.delay_auto``.
      - str: name of a runtime register; lowered to ``prog.delay_reg_auto``.
    """

    t: Union[float, QickParam, str] = 0.0
    gens: bool = True
    ros: bool = True
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRRegOp(IRNode):
    """Register arithmetic: dst = lhs <op> rhs."""

    dst: str
    lhs: str
    op: RegOp
    rhs: Union[int, str, None]
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRReadDmem(IRNode):
    """Read value from DMEM."""

    dst: str
    addr: str
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRCondJump(IRNode):
    """Conditional jump: if (arg1 <test> arg2) goto target."""

    target: str
    arg1: str
    test: str
    op: Optional[str] = None
    arg2: Union[int, str, None] = None
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRJump(IRNode):
    """Unconditional jump."""

    target: str
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

    `name` is BOTH the loop identifier and the counter register name.
    The body may read/write that register; ``UnrollShortLoops`` detects this
    and synthesizes counter init/increment ops when unrolling.
    """

    name: str
    n: int
    body: IRNode
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRRegLoop(IRNode):
    """Register-driven loop: repeat body n_reg times."""

    name: str
    n_reg: str
    body: IRNode
    meta: IRMeta = field(default_factory=IRMeta)


@dataclass(frozen=True)
class IRBranch(IRNode):
    """N-way dispatch based on register comparison.

    The emitter lowers this with a binary-search dispatch tree comparing
    ``compare_reg`` against arm-index midpoints. Selection semantics:
      - compare_reg <  0          -> first arm
      - compare_reg >= len(arms)  -> last arm
      - otherwise                 -> arms[compare_reg]

    Requires len(arms) >= 2 (validated by ValidateInvariants).
    """

    compare_reg: str
    arms: Tuple[IRNode, ...] = field(default_factory=tuple)
    meta: IRMeta = field(default_factory=IRMeta)


# ============================================================================
# Type aliases
# ============================================================================

IRLeaf = Union[
    IRPulse,
    IRSendReadoutConfig,
    IRReadout,
    IRPulseByReg,
    IRDelay,
    IRDelayAuto,
    IRRegOp,
    IRReadDmem,
    IRCondJump,
    IRJump,
    IRLabel,
    IRNop,
]
IRComposite = Union[IRSeq, IRLoop, IRRegLoop, IRBranch]
