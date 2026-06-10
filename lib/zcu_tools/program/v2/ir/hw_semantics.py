"""Hardware-implicit register names and predicates for tProc v2.

Centralised so passes & instruction definitions share a single source of truth.
Pure constants and predicates — must not import any IR module to avoid cycles.
"""

from __future__ import annotations

from typing import Optional

TIMED_BASE_REG = (
    "s14"  # implicit base for WPORT/WMEM/DPORT/TRIG; written by TIME inc_ref/set_ref
)
USR_TIME_REG = "s11"  # read by WAIT time / TIME updt
STATUS_REG = "s10"  # read by WAIT port_dt / div_rdy / div_dt / qpa_*
ADDR_REG = "s15"  # big-PMEM jump target / dispatch base

WAVE_REGS = frozenset({"w0", "w1", "w2", "w3", "w4", "w5"})

# Volatile system regs (s0..s15). s15 is included because it is the dispatch
# address register — writes to it must not be reordered past counter increments.
VOLATILE_REGS = frozenset({f"s{i}" for i in range(16)})

# General regs
GENERAL_REGS = frozenset({f"r{i}" for i in range(32)})  # r0~r31

# Big-jump threshold: when pmem_size > 2048 words, label jumps need the
# two-instruction indirect idiom (REG_WR s15 label; JUMP [s15]).
BIG_JUMP_PMEM_THRESHOLD = 2**11


def needs_big_jump(pmem_size: int | None) -> bool:
    """Return True when pmem_size exceeds the direct-jump address range."""
    return pmem_size is not None and pmem_size > BIG_JUMP_PMEM_THRESHOLD
