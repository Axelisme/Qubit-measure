"""Hardware-implicit register names for tProc v2.

Centralised so passes & instruction definitions share a single source of truth.
Pure constants — must not import any IR module to avoid cycles.
"""

from __future__ import annotations

TIMED_BASE_REG = (
    "s14"  # implicit base for WPORT/WMEM/DPORT/TRIG; written by TIME inc_ref/set_ref
)
USR_TIME_REG = "s11"  # read by WAIT time / TIME updt
STATUS_REG = "s10"  # read by WAIT port_dt / div_rdy / div_dt / qpa_*
ADDR_REG = "s15"  # big-PMEM jump target / dispatch base

WAVE_REGS = frozenset({"w0", "w1", "w2", "w3", "w4", "w5"})

# Volatile system regs (s0..s14). Mirrors labels.is_volatile_reg_name as a set.
VOLATILE_REGS = frozenset({f"s{i}" for i in range(15)})

# General regs
GENERAL_REGS = frozenset({f"r{i}" for i in range(15)})
