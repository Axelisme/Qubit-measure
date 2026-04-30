from __future__ import annotations

import logging

from qick.asm_v2 import QickParam
from typing_extensions import Optional, Self, TypeAlias, Union

from zcu_tools.program.v2.modular import ModularProgramV2

from .base import Module

logger = logging.getLogger(__name__)

SubModule: TypeAlias = Union[Module, list[Module]]


class Repeat(Module):
    """
    Repeat sub-modules n times.

    If n is an int, uses QICK's open_loop/close_loop (compile-time constant).
    If n is a str, treats it as a register name: emits a register-driven loop
    using cond_jump / jump / label, so the loop count can vary at runtime.
    """

    def __init__(self, name: str, n: Union[int, str]) -> None:
        self.name = name
        self.n = n
        self.sub_modules = []
        self.counter_reg = self.name

        if isinstance(n, int) and n < 0:
            raise ValueError(f"Repeat n must be greater than or equal to 0, got {n}")

    def add_content(self, mod: SubModule) -> Self:
        if isinstance(mod, Module):
            mod = [mod]
        self.sub_modules.extend(mod)
        return self

    def init(self, prog: ModularProgramV2) -> None:
        prog.add_reg(self.counter_reg)
        for mod in self.sub_modules:
            mod.init(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        logger.debug(
            "Repeat.run: name='%s', counter_reg='%s', n='%s', t=%s",
            self.name,
            self.counter_reg,
            self.n,
            t,
        )

        prog.delay(t=t)
        prog.delay_auto(t=0.0)

        prog.open_inner_loop(self.name, self.counter_reg, self.n)

        cur_t = 0.0
        for mod in self.sub_modules:
            cur_t = mod.run(prog, cur_t)
        prog.delay(t=cur_t)
        prog.delay_auto(t=0.0)

        prog.close_inner_loop(self.name, self.counter_reg)

        return 0.0  # prog.delay will modify ref time


class Branch(Module):
    """Select and execute one branch per outer-loop iteration based on the loop counter.

    Branch does NOT create its own loop. It relies on an external sweep loop
    (created via add_loop) whose counter register matches ``name``. On each
    iteration of that outer loop the counter selects the corresponding branch,
    so branch *i* runs exactly once (at iteration *i*).

    Branch selection uses a binary cond_jump dispatch tree. For non-power-of-two
    branch counts, shorter paths are padded with NOP so each branch starts after
    the same number of control instructions. Each branch is wrapped with delay /
    delay_auto to flush timing, so branches may have different durations. The
    module always returns 0.0.
    """

    def __init__(
        self, name: str, *branches: SubModule, compare_by: Optional[str] = None
    ) -> None:
        self.name = name
        self.compare_reg = compare_by if compare_by is not None else name

        if len(branches) < 2:
            raise ValueError("Branch requires at least 2 branches")

        self.branches: list[list[Module]] = [
            [b] if isinstance(b, Module) else list(b) for b in branches
        ]

    def init(self, prog: ModularProgramV2) -> None:
        for branch in self.branches:
            for mod in branch:
                mod.init(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        logger.debug(
            "Branch.run: name='%s', compare_reg='%s', n_branches=%d, t=%s",
            self.name,
            self.compare_reg,
            len(self.branches),
            t,
        )

        prog.delay(t=t)
        prog.delay_auto(t=0)

        n = len(self.branches)
        max_depth = (n - 1).bit_length()
        
        prog._add_asm({"CMD": "__META__", "TYPE": "BRANCH_START", "NAME": self.name}, 0)

        def run_branch(i: int, depth: int) -> None:
            prog._add_asm({"CMD": "__META__", "TYPE": "BRANCH_CASE_START", "NAME": str(i)}, 0)
            for _ in range(max_depth - depth):
                prog.nop()

            with prog.disable_delay():
                cur_t: Union[float, QickParam] = 0.0
                for mod in self.branches[i]:
                    if logger.isEnabledFor(logging.DEBUG):
                        prog.debug_macro(
                            f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                        )
                    cur_t = mod.run(prog, cur_t)

            # TODO: support branch with swept duration
            if isinstance(cur_t, QickParam):
                raise NotImplementedError("Branch with swept duration is not supported")

            prog.delay(t=cur_t)
            prog._add_asm({"CMD": "__META__", "TYPE": "BRANCH_CASE_END", "NAME": str(i)}, 0)

        def emit_dispatch(lo: int, hi: int, depth: int) -> None:
            if hi - lo == 1:
                run_branch(lo, depth)
                return

            mid = (lo + hi) // 2
            left_label = f"{self.name}_branch_l_{lo}_{mid}"
            end_label = f"{self.name}_branch_e_{lo}_{hi}"

            # compare_reg < mid -> left half, else right half
            prog.cond_jump(left_label, self.compare_reg, "S", "-", mid)
            emit_dispatch(mid, hi, depth + 1)
            prog.jump(end_label)
            prog.label(left_label)
            emit_dispatch(lo, mid, depth + 1)
            prog.label(end_label)

        emit_dispatch(0, n, 0)
        
        prog._add_asm({"CMD": "__META__", "TYPE": "BRANCH_END", "NAME": self.name}, 0)

        prog.delay_auto(t=0)

        return 0.0
