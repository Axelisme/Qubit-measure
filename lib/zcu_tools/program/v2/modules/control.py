from __future__ import annotations

import logging

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Optional, Self, TypeAlias, Union

from zcu_tools.program.v2.modular import ModularProgramV2

from .base import Module

if TYPE_CHECKING:
    from zcu_tools.program.v2.ir.builder import IRBuilder

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

        if isinstance(n, int) and n < 0:
            raise ValueError(f"Repeat n must be greater than or equal to 0, got {n}")

    def add_content(self, mod: SubModule) -> Self:
        if isinstance(mod, Module):
            mod = [mod]
        self.sub_modules.extend(mod)
        return self

    def init(self, prog: ModularProgramV2) -> None:
        for mod in self.sub_modules:
            mod.init(prog)

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        builder.ir_delay(t)
        builder.ir_delay_auto(t=0.0)

        if isinstance(self.n, int) and self.n <= 0:
            return 0.0

        if isinstance(self.n, str):
            loop_ctx = builder.ir_reg_loop(name=self.name, n_reg=self.n)
        else:
            loop_ctx = builder.ir_loop(name=self.name, n=self.n)

        with loop_ctx:
            cur_t: Union[float, QickParam] = 0.0
            for mod in self.sub_modules:
                cur_t = mod.ir_run(builder, cur_t, prog)

            if not cur_t > 0.09:
                logger.warning(
                    "Repeat '%s' has short body duration %s; loop overhead may dominate timing precision. Consider using SoftRepeat for better timing accuracy.",
                    self.name,
                    cur_t,
                )

            builder.ir_delay(cur_t)

        return 0.0


class SoftRepeat(Module):
    """Repeat a module or a list of modules n times"""

    def __init__(self, name: str, n: int) -> None:
        self.name = name
        self.n = n
        self.sub_modules = []

        if n < 0:
            raise ValueError(f"Repeat n must be greater than or equal to 0, got {n}")

    def add_content(self, mod: SubModule) -> Self:
        if isinstance(mod, Module):
            mod = [mod]
        self.sub_modules.extend(mod)
        return self

    def init(self, prog: ModularProgramV2) -> None:
        for mod in self.sub_modules:
            mod.init(prog)

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        for _ in range(self.n):
            for mod in self.sub_modules:
                t = mod.ir_run(builder, t, prog)
        return t


class Branch(Module):
    """Select and execute one branch per outer-loop iteration based on the loop counter.

    Branch does NOT create its own loop. It relies on an external sweep loop
    (created via add_loop) whose counter register matches ``name``. On each
    iteration of that outer loop the counter selects the corresponding branch,
    so branch *i* runs exactly once (at iteration *i*).

    Branch selection uses a binary cond_jump dispatch tree. Branch-path balancing
    (NOP padding) is handled later by the IR pass pipeline
    (`AlignBranchDispatch`). Each branch is wrapped with delay / delay_auto to
    flush timing, so branches may have different durations. The module always
    returns 0.0.
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

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        builder.ir_delay(t)
        builder.ir_delay_auto(t=0.0)

        with builder.ir_branch(self.compare_reg) as branch:
            for mods in self.branches:
                with branch.arm():
                    with prog.disable_delay():
                        cur_t: Union[float, QickParam] = 0.0
                        for mod in mods:
                            cur_t = mod.ir_run(builder, cur_t, prog)

                    if isinstance(cur_t, QickParam):
                        raise NotImplementedError(
                            "Branch with swept duration is not supported"
                        )

                    builder.ir_delay(cur_t)

        builder.ir_delay_auto(t=0.0)
        return 0.0
