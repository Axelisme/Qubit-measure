from __future__ import annotations

import logging

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Optional, Self, TypeAlias, Union

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2


from .base import Module

logger = logging.getLogger(__name__)

SubModule: TypeAlias = Union[Module, list[Module]]


class Repeat(Module):
    """
    Repeat a module or a list of modules n times
    It will call delay before and inside each loop, this may cause unexpected behavior
    """

    def __init__(self, name: str, n: int) -> None:
        self.name = name
        self.n = n
        self.sub_modules = []
        self.idx_reg = f"{name}_idx"

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

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        if self.n == 0:
            return t

        # this n must > 0, to prevent infinite loop in qick
        assert self.n > 0

        logger.debug("Repeat.run: name='%s', n=%d, t=%s", self.name, self.n, t)

        prog.delay(t=t)
        prog.open_loop(name=self.idx_reg, n=self.n)

        cur_t = 0.0
        for mod in self.sub_modules:
            if logger.isEnabledFor(logging.DEBUG):
                prog.debug_macro(
                    f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                )
            cur_t = mod.run(prog, cur_t)
        prog.delay(t=cur_t)
        prog.delay_auto(t=0)

        prog.close_loop()

        return 0.0  # prog.delay will modify ref time


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

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        for _ in range(self.n):
            for mod in self.sub_modules:
                t = mod.run(prog, t)
        return t


class RepeatByRegister(Module):
    """Repeat a module or a list of modules n times, where n is the value of a register"""

    def __init__(self, name: str, n_reg: str) -> None:
        self.name = name
        self.n_reg = n_reg
        self.sub_modules = []
        self.counter_reg = f"{name}_counter"

    def add_content(self, mod: SubModule) -> Self:
        if isinstance(mod, Module):
            mod = [mod]
        self.sub_modules.extend(mod)
        return self

    def init(self, prog: ModularProgramV2) -> None:
        for mod in self.sub_modules:
            mod.init(prog)

        prog.add_reg(self.counter_reg)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        logger.debug(
            "RepeatByRegister.run: name='%s', n_reg='%s', t=%s", self.name, self.n_reg, t
        )
        prog.delay(t=t)
        prog.delay_auto(t=0)

        start_label = f"{self.name}_start"
        end_label = f"{self.name}_end"

        # Guard against negative loop bounds to avoid non-terminating loops.
        prog.cond_jump(end_label, self.n_reg, "S")
        prog.write_reg(self.counter_reg, 0)

        prog.label(start_label)
        prog.cond_jump(end_label, self.n_reg, "Z", "-", self.counter_reg)

        cur_t = 0.0
        for mod in self.sub_modules:
            if logger.isEnabledFor(logging.DEBUG):
                prog.debug_macro(
                    f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                )
            cur_t = mod.run(prog, cur_t)
        prog.delay(t=cur_t)
        prog.delay_auto(t=0)

        prog.inc_reg(self.counter_reg, 1)
        prog.jump(start_label)

        prog.label(end_label)

        return 0.0  # prog.delay will modify ref time


class Branch(Module):
    """Select and execute one branch per outer-loop iteration based on the loop counter.

    Branch does NOT create its own loop. It relies on an external sweep loop
    (created via add_loop) whose counter register matches ``name``. On each
    iteration of that outer loop the counter selects the corresponding branch,
    so branch *i* runs exactly once (at iteration *i*).

    Each branch is wrapped with delay / delay_auto to flush timing, so branches
    may have different durations. The module always returns 0.0.
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

        end_label = f"{self.name}_branch_end"
        n = len(self.branches)

        for i, branch in enumerate(self.branches):
            is_last = i == n - 1

            if not is_last:
                skip_label = f"{self.name}_branch_skip_{i}"
                prog.cond_jump(skip_label, self.compare_reg, "NZ", "-", i)

            cur_t: Union[float, QickParam] = 0.0
            for mod in branch:
                if logger.isEnabledFor(logging.DEBUG):
                    prog.debug_macro(
                        f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                    )
                cur_t = mod.run(prog, cur_t)

            # TODO: support branch with swept duration
            if isinstance(cur_t, QickParam):
                raise NotImplementedError("Branch with swept duration is not supported")

            prog.delay(t=cur_t)

            if not is_last:
                prog.jump(end_label)
                prog.label(skip_label)

        prog.label(end_label)

        prog.delay_auto(t=0)

        return 0.0
