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

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        logger.debug("Repeat.run: name='%s', n='%s', t=%s", self.name, self.n, t)

        prog.delay(t=t)
        prog.delay_auto(t=0)

        if isinstance(self.n, int) and self.n <= 0:
            return 0.0

        if isinstance(self.n, str):
            prog.open_loop_reg(self.n, self.name)
        else:
            prog.open_loop(self.n, self.name)

        cur_t = 0.0
        for mod in self.sub_modules:
            if logger.isEnabledFor(logging.DEBUG):
                prog.debug_macro(
                    f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                )
            cur_t = mod.run(prog, cur_t)

        if not cur_t > 0.09:
            logger.warning(
                "Repeat '%s' has long body duration %s, which may cause imprecise timing due to loop overhead. Consider using SoftRepeat for better timing accuracy.",
                self.name,
                cur_t,
            )

        prog.delay(t=cur_t)

        if isinstance(self.n, str):
            prog.close_loop_reg(self.name)
        else:
            prog.close_loop()

        return 0.0  # prog.delay will modify ref time

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
                    "Repeat '%s' has long body duration %s, which may cause imprecise timing due to loop overhead. Consider using SoftRepeat for better timing accuracy.",
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

        for m in mod:
            if not m.allow_rerun():
                raise ValueError(
                    f"Module {m.name} does not allow rerun, cannot be used in SoftRepeat"
                )

        self.sub_modules.extend(mod)
        return self

    def allow_rerun(self) -> bool:
        return all(mod.allow_rerun() for mod in self.sub_modules)

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

        def run_branch(i: int, depth: int) -> None:
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

        prog.delay_auto(t=0)

        return 0.0

    def _branch_depths(self) -> list[int]:
        n = len(self.branches)
        depths = [0] * n

        def fill_depth(lo: int, hi: int, depth: int) -> None:
            if hi - lo == 1:
                depths[lo] = depth
                return
            mid = (lo + hi) // 2
            fill_depth(mid, hi, depth + 1)
            fill_depth(lo, mid, depth + 1)

        fill_depth(0, n, 0)
        return depths

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        builder.ir_delay(t)
        builder.ir_delay_auto(t=0.0)

        n = len(self.branches)
        max_depth = (n - 1).bit_length()
        branch_depths = self._branch_depths()

        with builder.ir_branch(self.compare_reg) as branch:
            for i, mods in enumerate(self.branches):
                with branch.arm():
                    for _ in range(max_depth - branch_depths[i]):
                        builder.ir_nop()

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
