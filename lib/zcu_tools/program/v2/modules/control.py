from __future__ import annotations

import logging

from qick.asm_v2 import AsmInst, Label, Macro, QickParam, WriteLabel
from typing_extensions import TYPE_CHECKING, Optional, Self, TypeAlias, Union

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2


from .base import Module

logger = logging.getLogger(__name__)

SubModule: TypeAlias = Union[Module, list[Module]]


class OpenLoopByRegMacro(Macro):
    """Register-based equivalent of QICK's OpenLoop.

    Generates: guard against non-positive n, initialize counter to 0,
    loop-start label, and exit-condition check.
    The loop body (sub-module macros) is appended to macro_list by Repeat.run()
    between this macro and CloseLoopByRegMacro.
    """

    def __init__(self, name: str, n_reg: str, counter_reg: str) -> None:
        self.name = name
        self.n_reg = n_reg
        self.counter_reg = counter_reg

    def expand(self, prog) -> list:  # type: ignore
        start_label = f"{self.name}_start"
        end_label = f"{self.name}_end"
        large_pmem = prog.tproccfg["pmem_size"] > 2**11

        n_reg = prog._get_reg(self.n_reg)
        counter_reg = prog._get_reg(self.counter_reg)

        insts = []

        # Guard: skip loop if n_reg is signed/negative
        insts.append(AsmInst(inst={"CMD": "TEST", "OP": n_reg, "UF": "1"}, addr_inc=1))
        if large_pmem:
            insts.append(WriteLabel(label=end_label))
            insts.append(
                AsmInst(inst={"CMD": "JUMP", "IF": "S", "ADDR": "s15"}, addr_inc=1)
            )
        else:
            insts.append(
                AsmInst(inst={"CMD": "JUMP", "IF": "S", "LABEL": end_label}, addr_inc=1)
            )

        # Initialize counter to 0
        insts.append(
            AsmInst(
                inst={"CMD": "REG_WR", "DST": counter_reg, "SRC": "imm", "LIT": "#0"},
                addr_inc=1,
            )
        )

        # Loop start label
        insts.append(Label(label=start_label))

        # Exit when counter reaches n_reg (n_reg - counter == 0)
        insts.append(
            AsmInst(
                inst={"CMD": "TEST", "OP": f"{n_reg} - {counter_reg}", "UF": "1"},
                addr_inc=1,
            )
        )
        if large_pmem:
            insts.append(WriteLabel(label=end_label))
            insts.append(
                AsmInst(inst={"CMD": "JUMP", "IF": "Z", "ADDR": "s15"}, addr_inc=1)
            )
        else:
            insts.append(
                AsmInst(inst={"CMD": "JUMP", "IF": "Z", "LABEL": end_label}, addr_inc=1)
            )

        return insts


class CloseLoopByRegMacro(Macro):
    """Register-based equivalent of QICK's CloseLoop.

    Generates: increment counter, unconditional jump back to loop start, end label.
    """

    def __init__(self, name: str, counter_reg: str) -> None:
        self.name = name
        self.counter_reg = counter_reg

    def expand(self, prog) -> list:  # type: ignore
        start_label = f"{self.name}_start"
        end_label = f"{self.name}_end"
        large_pmem = prog.tproccfg["pmem_size"] > 2**11

        counter_reg = prog._get_reg(self.counter_reg)

        insts = []

        # Increment counter
        insts.append(
            AsmInst(
                inst={
                    "CMD": "REG_WR",
                    "DST": counter_reg,
                    "SRC": "op",
                    "OP": f"{counter_reg} + #1",
                },
                addr_inc=1,
            )
        )

        # Jump back to loop start
        if large_pmem:
            insts.append(WriteLabel(label=start_label))
            insts.append(AsmInst(inst={"CMD": "JUMP", "ADDR": "s15"}, addr_inc=1))
        else:
            insts.append(
                AsmInst(inst={"CMD": "JUMP", "LABEL": start_label}, addr_inc=1)
            )

        # End label
        insts.append(Label(label=end_label))

        return insts


class Repeat(Module):
    """
    Repeat sub-modules n times.

    If n is an int, uses QICK's open_loop/close_loop (compile-time constant).
    If n is a str, treats it as a register name: bookends sub-module runs with
    OpenLoopByRegMacro / CloseLoopByRegMacro for a runtime loop count.
    """

    def __init__(self, name: str, n: Union[int, str]) -> None:
        self.name = name
        self.n = n
        self.sub_modules = []

        self.idx_reg = ""
        if isinstance(n, int):
            if n < 0:
                raise ValueError(
                    f"Repeat n must be greater than or equal to 0, got {n}"
                )
            self.idx_reg = f"{name}_idx"

    def add_content(self, mod: SubModule) -> Self:
        if isinstance(mod, Module):
            mod = [mod]
        self.sub_modules.extend(mod)
        return self

    def init(self, prog: ModularProgramV2) -> None:
        for mod in self.sub_modules:
            mod.init(prog)
        if isinstance(self.n, str):
            self.counter_reg = f"{self.name}_counter"
            prog.add_reg(self.counter_reg)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        if isinstance(self.n, str):
            logger.debug(
                "Repeat.run (by register): name='%s', n_reg='%s', t=%s",
                self.name,
                self.n,
                t,
            )
            prog.delay(t=t)
            prog.delay_auto(t=0)
            prog.append_macro(OpenLoopByRegMacro(self.name, self.n, self.counter_reg))

            cur_t = 0.0
            for mod in self.sub_modules:
                if logger.isEnabledFor(logging.DEBUG):
                    prog.debug_macro(
                        f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                    )
                cur_t = mod.run(prog, cur_t)
            prog.delay(t=cur_t)

            prog.append_macro(CloseLoopByRegMacro(self.name, self.counter_reg))
            prog.delay_auto(t=0)

            return 0.0

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
