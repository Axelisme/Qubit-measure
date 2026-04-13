from __future__ import annotations

import logging

import qick.asm_v2 as qasm
from qick.asm_v2 import QickParam
from typing_extensions import TypeAlias, Union

from ..base import MyProgramV2
from ..utils import PrintTimeStamp
from .base import Module

logger = logging.getLogger(__name__)

SubModule: TypeAlias = Union[Module, list[Module]]


class Repeat(Module):
    """
    Repeat a module or a list of modules n times
    It will call delay before and inside each loop, this may cause unexpected behavior
    """

    def __init__(self, name: str, n: int, sub_module: SubModule) -> None:
        self.name = name
        self.n = n

        if isinstance(sub_module, Module):
            sub_module = [sub_module]
        self.sub_module = sub_module

        if n < 0:
            raise ValueError(f"Repeat n must be greater than or equal to 0, got {n}")

    def init(self, prog: MyProgramV2) -> None:
        for mod in self.sub_module:
            mod.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        if self.n == 0:
            return t

        # this n must > 0, to prevent infinite loop in qick
        assert self.n > 0

        logger.debug("Repeat.run: name='%s', n=%d, t=%s", self.name, self.n, t)

        prog.delay(t=t)
        prog.append_macro(qasm.OpenLoop(name=self.name, n=self.n))
        cur_t = 0.0
        for mod in self.sub_module:
            if logger.isEnabledFor(logging.DEBUG):
                prog.append_macro(
                    PrintTimeStamp(
                        f"{mod.__class__.__name__}({mod.name})", cur_t, prefix="\t"
                    )
                )
            cur_t = mod.run(prog, cur_t)
        prog.delay(t=cur_t)
        prog.append_macro(qasm.CloseLoop())

        return 0.0  # prog.delay will modify ref time


class SoftRepeat(Module):
    """Repeat a module or a list of modules n times"""

    def __init__(self, name: str, n: int, sub_module: SubModule) -> None:
        self.name = name
        self.n = n

        if isinstance(sub_module, Module):
            sub_module = [sub_module]
        self.sub_module = sub_module

        if n < 0:
            raise ValueError(f"Repeat n must be greater than or equal to 0, got {n}")

    def init(self, prog: MyProgramV2) -> None:
        for mod in self.sub_module:
            mod.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        for _ in range(self.n):
            for mod in self.sub_module:
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

    def __init__(self, name: str, *branches: SubModule) -> None:
        self.name = name

        if len(branches) < 2:
            raise ValueError("Branch requires at least 2 branches")

        self.branches: list[list[Module]] = [
            [b] if isinstance(b, Module) else list(b) for b in branches
        ]

    def init(self, prog: MyProgramV2) -> None:
        for branch in self.branches:
            for mod in branch:
                mod.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        logger.debug(
            "Branch.run: name='%s', n_branches=%d, t=%s",
            self.name, len(self.branches), t,
        )

        prog.delay(t=t)
        prog.delay_auto(t=0)

        end_label = f"_branch_{self.name}_end"
        n = len(self.branches)

        for i, branch in enumerate(self.branches):
            is_last = i == n - 1

            if not is_last:
                skip_label = f"_branch_{self.name}_skip_{i}"
                prog.append_macro(
                    qasm.CondJump(
                        arg1=self.name, arg2=i, op="-", test="NZ", label=skip_label
                    )
                )

            cur_t: Union[float, QickParam] = 0.0
            for mod in branch:
                if logger.isEnabledFor(logging.DEBUG):
                    prog.append_macro(
                        PrintTimeStamp(
                            f"{mod.__class__.__name__}({mod.name})",
                            cur_t,
                            prefix="\t",
                        )
                    )
                cur_t = mod.run(prog, cur_t)

            # TODO: support branch with swept duration
            if isinstance(cur_t, QickParam):
                raise NotImplementedError("Branch with swept duration is not supported")

            prog.delay(t=cur_t)

            if not is_last:
                prog.append_macro(qasm.Jump(label=end_label))
                prog.append_macro(qasm.Label(label=skip_label))

        prog.append_macro(qasm.Label(label=end_label))

        prog.delay_auto(t=0)

        return 0.0
