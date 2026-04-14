from __future__ import annotations

import logging

import numpy as np
import qick.asm_v2 as qasm
from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Sequence, TypeAlias, Union

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2

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

    def init(self, prog: ModularProgramV2) -> None:
        for mod in self.sub_module:
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
        prog.open_loop(name=self.name, n=self.n)
        cur_t = 0.0
        for mod in self.sub_module:
            if logger.isEnabledFor(logging.DEBUG):
                prog.debug_macro(
                    f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                )
            cur_t = mod.run(prog, cur_t)
        prog.delay(t=cur_t)
        prog.close_loop()

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

    def init(self, prog: ModularProgramV2) -> None:
        for mod in self.sub_module:
            mod.init(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        for _ in range(self.n):
            for mod in self.sub_module:
                t = mod.run(prog, t)
        return t


class ScanWith(Module):
    """Run ``sub_module`` once per entry in ``values``, loading each into register ``name``.

    An internal loop counter register holds the iteration index; register ``name`` is
    filled from data memory so it can drive :class:`Branch` and similar modules.

    Values are staged into program dmem init data during :meth:`init`.
    """

    def __init__(self, name: str, values: Sequence[int], sub_module: SubModule) -> None:
        self.name = name
        self.values = list(values)

        if len(values) == 0:
            raise ValueError("ScanWith requires a non-empty values sequence")

        if isinstance(sub_module, Module):
            sub_module = [sub_module]
        self.sub_module = sub_module

        self.addr_reg = f"{name}_addr"
        self.offset = 0

    def init(self, prog: ModularProgramV2) -> None:
        self.offset = prog.set_dmem(self.values)

        prog.add_reg(self.name)
        prog.add_reg(self.addr_reg)

        for mod in self.sub_module:
            mod.init(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        n = len(self.values)
        logger.debug(
            f"ScanWith.run: name='{self.name}', n={n}, offset={self.offset}, t={t}"
        )

        prog.delay(t=t)
        prog.write_reg(self.addr_reg, self.offset)

        prog.open_loop(name=f"{self.name}_loop", n=n)
        prog.read_dmem(dst=self.name, addr=self.addr_reg)

        cur_t: Union[float, QickParam] = 0.0
        for mod in self.sub_module:
            if logger.isEnabledFor(logging.DEBUG):
                prog.debug_macro(
                    f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                )
            cur_t = mod.run(prog, cur_t)
        prog.delay(t=cur_t)

        prog.inc_reg(self.addr_reg, 1)
        prog.close_loop()

        return 0.0


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

    def init(self, prog: ModularProgramV2) -> None:
        for branch in self.branches:
            for mod in branch:
                mod.init(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        logger.debug(
            f"Branch.run: name='{self.name}', n_branches={len(self.branches)}, t={t}"
        )

        prog.delay(t=t)
        prog.delay_auto(t=0)

        end_label = f"_branch_{self.name}_end"
        n = len(self.branches)

        for i, branch in enumerate(self.branches):
            is_last = i == n - 1

            if not is_last:
                skip_label = f"_branch_{self.name}_skip_{i}"
                prog.cond_jump(skip_label, self.name, "NZ", "-", i)

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
