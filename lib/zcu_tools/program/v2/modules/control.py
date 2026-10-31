from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Self, override

from qick.asm_v2 import AsmInst, QickParam, WriteLabel

from zcu_tools.program.v2.modular import ModularProgramV2

from ..ir.hw_semantics import ADDR_REG, needs_big_jump
from ..ir.operands import parse_register
from .base import Module, ModuleGroup, as_modules

if TYPE_CHECKING:
    from zcu_tools.program.v2.timeline import TimelineProgramV2

logger = logging.getLogger(__name__)


def _emit_jump_s15(prog) -> None:
    prog.append_macro(AsmInst(inst={"CMD": "JUMP", "ADDR": "s15"}, addr_inc=1))


def _emit_write_label(prog, label: str) -> None:
    prog.append_macro(WriteLabel(label=label))


def _reject_addr_reg_selector(reg_name: str, *, field: str) -> None:
    reg = parse_register(reg_name)
    if reg is not None and reg.canonical_name == ADDR_REG:
        raise ValueError(f"{field} must not use reserved address register {ADDR_REG}")


class Repeat(Module):
    """
    Repeat sub-modules n times.

    If n is an int, uses QICK's open_loop/close_loop (compile-time constant).
    If n is a str, treats it as a register name: emits a register-driven loop
    using cond_jump / jump / label, so the loop count can vary at runtime.
    """

    def __init__(
        self,
        name: str,
        n: int | str,
        *,
        range_hint: tuple[int, int] | None = None,
    ) -> None:
        self.name = name
        self.n = n
        self.sub_modules = []
        self.counter_reg = self.name
        self.range_hint = (
            (range_hint[0], range_hint[1]) if range_hint is not None else None
        )

        if isinstance(n, int) and n < 0:
            raise ValueError(f"Repeat n must be greater than or equal to 0, got {n}")

        # counter_reg is named after `name`; a register-driven n sharing that
        # name would make the back-edge `op(counter - n)` evaluate to a
        # constant 0, breaking the loop condition.
        if isinstance(n, str) and n == self.counter_reg:
            raise ValueError(
                f"Repeat n register {n!r} must differ from the counter register "
                f"(named after Repeat name {name!r})"
            )

    def add_content(self, mod: ModuleGroup) -> Self:
        """Append a module or ordered group; skip None entries (no module)."""
        self.sub_modules.extend(as_modules(mod))
        return self

    def init(self, prog: ModularProgramV2) -> None:
        if isinstance(self.n, int) and self.n == 0:
            logger.debug("Repeat.init: skip zero-iteration loop name='%s'", self.name)
            return

        prog.add_reg(self.counter_reg)
        for mod in self.sub_modules:
            mod.init(prog)

    def run(
        self,
        prog: ModularProgramV2,
        t: float | QickParam = 0.0,
        *,
        internal_modules: tuple[Module, ...] = (),
    ) -> float | QickParam:
        """Run the loop at t us and return its reference-relative blocking end.

        prog must have initialized this Repeat. internal_modules identifies, by
        object identity, children owned by the caller's semantic Module. These
        helpers run directly without a source scope, retaining the current owner.
        Other children emit with their own scope and restore that owner afterward.
        The default empty tuple keeps every child as an authored emission.
        Repeated occurrences of one object share its classification. A specified
        object outside sub_modules raises ValueError before any program emission.
        Loop count, scheduling and iteration metadata do not depend on ownership.
        """
        if any(
            not any(helper is child for child in self.sub_modules)
            for helper in internal_modules
        ):
            raise ValueError("internal_modules must identify this Repeat's children")
        logger.debug(
            "Repeat.run: name='%s', counter_reg='%s', n='%s', t=%s",
            self.name,
            self.counter_reg,
            self.n,
            t,
        )

        prog.delay(t=t)
        prog.delay_auto(t=0.0)

        if isinstance(self.n, int) and self.n == 0:
            logger.debug(
                "Repeat.run: short-circuit zero-iteration loop name='%s'", self.name
            )
            return 0.0

        prog.open_inner_loop(
            self.name, self.counter_reg, self.n, range_hint=self.range_hint
        )

        cur_t = 0.0
        for mod in self.sub_modules:
            if any(mod is helper for helper in internal_modules):
                cur_t = mod.run(prog, cur_t)
            else:
                cur_t = mod.emit(prog, cur_t)
        prog.delay(t=cur_t)
        prog.delay_auto(t=0.0)

        prog.close_inner_loop(self.name, self.counter_reg, self.n)

        return 0.0  # prog.delay will modify ref time


class Branch(Module):
    """Select and execute one branch per outer-loop iteration based on the loop counter.

    Branch does NOT create its own loop. It relies on an external sweep loop
    (created via add_loop) whose counter register matches ``name``. On each
    iteration of that outer loop the counter selects the corresponding branch,
    so branch *i* runs exactly once (at iteration *i*).

    Branches accept modules or sequences; None entries mean no module and are
    skipped. A None branch remains an empty selectable case.

    On TimelineProgramV2, declare registers the children and lower uses native
    branch/case control with compare_reg as the sweep selector. Children execute
    in order on each implicit case timeline. align_timeline=True joins case ends
    into the emission parent, using static max or exact runtime pointwise max. Reference does
    not advance and no wait is inserted. Later emissions resume at the joined
    end. Reference-moving children fail at build, including explicit delay_by_timeline or
    native delay inside Repeat. Accumulating Repeat scopes are allowed.
    Measurement feedback is not supported.

    Legacy init/run selection uses a dispatch-table island:

    1. compute ``s15 = &table_0 + compare_reg * entry_words``
    2. indirect jump into the table
    3. table stub performs one more jump to the real case entry

    This keeps dispatch depth constant while allowing each case body to be
    independently optimised. Each branch is wrapped with delay / delay_auto to
    flush timing, so branches may have different durations. The module always
    returns 0.0.
    """

    def __init__(
        self, name: str, *branches: ModuleGroup, compare_by: str | None = None
    ) -> None:
        self.name = name
        self.compare_reg = compare_by if compare_by is not None else name
        _reject_addr_reg_selector(self.compare_reg, field="Branch.compare_by")

        if len(branches) < 2:
            raise ValueError("Branch requires at least 2 branches")

        self.branches: list[list[Module]] = [as_modules(branch) for branch in branches]

    @override
    def declare(self, prog: TimelineProgramV2) -> None:
        """Declare every child in case order; shared declarations are reused.

        prog must be in _initialize. Child declaration failures propagate.
        """
        for branch in self.branches:
            for mod in branch:
                prog.declare(mod)

    @override
    def lower(
        self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
    ) -> None:
        """Emit the sweep-selected case on timeline (default when None).

        Preserve empty cases and child order. Use align_timeline=True to join
        all implicit case ends into the parent, without advancing reference. Native
        reference-moving children, including operations inside Repeat, raise ValueError.
        tag identifies this emission; children remain separate emissions.
        Invalid selectors, timelines and child lowering failures propagate.
        """
        with prog.branch(
            by=self.compare_reg, align_timeline=True, timeline=timeline
        ) as cases:
            for index, branch in enumerate(self.branches):
                with cases.case(index):
                    for mod in branch:
                        prog.emit(mod)

    def init(self, prog: ModularProgramV2) -> None:
        for branch in self.branches:
            for mod in branch:
                mod.init(prog)

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        logger.debug(
            "Branch.run: name='%s', compare_reg='%s', n_branches=%d, t=%s",
            self.name,
            self.compare_reg,
            len(self.branches),
            t,
        )

        def run_branch(i: int) -> None:
            prog.meta_macro(type="BRANCH_CASE_START", name=str(i))
            prog.label(f"{self.name}_case_entry_{i}")

            cur_t: float | QickParam = 0.0
            for mod in self.branches[i]:
                if logger.isEnabledFor(logging.DEBUG):
                    prog.debug_macro(
                        f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                    )
                cur_t = mod.emit(prog, cur_t)

            # Swept-duration branches cannot be balanced with one scalar tail delay.
            if isinstance(cur_t, QickParam):
                raise NotImplementedError("Branch with swept duration is not supported")

            prog.delay(t=cur_t)
            prog.delay_auto()

            prog.meta_macro(type="BRANCH_CASE_END", name=str(i))

        prog.delay(t=t)
        prog.delay_auto(t=0)

        n = len(self.branches)

        prog.meta_macro(
            type="BRANCH_START",
            name=self.name,
            regs={"compare_reg": self.compare_reg},
        )

        big_jump = needs_big_jump(prog.tproccfg["pmem_size"])

        table_base = f"{self.name}_dispatch_0"
        _emit_write_label(prog, table_base)
        prog.write_reg_op("s15", "s15", "+", self.compare_reg)
        if big_jump:
            prog.write_reg_op("s15", "s15", "+", self.compare_reg)
        _emit_jump_s15(prog)

        for i in range(n):
            prog.label(f"{self.name}_dispatch_{i}")
            if big_jump:
                _emit_write_label(prog, f"{self.name}_case_entry_{i}")
                _emit_jump_s15(prog)
            else:
                prog.jump(f"{self.name}_case_entry_{i}")

        end_label = f"{self.name}_end"

        for i in range(n):
            run_branch(i)
            if i < n - 1:
                if big_jump:
                    _emit_write_label(prog, end_label)
                    _emit_jump_s15(prog)
                else:
                    prog.jump(end_label)
        prog.meta_macro(type="BRANCH_END", name=self.name)
        prog.label(end_label)

        return 0.0
