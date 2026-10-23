from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, override

from qick.asm_v2 import QickParam

from ..utils import is_qick_param
from .base import Module
from .util import merge_max_length, round_timestamp

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2

    from .pulse import TableLengthPulse

logger = logging.getLogger(__name__)


class Delay(Module):
    def __init__(
        self, name: str, delay: float | QickParam, tag: str | None = None
    ) -> None:
        self.name = name
        self.delay = delay
        self.tag = tag

    def init(self, prog: ModularProgramV2) -> None: ...

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        prog.delay(t=round_timestamp(prog, t + self.delay), tag=self.tag)

        return 0.0  # reset reference time

    def allow_rerun(self) -> bool:
        return self.tag is None


class SoftDelay(Module):
    def __init__(self, name: str, delay: float | QickParam) -> None:
        self.name = name
        self.delay = delay

    def init(self, prog: ModularProgramV2) -> None:
        pass

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        return round_timestamp(prog, t + self.delay)

    def allow_rerun(self) -> bool:
        return True


class DelayAuto(Module):
    """Wait for selected channel timelines and then advance the reference time.

    name identifies the module. t is extra time in microseconds (including QickParam
    sweeps), or a register name holding tProc cycles. gens and ros select generator
    and readout timelines. At least one must be True; otherwise construction raises
    ValueError. tag names a time parameter and cannot be used with a register name.
    """

    def __init__(
        self,
        name: str,
        t: float | QickParam | str = 0.0,
        tag: str | None = None,
        *,
        gens: bool = True,
        ros: bool = True,
    ) -> None:
        if not gens and not ros:
            raise ValueError("DelayAuto requires at least one channel: gens or ros")

        self.name = name
        self.t = t
        self.gens = gens
        self.ros = ros
        self.tag = tag

        if tag is not None and isinstance(t, str):
            raise ValueError("DelayAuto with tag cannot have t as a register name")

    def init(self, prog: ModularProgramV2) -> None: ...

    @override
    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        if isinstance(self.t, str):
            prog.delay_reg_auto(time_reg=self.t, gens=self.gens, ros=self.ros)
        else:
            prog.delay_auto(t=self.t, gens=self.gens, ros=self.ros, tag=self.tag)
        return 0.0

    def allow_rerun(self) -> bool:
        return self.tag is None


class Join(Module):
    """Start independent branches together and wait for their longest duration.

    args are modules or module sequences. Delay/DelayAuto branches are invalid.
    Table-length joins require exactly one TableLengthPulse per branch, distinct
    generators and a shared index register/table size. Other table combinations
    raise ValueError. table_pulses holds those branches, or an empty tuple for
    ordinary joins. Their common wait uses each point's quantized maximum, not
    a sweep-wide maximum. Input modules become owned by this Join.
    """

    def __init__(self, *args: Module | Sequence[Module]) -> None:
        self.name = ""

        if len(args) == 0:
            raise ValueError("Join must contain at least one module")

        join_modules = [[m] if isinstance(m, Module) else list(m) for m in args]
        for mod_list in join_modules:
            for m in mod_list:
                if isinstance(m, (DelayAuto, Delay)):
                    raise ValueError("modules cannot contain DelayAuto or Delay")

        self.join_modules = join_modules
        # Pulse imports DelayAuto, so resolve the concrete table type at construction.
        from .pulse import TableLengthPulse

        tables = [
            m
            for branch in join_modules
            for m in branch
            if isinstance(m, TableLengthPulse)
        ]
        self.table_pulses: tuple[TableLengthPulse, ...] = tuple(tables)
        if tables:
            if len(tables) != len(join_modules) or any(
                len(branch) != 1 for branch in join_modules
            ):
                raise ValueError(
                    "Table-length Join requires one TableLengthPulse per branch"
                )
            if len({pulse.cfg.ch for pulse in tables}) != len(tables):
                raise ValueError("Table-length Join requires independent generators")
            if len({(pulse.idx_reg, len(pulse.lengths)) for pulse in tables}) != 1:
                raise ValueError(
                    "Table-length Join requires a shared index and table size"
                )

    def init(self, prog: ModularProgramV2) -> None:
        for list in self.join_modules:
            for m in list:
                m.init(prog)
        if self.table_pulses:
            from .dmem import LoadValue

            first = self.table_pulses[0]
            duration_reg = f"{first.name}_join_duration"
            durations = [
                max(
                    pulse.playback_parameters(prog, i)[2] for pulse in self.table_pulses
                )
                for i in range(len(first.lengths))
            ]
            self._duration_loader = LoadValue(
                f"{first.name}_join_load_duration",
                durations,
                idx_reg=first.idx_reg,
                val_reg=duration_reg,
            )
            self._duration_loader.init(prog)
            self._wait = DelayAuto(f"{first.name}_join_wait", t=duration_reg)
            self._wait.init(prog)

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        if self.table_pulses:
            # Set one common origin before emitting any runtime-length branch.
            prog.delay(t=t)
            prog.delay_auto(t=0.0)
            with prog.disable_delay():
                for pulse in self.table_pulses:
                    pulse.play_parallel(prog)
            self._duration_loader.run(prog)
            return self._wait.run(prog)

        logger.debug(
            "Join.run: %d parallel branches, t=%s",
            len(self.join_modules),
            t,
        )

        list_modules = list(list(list_mod) for list_mod in self.join_modules)
        cur_t_list: list[float | QickParam] = [t for _ in list_modules]

        def find_next_branch() -> int | None:
            min_i = None
            min_t = 0.0
            for i, (t, mod_list) in enumerate(
                zip(cur_t_list, list_modules, strict=False)
            ):
                if len(mod_list) == 0:
                    continue  # skip empty branch
                if isinstance(t, QickParam):
                    if not is_qick_param(t):
                        raise TypeError(
                            f"unsupported QickParam implementation: {type(t).__name__}"
                        )
                    t = t.minval()
                if min_i is None or t < min_t:
                    min_i = i
                    min_t = t
            return min_i

        with prog.disable_delay():
            # use interleaved execution to avoid execution delay afftected actual pulse time
            while (i := find_next_branch()) is not None:
                cur_t = cur_t_list[i]
                mod = list_modules[i].pop(0)

                if logger.isEnabledFor(logging.DEBUG):
                    prog.debug_macro(
                        f"{type(mod).__name__}({mod.name})", cur_t, prefix="\t"
                    )

                cur_t_list[i] = mod.run(prog, cur_t)

        return merge_max_length(*cur_t_list)

    def allow_rerun(self) -> bool:
        return all(m.allow_rerun() for mlist in self.join_modules for m in mlist)
