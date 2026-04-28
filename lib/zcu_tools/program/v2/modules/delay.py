from __future__ import annotations

import logging

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Optional, Sequence, Union

from .base import Module
from .util import merge_max_length, round_timestamp

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2

logger = logging.getLogger(__name__)


class Delay(Module):
    def __init__(
        self, name: str, delay: Union[float, QickParam], tag: Optional[str] = None
    ) -> None:
        self.name = name
        self.delay = delay
        self.tag = tag

    def init(self, prog: ModularProgramV2) -> None: ...

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        prog.delay(t=round_timestamp(prog, self.delay), tag=self.tag)

        return 0.0  # reset reference time

    def allow_rerun(self) -> bool:
        return self.tag is None


class SoftDelay(Module):
    def __init__(self, name: str, delay: Union[float, QickParam]) -> None:
        self.name = name
        self.delay = delay

    def init(self, prog: ModularProgramV2) -> None:
        pass

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:

        return round_timestamp(prog, self.delay)

    def allow_rerun(self) -> bool:
        return True


class DelayAuto(Module):
    def __init__(
        self,
        name: str,
        t: Union[float, QickParam, str] = 0.0,
        gens: bool = True,
        ros: bool = True,
        tag: Optional[str] = None,
    ) -> None:
        self.name = name
        self.t = t
        self.gens = gens
        self.ros = ros
        self.tag = tag

        if tag is not None and isinstance(t, str):
            raise ValueError("DelayAuto with tag cannot have t as a register name")

    def init(self, prog: ModularProgramV2) -> None: ...

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        if isinstance(self.t, str):
            prog.delay_reg_auto(time_reg=self.t, gens=self.gens, ros=self.ros)
        else:
            prog.delay_auto(t=self.t, gens=self.gens, ros=self.ros, tag=self.tag)  # type: ignore
        return 0.0

    def allow_rerun(self) -> bool:
        return self.tag is None


class Join(Module):
    def __init__(self, *args: Union[Module, Sequence[Module]]) -> None:
        self.name = ""

        if len(args) == 0:
            raise ValueError("Join must contain at least one module")

        join_modules = [[m] if isinstance(m, Module) else m for m in args]
        for list in join_modules:
            for m in list:
                if isinstance(m, (DelayAuto, Delay)):
                    raise ValueError("modules cannot contain DelayAuto or Delay")

        self.join_modules = join_modules

    def init(self, prog: ModularProgramV2) -> None:
        for list in self.join_modules:
            for m in list:
                m.init(prog)

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        logger.debug(
            "Join.run: %d parallel branches, t=%s",
            len(self.join_modules),
            t,
        )

        end_t_list = []
        with prog.disable_delay():
            for list in self.join_modules:
                cur_t = t
                for m in list:
                    if logger.isEnabledFor(logging.DEBUG):
                        prog.debug_macro(
                            f"{type(m).__name__}({m.name})", cur_t, prefix="\t"
                        )
                    cur_t = m.run(prog, cur_t)

                end_t_list.append(cur_t)
        end_t = merge_max_length(*end_t_list)

        return end_t

    def allow_rerun(self) -> bool:
        return all(m.allow_rerun() for mlist in self.join_modules for m in mlist)
