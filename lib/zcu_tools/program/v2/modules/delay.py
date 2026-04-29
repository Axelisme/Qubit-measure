from __future__ import annotations

import logging

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Optional, Sequence, Union

from .base import Module
from .util import merge_max_length, round_timestamp

if TYPE_CHECKING:
    from zcu_tools.program.v2.ir.builder import IRBuilder
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

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        builder.ir_delay(t=round_timestamp(prog, self.delay), tag=self.tag)
        return 0.0



class SoftDelay(Module):
    def __init__(self, name: str, delay: Union[float, QickParam]) -> None:
        self.name = name
        self.delay = delay

    def init(self, prog: ModularProgramV2) -> None:
        pass

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        # Keep legacy semantics: SoftDelay returns its own rounded delay and
        # does not accumulate with incoming t.
        return round_timestamp(prog, self.delay)



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

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        builder.ir_delay_auto(self.t, gens=self.gens, ros=self.ros, tag=self.tag)
        return 0.0



class Join(Module):
    def __init__(self, *args: Union[Module, Sequence[Module]]) -> None:
        self.name = ""

        if len(args) == 0:
            raise ValueError("Join must contain at least one module")

        join_modules = [[m] if isinstance(m, Module) else list(m) for m in args]
        for mod_list in join_modules:
            for m in mod_list:
                if isinstance(m, (DelayAuto, Delay)):
                    raise ValueError("modules cannot contain DelayAuto or Delay")

        self.join_modules = join_modules

    def init(self, prog: ModularProgramV2) -> None:
        for list in self.join_modules:
            for m in list:
                m.init(prog)

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        list_modules = [list(branch) for branch in self.join_modules]
        cur_t_list: list[Union[float, QickParam]] = [t for _ in list_modules]

        def find_next_branch() -> Optional[int]:
            min_i = None
            min_t = 0.0
            for i, (ct, mod_list) in enumerate(zip(cur_t_list, list_modules)):
                if not mod_list:
                    continue
                ct_val = ct.minval() if isinstance(ct, QickParam) else ct
                if min_i is None or ct_val < min_t:
                    min_i = i
                    min_t = ct_val
            return min_i

        with prog.disable_delay():
            while (i := find_next_branch()) is not None:
                cur_t = cur_t_list[i]
                mod = list_modules[i].pop(0)
                cur_t_list[i] = mod.ir_run(builder, cur_t, prog)

        end_t = merge_max_length(*cur_t_list)
        builder.ir_delay(end_t)
        builder.ir_delay_auto(0.0)
        return 0.0
