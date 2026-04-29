from __future__ import annotations

import logging

from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Optional, Sequence, Union

from .base import Module
from .util import merge_max_length, round_timestamp

if TYPE_CHECKING:
    from zcu_tools.program.v2.modular import ModularProgramV2
    from zcu_tools.program.v2.lower import LowerCtx
    from zcu_tools.program.v2.ir import IRNode

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

    def lower(self, ctx: LowerCtx) -> IRNode:
        from ..ir import IRDelay, IRMeta

        return IRDelay(
            duration=self.delay,
            auto=False,
            tag=self.tag,
            meta=IRMeta(source_module=".".join(ctx.parent_path + (self.name,))),
        )

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

    def lower(self, ctx: LowerCtx) -> IRNode:
        from ..ir import IRMeta, IRSoftDelay

        return IRSoftDelay(
            duration=self.delay,
            meta=IRMeta(source_module=".".join(ctx.parent_path + (self.name,))),
        )

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

    def lower(self, ctx: LowerCtx) -> IRNode:
        from ..ir import IRDelay, IRMeta

        return IRDelay(
            duration=self.t,
            auto=True,
            gens=self.gens,
            ros=self.ros,
            tag=self.tag,
            meta=IRMeta(source_module=".".join(ctx.parent_path + (self.name,))),
        )

    def allow_rerun(self) -> bool:
        return self.tag is None


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

    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        logger.debug(
            "Join.run: %d parallel branches, t=%s",
            len(self.join_modules),
            t,
        )

        list_modules = list(list(list_mod) for list_mod in self.join_modules)
        cur_t_list: list[Union[float, QickParam]] = [t for _ in list_modules]

        def find_next_branch() -> Optional[int]:
            min_i = None
            min_t = 0.0
            for i, (t, mod_list) in enumerate(zip(cur_t_list, list_modules)):
                if len(mod_list) == 0:
                    continue  # skip empty branch
                if isinstance(t, QickParam):
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

        end_t = merge_max_length(*cur_t_list)

        return end_t

    def allow_rerun(self) -> bool:
        return all(m.allow_rerun() for mlist in self.join_modules for m in mlist)
