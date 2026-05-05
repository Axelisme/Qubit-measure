from __future__ import annotations

import logging

from qick.asm_v2 import AsmInst, IncReg, Jump, Label, Macro, WriteLabel, WriteReg

from .meta import MetaMacro

logger = logging.getLogger(__name__)


def _needs_big_jump(prog) -> bool:
    tproccfg = getattr(prog, "tproccfg", None)
    if not isinstance(tproccfg, dict):
        return False
    pmem_size = tproccfg.get("pmem_size")
    return isinstance(pmem_size, int) and pmem_size > 2**11


class OpenInnerLoop(Macro):
    """Register-driven counterpart of qick's OpenLoop."""

    # fields: name (str), counter_reg (str), n (str | int)
    def __init__(
        self,
        name: str,
        counter_reg: str,
        n: int | str,
        *,
        range_hint: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(name=name, counter_reg=counter_reg, n=n, range_hint=range_hint)

    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"

        mapped_counter = prog._get_reg(self.counter_reg)
        mapped_n = prog._get_reg(self.n) if isinstance(self.n, str) else self.n
        n_term = f"#{mapped_n}" if isinstance(mapped_n, int) else mapped_n

        cond_jump: list[Macro] = []
        if _needs_big_jump(prog):
            cond_jump.append(WriteLabel(label=end))
            cond_jump.append(
                AsmInst(
                    inst={
                        "CMD": "JUMP",
                        "IF": "NS",
                        "OP": f"{mapped_counter} - {n_term}",
                        "ADDR": "s15",
                    },
                    addr_inc=1,
                )
            )
        else:
            cond_jump.append(
                AsmInst(
                    inst={
                        "CMD": "JUMP",
                        "IF": "NS",
                        "OP": f"{mapped_counter} - {n_term}",
                        "LABEL": end,
                    },
                    addr_inc=1,
                )
            )

        return [
            MetaMacro(
                type="LOOP_START",
                name=self.name,
                info=dict(
                    counter_reg=mapped_counter, n=mapped_n, range_hint=self.range_hint
                ),
            ),
            WriteReg(dst=self.counter_reg, src=0),
            Label(label=start),
            *cond_jump,
            MetaMacro(type="LOOP_BODY_START", name=self.name),
        ]


class CloseInnerLoop(Macro):
    """Register-driven counterpart of qick's CloseLoop.

    Emits: counter increment, unconditional jump back to loop start, end label.
    Must be paired with an OpenLoopReg sharing the same ``name``.
    """

    # fields: name (str), counter_reg (str)
    def __init__(self, name: str, counter_reg: str) -> None:
        super().__init__(name=name, counter_reg=counter_reg)

    def expand(self, prog):  # type: ignore[override]
        start = f"{self.name}_start"
        end = f"{self.name}_end"
        if _needs_big_jump(prog):
            jump_back: list[Macro] = [
                WriteLabel(label=start),
                AsmInst(inst={"CMD": "JUMP", "ADDR": "s15"}, addr_inc=1),
            ]
        else:
            jump_back = [Jump(label=start)]
        return [
            IncReg(dst=self.counter_reg, src=1),
            MetaMacro(type="LOOP_BODY_END", name=self.name),
            *jump_back,
            Label(label=end),
            MetaMacro(type="LOOP_END", name=self.name),
        ]
