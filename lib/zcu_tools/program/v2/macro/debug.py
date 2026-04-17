from __future__ import annotations

import logging

from typing_extensions import Optional, Union
from qick.asm_v2 import AsmInst, Macro, QickParam

from ..utils import param2str

logger = logging.getLogger(__name__)


class PrintTimeStamp(Macro):
    """A helper macro to print the timestamp of the program."""

    def __init__(
        self,
        name: str,
        t: Union[float, QickParam] = 0.0,
        prefix: str = "",
        gen_chs: Optional[list[int]] = None,
        ro_chs: Optional[list[int]] = None,
    ) -> None:
        self.name = name
        self.t = t
        self.prefix = prefix
        self.gen_chs = gen_chs
        self.ro_chs = ro_chs

    def expand(self, prog) -> list[AsmInst]:  # type: ignore
        return []

    def preprocess(self, prog) -> None:
        gen_chs = self.gen_chs
        ro_chs = self.ro_chs
        if gen_chs is None:
            gen_chs = list(range(len(prog._gen_ts)))
        if ro_chs is None:
            ro_chs = list(range(len(prog._ro_ts)))

        lines = [self.prefix + self.name]
        lines.append(self.prefix + f"\tglobal time: {param2str(self.t)}")
        for ch in gen_chs:
            t = prog._gen_ts[ch]
            if t != 0.0 or self.gen_chs is not None:
                lines.append(self.prefix + f"\tgen[{ch}] " + param2str(t))
        for ch in ro_chs:
            t = prog._ro_ts[ch]
            if t != 0.0 or self.ro_chs is not None:
                lines.append(self.prefix + f"\t ro[{ch}] " + param2str(t))
        logger.debug("\n".join(lines))
