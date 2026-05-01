from __future__ import annotations

from qick.asm_v2 import QickProgramV2
from typing_extensions import Any

from .builder import IRBuilder
from .pipeline import PipeLineConfig, make_default_pipeline


class IRComplieMixin(QickProgramV2):
    def compile(self):
        self._make_asm()
        self.optimize_asm()
        self._make_binprog()

    def optimize_asm(self) -> None:
        insts: list[dict[str, Any]] = self.prog_list

        builder = IRBuilder()
        ir = builder.build(insts)

        config = PipeLineConfig()
        pipeline = make_default_pipeline(config)

        opt_ir, _ctx = pipeline(ir)

        opt_insts, opt_labels = builder.unbuild(opt_ir)

        self.prog_list = opt_insts
        self.labels = opt_labels

    def _add_asm(self, inst: dict[str, Any], addr_inc: int = 1) -> None:
        super()._add_asm(inst, addr_inc)

    def _add_label(self, label: str) -> None:
        super()._add_label(label)
        self.prog_list.append({"LABEL": label})
