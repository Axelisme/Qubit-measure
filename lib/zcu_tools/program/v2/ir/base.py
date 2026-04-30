from __future__ import annotations

from typing_extensions import Any
from qick.asm_v2 import QickProgramV2

from .pipeline import make_default_pipeline, PipeLineConfig
from .builder import IRBuilder


class IRComplieMixin(QickProgramV2):
    def compile(self):
        self._make_asm()
        self.optimize_asm()
        self._make_binprog()

    def optimize_asm(self) -> None:
        insts: list[dict[str, Any]] = self.prog_list
        labels: dict[str, str] = self.labels

        builder = IRBuilder()
        ir = builder.build(insts, labels)

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
