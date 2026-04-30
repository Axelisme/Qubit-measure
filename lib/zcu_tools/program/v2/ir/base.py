from __future__ import annotations

from qick.asm_v2 import QickProgramV2, AsmInst

from .pipeline import make_default_pipeline, PipeLineConfig
from .builder import IRBuilder


class IROptMixin(QickProgramV2):
    def compile(self):
        self._make_asm()
        self.optimize_asm()
        self._make_binprog()

    def optimize_asm(self) -> None:
        insts = self.prog_list
        labels = self.labels

        builder = IRBuilder()
        ir = builder.build(insts, labels)

        config = PipeLineConfig()
        pipeline = make_default_pipeline(config)

        opt_ir, _ctx = pipeline(ir)

        opt_insts, opt_labels = builder.unbuild(opt_ir)

        self.prog_list = opt_insts
        self.labels = opt_labels

    def _add_asm(self, inst: AsmInst, addr_inc: int = 1) -> None:
        super()._add_asm(inst, addr_inc)

    def _add_label(self, label: str) -> None:
        super()._add_label(label)
