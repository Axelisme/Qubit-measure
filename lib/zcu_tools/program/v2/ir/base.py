from __future__ import annotations

from qick.asm_v2 import QickProgramV2
from typing_extensions import Any

from .builder import IRBuilder
from .pipeline import PipeLineConfig, make_default_pipeline


class IRCompileMixin(QickProgramV2):
    def compile(self):
        self._make_asm()
        self.optimize_asm()
        self._make_binprog()

    def optimize_asm(self) -> None:
        insts: list[dict[str, Any]] = self.prog_list

        builder = IRBuilder()
        ir = builder.build(insts, self.labels)

        config = PipeLineConfig()
        pipeline = make_default_pipeline(config)

        opt_ir, _ctx = pipeline(ir)

        opt_insts, opt_labels = builder.unbuild(opt_ir)

        self.prog_list = opt_insts
        self.labels = opt_labels

        self.p_addr = len(opt_insts)
        self.line = len(opt_insts)
