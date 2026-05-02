from __future__ import annotations

from typing import Optional

from qick.asm_v2 import QickProgramV2
from typing_extensions import Any

from .builder import IRBuilder
from .pipeline import PipeLineConfig, make_default_pipeline


class IRCompileMixin(QickProgramV2):
    def __init__(self, *args: Any, **kwargs: Any):
        self.meta_infos: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _add_label(self, label: str):
        self.meta_infos.append(dict(kind="label", name=label, p_addr=self.p_addr))
        super()._add_label(label)  # type: ignore

    def _add_meta(
        self, type: str, name: str, info: Optional[dict[str, Any]] = None
    ) -> None:
        self.meta_infos.append(
            dict(kind="meta", type=type, name=name, info=info or {}, p_addr=self.p_addr)
        )

    def compile(self):
        self._make_asm()
        self.optimize_asm()
        self._make_binprog()

    def optimize_asm(self) -> None:
        insts: list[dict[str, Any]] = self.prog_list
        meta_infos = self.meta_infos

        builder = IRBuilder()
        ir = builder.build(insts, self.labels, meta_infos)

        pmem_cap = self.tproccfg.get("pmem_size")
        pipeline = make_default_pipeline(
            PipeLineConfig(pmem_budget=int(0.8 * pmem_cap))
        )

        opt_ir, _ctx = pipeline(ir)

        opt_insts, opt_labels, opt_meta_infos, cursor = builder.unbuild(opt_ir)

        self.prog_list = opt_insts
        self.labels = opt_labels
        self.meta_infos = opt_meta_infos

        self.p_addr = cursor.final_p_addr
        self.line = cursor.final_line
