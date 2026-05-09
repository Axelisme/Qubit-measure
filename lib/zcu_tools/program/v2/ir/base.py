from __future__ import annotations

import logging

from qick.asm_v2 import QickProgramV2
from typing_extensions import Any, Optional

from .linker import IRLinker
from .pipeline import make_default_pipeline

logger = logging.getLogger(__name__)


class IRCompileMixin(QickProgramV2):
    def __init__(self, *args: Any, **kwargs: Any):
        self.meta_infos: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _add_label(self, label: str) -> None:
        self.meta_infos.append(dict(kind="label", name=label, p_addr=self.p_addr))
        super()._add_label(label)  # type: ignore

    def _add_meta(
        self, type: str, name: str, info: Optional[dict[str, Any]] = None
    ) -> None:
        self.meta_infos.append(
            dict(kind="meta", type=type, name=name, info=info or {}, p_addr=self.p_addr)
        )

    def compile(self) -> None:
        # Reset structural metadata on each compile to avoid stale markers.
        self.meta_infos = []
        self._make_asm()
        self.optimize_asm()
        try:
            self._make_binprog()
        except Exception:
            for inst in self.prog_list:
                if isinstance(inst, dict) and inst.get("CMD") == "WMEM_WR":
                    logger.error("FAILED WMEM_WR: %s", inst)
            raise

    def optimize_asm(self) -> None:
        insts: list[dict[str, Any]] = self.prog_list
        labels = self.labels
        meta_infos = self.meta_infos

        linker = IRLinker()
        logical_insts = linker.unlink(insts, labels, meta_infos)

        pipeline = make_default_pipeline(pmem_capacity=self.tproccfg["pmem_size"])

        opt_insts, _ctx = pipeline(logical_insts)

        opt_prog_list, opt_labels, opt_meta_infos, cursor = linker.link(opt_insts)

        self.prog_list = opt_prog_list
        self.labels = opt_labels
        self.meta_infos = opt_meta_infos

        self.p_addr = cursor.final_p_addr
        self.line = cursor.final_line
