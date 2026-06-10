from __future__ import annotations

import logging
from typing import Any, Optional

from qick.asm_v2 import QickProgramV2

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
        self, type: str, name: str, info: dict[str, Any] | None = None
    ) -> None:
        self.meta_infos.append(
            dict(kind="meta", type=type, name=name, info=info or {}, p_addr=self.p_addr)
        )

    def compile(self) -> None:
        # Reset structural metadata on each compile to avoid stale markers.
        self.meta_infos = []

        self._make_asm()
        self.optimize_asm()
        self._make_binprog()

    def optimize_asm(self) -> None:
        insts: list[dict[str, Any]] = self.prog_list
        labels = self.labels
        meta_infos = self.meta_infos

        linker = IRLinker()
        logical_insts = linker.unlink(insts, labels, meta_infos)

        # dmem dispatch tables (Phase 7) are appended after the IR layer; tell
        # the pipeline where in dmem they may start (current buffer length).
        dmem_base = len(getattr(self, "_dmem_buffer", []))

        pipeline = make_default_pipeline(pmem_capacity=self.tproccfg["pmem_size"])
        opt_insts, ctx = pipeline(logical_insts, dmem_base_offset=dmem_base)

        opt_prog_list, opt_labels, opt_meta_infos, cursor = linker.link(opt_insts)

        self.prog_list = opt_prog_list
        self.labels = opt_labels
        self.meta_infos = opt_meta_infos

        self.p_addr = cursor.final_p_addr
        self.line = cursor.final_line

        # Materialize dmem dispatch tables: resolve each table's entry labels
        # to program addresses and append them to dmem. The resolve step in the
        # pipeline already rewrote the DmemAddr operands to these base offsets.
        self._materialize_dmem_tables(ctx, opt_labels, dmem_base)

    def _materialize_dmem_tables(
        self,
        ctx: Any,
        opt_labels: dict[str, Any],
        dmem_base: int,
    ) -> None:
        if not ctx.dmem_tables:
            return

        # Map each program address to the instruction physically at it, so the
        # debug log can show what each dispatch entry actually jumps to.
        # (WAIT occupies 2 words, so not every P_ADDR has an entry.)
        inst_at_addr: dict[int, dict[str, Any]] = {
            d["P_ADDR"]: d for d in self.prog_list if "P_ADDR" in d
        }

        def _inst_summary(addr: int) -> str:
            d = inst_at_addr.get(addr)
            if d is None:
                return "(no inst)"
            fields = " ".join(
                f"{k}={v}" for k, v in d.items() if k not in ("P_ADDR", "LINE")
            )
            return fields or "(empty)"

        cursor = dmem_base
        for idx, table_labels in enumerate(ctx.dmem_tables):
            entry_addrs = [
                IRLinker._parse_label_addr(lbl.name, opt_labels[lbl.name])
                for lbl in table_labels
            ]
            offset = self.add_dmem(entry_addrs)  # type: ignore[attr-defined]
            if offset != cursor:
                raise RuntimeError(
                    f"dmem dispatch table allocation mismatch: pipeline reserved "
                    f"offset {cursor}, add_dmem returned {offset}."
                )
            logger.debug(
                "dmem dispatch: materialized table #%d at dmem offset %d, %d entries:",
                idx,
                offset,
                len(entry_addrs),
            )
            for i, (lbl, addr) in enumerate(zip(table_labels, entry_addrs)):
                logger.debug(
                    "  dmem[%d] %s -> P_ADDR %d: %s",
                    offset + i,
                    lbl.name,
                    addr,
                    _inst_summary(addr),
                )
            cursor += len(entry_addrs)
        logger.debug(
            "dmem dispatch: materialized %d table(s), dmem offsets [%d, %d)",
            len(ctx.dmem_tables),
            dmem_base,
            cursor,
        )
