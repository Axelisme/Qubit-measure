from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from typing_extensions import Any

from .instructions import BaseInst, Instruction, LabelInst, MetaInst
from .labels import Label


@dataclass(frozen=True)
class IRCursor:
    final_p_addr: int
    final_line: int


class IRLinker:
    """Assigns physical addresses to a flat instruction list and resolves labels."""

    def link(
        self, inst_list: list[Instruction]
    ) -> tuple[list[dict], dict[str, str], list[dict[str, Any]], IRCursor]:
        """Link a flat instruction list into QICK-compatible dicts."""
        prog_list: list[dict] = []
        labels: dict[str, str] = {}
        meta_infos: list[dict[str, Any]] = []

        p_addr = 0
        line = 0
        for inst in inst_list:
            if isinstance(inst, MetaInst):
                d = inst.to_dict()
                d["p_addr"] = p_addr
                meta_infos.append(d)
                continue
            line += 1
            if isinstance(inst, LabelInst):
                labels[str(inst.name)] = f"&{p_addr}"
                d = inst.to_dict()
                d["p_addr"] = p_addr
                meta_infos.append(d)
            else:
                assert isinstance(inst, BaseInst)
                d = inst.to_dict()
                d["P_ADDR"] = p_addr
                d["LINE"] = line
                prog_list.append(d)
                p_addr += inst.addr_inc

        return (
            prog_list,
            labels,
            meta_infos,
            IRCursor(final_p_addr=p_addr, final_line=line),
        )

    def unlink(
        self,
        prog_list: list[dict[str, Any]],
        labels: dict[str, Any],
        meta_infos: list[dict[str, Any]],
    ) -> list[Instruction]:
        Label.reset()

        # 1. Strict Validation: Compare labels.keys() with labels tracked in meta_infos
        tracked_label_names = {
            m["name"] for m in meta_infos if m.get("kind") == "label"
        }
        provided_label_names = set(labels.keys())

        if tracked_label_names != provided_label_names:
            only_in_tracked = tracked_label_names - provided_label_names
            only_in_provided = provided_label_names - tracked_label_names
            msg = "Mismatch between tracked labels in meta_infos and provided labels dict."
            if only_in_tracked:
                msg += f" Missing in labels dict: {sorted(list(only_in_tracked))}."
            if only_in_provided:
                msg += f" Not found in meta_infos: {sorted(list(only_in_provided))}."
            raise ValueError(msg)

        # 2. Pre-allocation: Initialize Label identities based on meta_infos
        # We sort to ensure deterministic behavior if suffixes are ever needed,
        # though since we just reset, they will match exactly.
        for name in sorted(list(tracked_label_names)):
            Label.make_new(name)

        # Group tracked markers by p_addr
        markers_by_addr: dict[int, list[dict]] = defaultdict(list)
        for m in meta_infos:
            markers_by_addr[m["p_addr"]].append(m)

        logical_insts: list[Instruction] = []

        for d in prog_list:
            p_addr = d["P_ADDR"]

            # 1. Insert tracked markers from meta_infos for this index
            for m in markers_by_addr.get(p_addr, []):
                if m["kind"] == "label":
                    logical_insts.append(
                        LabelInst.from_dict(
                            {
                                "kind": "label",
                                "name": m["name"],
                                "can_remove": m.get("can_remove", False),
                            }
                        )
                    )
                elif m["kind"] == "meta":
                    logical_insts.append(
                        MetaInst.from_dict(
                            {
                                "kind": "meta",
                                "type": m["type"],
                                "name": m["name"],
                                "info": m.get("info", {}),
                            }
                        )
                    )

            # Remove from markers_by_addr so we don't process it again for trailing
            if p_addr in markers_by_addr:
                del markers_by_addr[p_addr]

            # 2. Insert the instruction itself
            logical_insts.append(BaseInst.from_dict(d))

        # Handle trailing markers from meta_infos
        for p_addr, markers in sorted(markers_by_addr.items()):
            for m in markers:
                if m["kind"] == "label":
                    logical_insts.append(
                        LabelInst.from_dict(
                            {
                                "kind": "label",
                                "name": m["name"],
                                "can_remove": m.get("can_remove", False),
                            }
                        )
                    )
                elif m["kind"] == "meta":
                    logical_insts.append(
                        MetaInst.from_dict(
                            {
                                "kind": "meta",
                                "type": m["type"],
                                "name": m["name"],
                                "info": m.get("info", {}),
                            }
                        )
                    )

        return logical_insts

    @staticmethod
    def _parse_label_addr(label_name: str, label_addr: Any) -> int:
        if isinstance(label_addr, int):
            return label_addr

        if isinstance(label_addr, str):
            addr_str = label_addr[1:] if label_addr.startswith("&") else label_addr
            try:
                return int(addr_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label address for {label_name!r}: {label_addr!r}"
                ) from exc

        raise ValueError(
            f"Invalid label address type for {label_name!r}: {label_addr!r}"
        )
