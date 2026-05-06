from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from typing_extensions import Any

from .labels import Label
from .instructions import Instruction, LabelInst, MetaInst


@dataclass(frozen=True)
class IRCursor:
    final_p_addr: int
    final_line: int


class IRLinker:
    """Responsible for flattening the IR tree, assigning physical addresses, and resolving labels."""

    def link(
        self, inst_list: list[Instruction]
    ) -> tuple[list[dict], dict[str, str], list[dict[str, Any]], IRCursor]:
        prog_list: list[dict] = []
        labels: dict[str, str] = {}
        meta_infos: list[dict[str, Any]] = []

        p_addr = 0
        line = 0
        for inst in inst_list:
            if isinstance(inst, MetaInst):
                meta_infos.append(
                    dict(
                        kind="meta",
                        type=inst.type,
                        name=inst.name,
                        info=inst.info,
                        p_addr=p_addr,
                    )
                )
                continue
            line += 1
            if isinstance(inst, LabelInst):
                labels[str(inst.name)] = f"&{p_addr}"
                meta_infos.append(
                    dict(kind="label", name=str(inst.name), p_addr=p_addr)
                )
            else:
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
        # Parse fallback labels (labels added manually without calling _add_label)
        labels_by_addr: dict[int, list[str]] = defaultdict(list)
        tracked_labels = {m["name"] for m in meta_infos if m.get("kind") == "label"}
        for label_name in labels.keys():
            if label_name not in tracked_labels:
                raise ValueError(f"Untracked label {label_name!r} in labels dict")

        logical_insts: list[Instruction] = []
        label_map: dict[str, Label] = {}

        def get_label(name: str) -> Label:
            if name not in label_map:
                label_map[name] = Label.make_new(name)
            return label_map[name]

        # Group tracked markers by p_addr
        markers_by_addr: dict[int, list[dict]] = defaultdict(list)
        for m in meta_infos:
            markers_by_addr[m["p_addr"]].append(m)

        for d in prog_list:
            p_addr = d["P_ADDR"]

            # 1. Insert tracked markers from meta_infos for this index
            for m in markers_by_addr.get(p_addr, []):
                if m["kind"] == "label":
                    logical_insts.append(LabelInst(name=get_label(m["name"])))
                elif m["kind"] == "meta":
                    logical_insts.append(
                        MetaInst(type=m["type"], name=m["name"], info=m.get("info", {}))
                    )

            # Remove from markers_by_addr so we don't process it again for trailing
            if p_addr in markers_by_addr:
                del markers_by_addr[p_addr]

            # 2. Insert untracked fallback labels pointing to this P_ADDR
            for name in labels_by_addr.get(p_addr, []):
                logical_insts.append(LabelInst(name=get_label(name)))

            # 3. Insert the instruction itself
            logical_insts.append(Instruction.from_dict(d, label_map=label_map))

        # Handle trailing markers from meta_infos
        # Any markers left in markers_by_addr are trailing
        for p_addr, markers in sorted(markers_by_addr.items()):
            for m in markers:
                if m["kind"] == "label":
                    logical_insts.append(LabelInst(name=get_label(m["name"])))
                elif m["kind"] == "meta":
                    logical_insts.append(
                        MetaInst(type=m["type"], name=m["name"], info=m.get("info", {}))
                    )

        # Handle trailing fallback labels
        if prog_list:
            last_dict = prog_list[-1]
            last_inst = Instruction.from_dict(last_dict, label_map=label_map)
            max_addr = last_dict["P_ADDR"] + last_inst.addr_inc
            for name in labels_by_addr.get(max_addr, []):
                logical_insts.append(LabelInst(name=get_label(name)))

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
