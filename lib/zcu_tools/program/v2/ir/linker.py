from __future__ import annotations

from collections import defaultdict
from typing_extensions import Any

from .instructions import Instruction, LabelInst


class IRLinker:
    """Responsible for flattening the IR tree, assigning physical addresses, and resolving labels."""

    def link(self, inst_list: list[Instruction]) -> tuple[list[dict], dict[str, str]]:
        opt_prog_list: list[dict] = []
        opt_labels: dict[str, str] = {}

        p_addr = 0
        for inst in inst_list:
            if isinstance(inst, LabelInst):
                # It's a standalone label marker
                opt_labels[inst.name] = f"&{p_addr}"
            else:
                # It's an executable instruction; assign fresh P_ADDR
                d = inst.to_dict()
                d["P_ADDR"] = p_addr
                opt_prog_list.append(d)
                p_addr += inst.addr_inc

        return opt_prog_list, opt_labels

    def unlink(
        self, prog_list: list[dict[str, Any]], labels: dict[str, Any]
    ) -> list[Instruction]:
        """Reconstruct logical instruction sequence from QICK `prog_list + labels`.

        Uses `P_ADDR` from the dictionaries and addresses from `labels` to
        position instructions and labels correctly. Assumes prog_list comes from
        QICK (all instructions have P_ADDR set).
        """
        # Parse labels into address → label_names mapping
        labels_by_addr: dict[int, list[str]] = defaultdict(list)
        for label_name, label_addr in labels.items():
            p_addr = self._parse_label_addr(label_name, label_addr)
            labels_by_addr[p_addr].append(label_name)

        logical_insts: list[Instruction] = []

        # Single pass: insert labels at their addresses, then insert instructions
        for d in prog_list:
            p_addr = d["P_ADDR"]
            # Insert all labels pointing to this address
            for name in labels_by_addr.get(p_addr, []):
                logical_insts.append(LabelInst(name=name))
            # Insert the instruction itself
            logical_insts.append(Instruction.from_dict(d))

        # Handle trailing labels (labels pointing beyond the last instruction)
        if prog_list:
            last_dict = prog_list[-1]
            last_inst = Instruction.from_dict(last_dict)
            max_addr = last_dict["P_ADDR"] + last_inst.addr_inc
            for name in labels_by_addr.get(max_addr, []):
                logical_insts.append(LabelInst(name=name))

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
