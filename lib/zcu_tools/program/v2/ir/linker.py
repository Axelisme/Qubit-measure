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
        position instructions and labels correctly.
        """
        # Ensure P_ADDR exists for all instructions. If not, assume sequential.
        # This is common in tests or when unlinking a list that hasn't been linked.
        working_prog_list = []
        next_p_addr = 0
        for inst in prog_list:
            d = dict(inst)
            if "P_ADDR" not in d:
                d["P_ADDR"] = next_p_addr
                next_p_addr += 2 if d.get("CMD") == "WAIT" else 1
            else:
                next_p_addr = d["P_ADDR"] + (2 if d.get("CMD") == "WAIT" else 1)
            working_prog_list.append(d)

        labels_by_addr: dict[int, list[str]] = defaultdict(list)
        for label_name, label_addr in labels.items():
            p_addr = self._parse_label_addr(label_name, label_addr)
            labels_by_addr[p_addr].append(label_name)

        logical_insts: list[Instruction] = []

        # Max address for trailing labels
        if working_prog_list:
            last = working_prog_list[-1]
            max_addr = last["P_ADDR"] + (2 if last.get("CMD") == "WAIT" else 1)
        else:
            max_addr = 0

        # Validate label addresses
        for p_addr in labels_by_addr:
            if p_addr < 0 or p_addr > max_addr:
                raise ValueError(
                    f"Label address out of range: {p_addr} (valid: 0..{max_addr})"
                )

        for d in working_prog_list:
            p_addr = d["P_ADDR"]
            # Insert labels pointing to this address
            for name in labels_by_addr.get(p_addr, []):
                logical_insts.append(LabelInst(name=name))

            # Convert dictionary back to Instruction object
            logical_insts.append(Instruction.from_dict(d))

        # Handle trailing labels
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
