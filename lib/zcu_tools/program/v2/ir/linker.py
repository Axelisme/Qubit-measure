from __future__ import annotations

from .node import IRNode


class IRLinker:
    """Responsible for flattening the IR tree, assigning physical addresses, and resolving labels."""

    def link(self, ir: IRNode) -> tuple[list[dict], dict[str, str]]:
        raw_prog_list: list[dict] = []
        ir.emit(raw_prog_list)

        opt_prog_list: list[dict] = []
        opt_labels: dict[str, str] = {}

        p_addr = 0
        for d in raw_prog_list:
            if "LABEL" in d and "CMD" not in d:
                # It's a standalone label marker
                opt_labels[d["LABEL"]] = f"&{p_addr}"
            else:
                # It's an executable instruction; assign fresh P_ADDR
                d["P_ADDR"] = p_addr
                opt_prog_list.append(d)
                p_addr += 1

        return opt_prog_list, opt_labels
