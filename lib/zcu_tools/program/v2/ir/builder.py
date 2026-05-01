from __future__ import annotations

from .factory import InstructionStream, parse_root
from .instructions import Instruction
from .node import IRNode, RootNode


class IRBuilder:
    def build(self, prog_list: list[dict], labels: dict[str, str]) -> RootNode:
        inst_list = [Instruction.from_dict(d) for d in prog_list]
        stream = InstructionStream(inst_list)
        root = parse_root(stream)
        root.labels = labels

        if stream.peek() is not None:
            raise ValueError("Unparsed instructions remaining in stream")

        return root

    def unbuild(self, ir: IRNode) -> tuple[list[dict], dict[str, str]]:
        if not isinstance(ir, RootNode):
            raise ValueError("IR node passed to unbuild must be a RootNode")

        prog_list: list[dict] = []
        ir.emit(prog_list)

        opt_prog_list: list[dict] = []
        opt_labels: dict[str, str] = {}

        # Scan and calculate actual physical addresses (p_addr) for each instruction.
        # Remove standalone LabelInst outputs and update opt_labels instead.
        p_addr = 0
        for d in prog_list:
            if "LABEL" in d and "CMD" not in d:
                # It's a standalone label marker
                opt_labels[d["LABEL"]] = f"&{p_addr}"
            else:
                opt_prog_list.append(d)
                p_addr += 1

        # Include original labels from ir.labels if they weren't explicitly re-emitted,
        # though ideally all used labels should be emitted.
        for k, v in ir.labels.items():
            if k not in opt_labels:
                opt_labels[k] = v

        return opt_prog_list, opt_labels
