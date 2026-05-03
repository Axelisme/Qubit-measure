from __future__ import annotations

import pytest
from zcu_tools.program.v2.ir.instructions import (
    LabelInst,
    RegWriteInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.linker import IRLinker
from zcu_tools.program.v2.ir.node import InstNode, RootNode


def test_linker_wait_address_calculation():
    """Verify that IRLinker correctly handles WAIT with addr_inc=2."""

    # Sequence:
    # L1:
    #   REG_WR
    # L2:
    #   WAIT
    # L3:
    #   REG_WR
    # L4:

    ir = RootNode(
        insts=[
            InstNode(LabelInst(name=Label("L1"))),
            InstNode(RegWriteInst(dst="r1", src="imm", extra_args={"LIT": "#1"})),
            InstNode(LabelInst(name=Label("L2"))),
            InstNode(WaitInst()),
            InstNode(LabelInst(name=Label("L3"))),
            InstNode(RegWriteInst(dst="r2", src="imm", extra_args={"LIT": "#2"})),
            InstNode(LabelInst(name=Label("L4"))),
        ]
    )

    linker = IRLinker()
    inst_list = []
    ir.emit(inst_list)
    prog_list, labels, meta_infos, cursor = linker.link(inst_list)

    # Expected addresses:
    # L1: 0
    # REG_WR (at 0): occupies 1 word -> next addr: 1
    # L2: 1
    # WAIT (at 1): occupies 2 words -> next addr: 3
    # L3: 3
    # REG_WR (at 3): occupies 1 word -> next addr: 4
    # L4: 4

    assert labels["L1"] == "&0"
    assert labels["L2"] == "&1"
    assert labels["L3"] == "&3"
    assert labels["L4"] == "&4"

    assert prog_list[1]["P_ADDR"] == 1
    assert prog_list[1]["CMD"] == "WAIT"
    assert prog_list[2]["P_ADDR"] == 3
    assert cursor.final_p_addr == 4
    assert cursor.final_line == 7


def test_linker_cursor_counts_wait_and_trailing_labels():
    ir = RootNode(
        insts=[
            InstNode(LabelInst(name=Label("L1"))),
            InstNode(WaitInst()),
            InstNode(LabelInst(name=Label("L2"))),
            InstNode(RegWriteInst(dst="r0", src="imm", extra_args={"LIT": "#0"})),
        ]
    )

    linker = IRLinker()
    inst_list = []
    ir.emit(inst_list)
    prog_list, labels, _meta_infos, cursor = linker.link(inst_list)

    assert labels == {"L1": "&0", "L2": "&2"}
    assert [inst["P_ADDR"] for inst in prog_list] == [0, 2]
    assert cursor.final_p_addr == 3
    assert cursor.final_line == 4


def test_linker_wait_roundtrip():
    """Verify that unlink() correctly restores labels after WAIT."""

    ir = RootNode(
        insts=[
            InstNode(LabelInst(name=Label("L1"))),
            InstNode(WaitInst()),
            InstNode(LabelInst(name=Label("L2"))),
        ]
    )

    linker = IRLinker()
    inst_list = []
    ir.emit(inst_list)
    prog_list, labels, meta_infos, _cursor = linker.link(inst_list)

    # Roundtrip: unlink
    logical_insts = linker.unlink(prog_list, labels, meta_infos)

    # Compare CMD/LABEL
    actual_cmds = []
    for inst in logical_insts:
        if isinstance(inst, LabelInst):
            actual_cmds.append({"LABEL": str(inst.name)})
        else:
            actual_cmds.append(inst.to_dict())

    expected = [
        {"LABEL": "L1"},
        {"CMD": "WAIT", "C_OP": "time"},
        {"LABEL": "L2"},
    ]
    assert actual_cmds == expected


if __name__ == "__main__":
    pytest.main([__file__])
