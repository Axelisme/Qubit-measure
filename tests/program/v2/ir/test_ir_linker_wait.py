from __future__ import annotations

import pytest
from zcu_tools.program.v2.ir.factory import IRLexer, IRParser
from zcu_tools.program.v2.ir.instructions import (
    LabelInst,
    RegWriteInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.linker import IRLinker
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode
from zcu_tools.program.v2.ir.operands import Immediate, Register, SrcKeyword


def _link_root(linker: IRLinker, root: BlockNode):
    inst_list = IRLexer().flatten(IRParser().unparse(root))
    return linker.link(inst_list)


def test_linker_wait_address_calculation():
    """Verify that IRLinker correctly handles WAIT with addr_inc=2."""
    Label.reset()

    # Sequence:
    # L1:
    #   REG_WR
    # L2:
    #   WAIT
    # L3:
    #   REG_WR
    # L4:

    bb_l1: BasicBlockNode = BasicBlockNode(
        labels=[LabelInst(name=Label.make_new("L1"))],
        insts=[RegWriteInst(dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1))],
    )
    bb_l2: BasicBlockNode = BasicBlockNode(
        labels=[LabelInst(name=Label.make_new("L2"))],
        insts=[WaitInst(c_op="time")],
    )
    bb_l3: BasicBlockNode = BasicBlockNode(
        labels=[LabelInst(name=Label.make_new("L3"))],
        insts=[RegWriteInst(dst=Register("r2"), src=SrcKeyword.IMM, lit=Immediate(2))],
    )
    bb_l4: BasicBlockNode = BasicBlockNode(labels=[LabelInst(name=Label.make_new("L4"))])
    ir = BlockNode(insts=[bb_l1, bb_l2, bb_l3, bb_l4])

    linker = IRLinker()
    prog_list, labels, meta_infos, cursor = _link_root(linker, ir)

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
    Label.reset()
    bb2_l1: BasicBlockNode = BasicBlockNode(
        labels=[LabelInst(name=Label.make_new("L1"))],
        insts=[WaitInst(c_op="time")],
    )
    bb2_l2: BasicBlockNode = BasicBlockNode(
        labels=[LabelInst(name=Label.make_new("L2"))],
        insts=[RegWriteInst(dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(0))],
    )
    ir = BlockNode(insts=[bb2_l1, bb2_l2])

    linker = IRLinker()
    prog_list, labels, _meta_infos, cursor = _link_root(linker, ir)

    assert labels == {"L1": "&0", "L2": "&2"}
    assert [inst["P_ADDR"] for inst in prog_list] == [0, 2]
    assert cursor.final_p_addr == 3
    assert cursor.final_line == 4


def test_linker_wait_roundtrip():
    """Verify that unlink() correctly restores labels after WAIT."""
    Label.reset()

    ir = BlockNode(
        insts=[
            BasicBlockNode(labels=[LabelInst(name=Label.make_new("L1"))], insts=[WaitInst(c_op="time")]),
            BasicBlockNode(labels=[LabelInst(name=Label.make_new("L2"))]),
        ]
    )

    linker = IRLinker()
    prog_list, labels, meta_infos, _cursor = _link_root(linker, ir)

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
        {"LABEL": "&L1"},
        {"CMD": "WAIT", "C_OP": "time"},
        {"LABEL": "&L2"},
    ]
    assert actual_cmds == expected


if __name__ == "__main__":
    pytest.main([__file__])

