import pytest
from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.ir.instructions import (
    GenericInst,
    Instruction,
    JumpInst,
)
from zcu_tools.program.v2.ir.linker import IRLinker
from zcu_tools.program.v2.ir.node import BlockNode, IRBranch, IRBranchCase, IRLoop, InstNode


def test_instruction_parses_jump_label_to_jumpinst():
    inst = Instruction.from_dict({"CMD": "JUMP", "LABEL": "target"})

    assert isinstance(inst, JumpInst)
    assert inst.label == "target"
    assert inst.if_cond is None


def test_branch_roundtrip_preserves_cases():
    """Verify IRBranch.emit() includes dispatch + cases in order."""
    dispatch_inst = GenericInst(cmd="JUMP", args={"LABEL": "case_0"})
    case_0_inst = GenericInst(cmd="CASE_0_BODY")
    case_1_inst = GenericInst(cmd="CASE_1_BODY")

    case_0 = IRBranchCase(name="0", insts=[InstNode(case_0_inst)])
    case_1 = IRBranchCase(name="1", insts=[InstNode(case_1_inst)])

    branch = IRBranch(
        name="sel",
        dispatch=BlockNode(insts=[InstNode(dispatch_inst)]),
        cases=[case_0, case_1],
    )

    # Emit to Instruction list
    inst_list: list[Instruction] = []
    branch.emit(inst_list)

    # Should have: dispatch_inst, case_0_inst, case_1_inst
    assert len(inst_list) == 3
    assert inst_list[0].cmd == "JUMP"
    assert inst_list[1].cmd == "CASE_0_BODY"
    assert inst_list[2].cmd == "CASE_1_BODY"


def test_unlink_inserts_labels_and_strips_p_addr():
    linker = IRLinker()
    prog_list = [
        {"CMD": "REG_WR", "DST": "r0", "SRC": "imm", "LIT": "#1", "P_ADDR": 0},
        {"CMD": "JUMP", "LABEL": "end", "P_ADDR": 1},
    ]
    labels = {"start": "&0", "end": "&2"}

    logical_prog_list = linker.unlink(prog_list, labels)

    assert logical_prog_list == [
        {"LABEL": "start"},
        {"CMD": "REG_WR", "DST": "r0", "SRC": "imm", "LIT": "#1"},
        {"CMD": "JUMP", "LABEL": "end"},
        {"LABEL": "end"},
    ]


def test_unlink_supports_multiple_labels_same_address():
    linker = IRLinker()
    prog_list = [{"CMD": "NOP", "P_ADDR": 0}]
    labels = {"first": "&0", "second": "&0"}

    logical_prog_list = linker.unlink(prog_list, labels)

    assert logical_prog_list == [
        {"LABEL": "first"},
        {"LABEL": "second"},
        {"CMD": "NOP"},
    ]


def test_unlink_rejects_out_of_range_address():
    linker = IRLinker()
    prog_list = [{"CMD": "NOP", "P_ADDR": 0}]
    labels = {"bad": "&2"}

    with pytest.raises(ValueError, match="out of range"):
        linker.unlink(prog_list, labels)


def test_builder_build_accepts_qick_labels_map():
    builder = IRBuilder()
    prog_list = [
        {"CMD": "__META__", "TYPE": "LOOP_START", "NAME": "loop1", "ARGS": {"counter_reg": "r1", "n": 5}, "P_ADDR": 0},
        {"CMD": "REG_WR", "DST": "r1", "SRC": "imm", "LIT": "#0", "P_ADDR": 1},
        {"CMD": "TEST", "OP": "r1 - #5", "UF": "0", "P_ADDR": 2},
        {"CMD": "JUMP", "LABEL": "loop1_end", "IF": "NS", "P_ADDR": 3},
        {"CMD": "__META__", "TYPE": "LOOP_BODY_START", "NAME": "loop1", "P_ADDR": 4},
        {"CMD": "NOP", "P_ADDR": 5},
        {"CMD": "__META__", "TYPE": "LOOP_BODY_END", "NAME": "loop1", "P_ADDR": 6},
        {"CMD": "JUMP", "LABEL": "loop1_start", "P_ADDR": 7},
        {"CMD": "__META__", "TYPE": "LOOP_END", "NAME": "loop1", "P_ADDR": 8},
    ]
    labels = {"loop1_start": "&1", "loop1_end": "&8"}

    root = builder.build(prog_list, labels)

    assert len(root.insts) == 1
    loop = root.insts[0]
    assert isinstance(loop, IRLoop)
    assert loop.start_label == "loop1_start"
    assert loop.end_label == "loop1_end"
