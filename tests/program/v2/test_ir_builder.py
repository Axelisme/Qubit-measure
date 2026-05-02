from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.linker import IRLinker
from zcu_tools.program.v2.ir.node import InstNode, IRBranch, IRBranchCase, IRLoop


def test_instruction_parses_jump_label_to_jumpinst():
    inst = Instruction.from_dict({"CMD": "JUMP", "LABEL": "target"})

    assert isinstance(inst, JumpInst)
    assert str(inst.label) == "target"
    assert inst.if_cond is None


def test_branch_roundtrip_preserves_cases():
    """Verify IRBranch.emit() includes dispatch + cases in order."""
    case_0_inst = RegWriteInst(dst="r0", src="imm", lit="#1")
    case_1_inst = RegWriteInst(dst="r0", src="imm", lit="#2")

    case_0 = IRBranchCase(name="0", insts=[InstNode(case_0_inst)])
    case_1 = IRBranchCase(name="1", insts=[InstNode(case_1_inst)])

    branch = IRBranch(name="sel", cases=[case_0, case_1])

    # Emit to Instruction list
    inst_list: list[Instruction] = []
    branch.emit(inst_list)

    # TODO: compare with expected instruction list instead of just checking the cases are present
    raise AssertionError("\n" + "\n".join(str(inst) for inst in inst_list) + "\n")


def test_unlink_inserts_labels_and_strips_p_addr():
    linker = IRLinker()
    prog_list = [
        {"CMD": "REG_WR", "DST": "r0", "SRC": "imm", "LIT": "#1", "P_ADDR": 0},
        {"CMD": "JUMP", "LABEL": "end", "P_ADDR": 1},
    ]
    labels = {"start": "&0", "end": "&2"}

    logical_insts = linker.unlink(prog_list, labels, meta_infos=[])

    # Compare CMD/LABEL
    actual = []
    for inst in logical_insts:
        if isinstance(inst, LabelInst):
            actual.append({"LABEL": str(inst.name)})
        else:
            actual.append({"CMD": inst.to_dict()["CMD"]})

    assert actual == [
        {"LABEL": "start"},
        {"CMD": "REG_WR"},
        {"CMD": "JUMP"},
        {"LABEL": "end"},
    ]


def test_unlink_supports_multiple_labels_same_address():
    linker = IRLinker()
    prog_list = [{"CMD": "NOP", "P_ADDR": 0}]
    labels = {"first": "&0", "second": "&0"}

    logical_insts = linker.unlink(prog_list, labels, meta_infos=[])

    actual = []
    for inst in logical_insts:
        if isinstance(inst, LabelInst):
            actual.append({"LABEL": str(inst.name)})
        else:
            actual.append({"CMD": inst.to_dict()["CMD"]})

    assert actual == [
        {"LABEL": "first"},
        {"LABEL": "second"},
        {"CMD": "NOP"},
    ]


def test_builder_build_accepts_qick_labels_map():
    builder = IRBuilder()
    prog_list = [
        {"CMD": "REG_WR", "DST": "r1", "SRC": "imm", "LIT": "#0", "P_ADDR": 1},
        {"CMD": "TEST", "OP": "r1 - #5", "UF": "0", "P_ADDR": 2},
        {"CMD": "JUMP", "LABEL": "loop1_end", "IF": "NS", "P_ADDR": 3},
        {"CMD": "NOP", "P_ADDR": 5},
        {"CMD": "JUMP", "LABEL": "loop1_start", "P_ADDR": 7},
    ]
    labels = {"loop1_start": "&1", "loop1_end": "&8"}

    meta_infos = [
        {
            "kind": "meta",
            "type": "LOOP_START",
            "name": "loop1",
            "info": {"counter_reg": "r1", "n": 5},
            "p_addr": 1,
        },
        {"kind": "label", "name": "loop1_start", "p_addr": 1},
        {
            "kind": "meta",
            "type": "LOOP_BODY_START",
            "name": "loop1",
            "info": {},
            "p_addr": 5,
        },
        {
            "kind": "meta",
            "type": "LOOP_BODY_END",
            "name": "loop1",
            "info": {},
            "p_addr": 7,
        },
        {"kind": "label", "name": "loop1_end", "p_addr": 8},
        {
            "kind": "meta",
            "type": "LOOP_END",
            "name": "loop1",
            "info": {},
            "p_addr": 8,
        },
    ]

    root = builder.build(prog_list, labels, meta_infos)

    assert len(root.insts) == 1
    loop = root.insts[0]
    assert isinstance(loop, IRLoop)
    assert str(loop.start_label) == "loop1_start"
    assert str(loop.end_label) == "loop1_end"
