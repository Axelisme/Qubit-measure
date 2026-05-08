from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.linker import IRLinker
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode, IRBranch, IRLoop


def test_instruction_parses_jump_label_to_jumpinst():
    inst = Instruction.from_dict({"CMD": "JUMP", "LABEL": "target"})

    assert isinstance(inst, JumpInst)
    assert str(inst.label) == "target"
    assert inst.if_cond is None


def test_branch_lower_produces_basic_blocks():
    """Verify IRParser lowers IRBranch to a well-formed BasicBlockNode sequence."""
    from zcu_tools.program.v2.ir.factory import IRParser
    from zcu_tools.program.v2.ir.instructions import MetaInst
    from zcu_tools.program.v2.ir.labels import Label
    from zcu_tools.program.v2.ir.node import RootNode

    Label.reset()

    case_0_inst = RegWriteInst(dst="r0", src="imm", lit="#1")
    case_1_inst = RegWriteInst(dst="r0", src="imm", lit="#2")

    case_0 = BlockNode(insts=[BasicBlockNode(insts=[case_0_inst])])
    case_1 = BlockNode(insts=[BasicBlockNode(insts=[case_1_inst])])

    branch = IRBranch(name="sel", compare_reg="r_sel", cases=[case_0, case_1])
    blocks = IRParser().unparse(RootNode(insts=[branch]))

    meta_blocks = [b for b in blocks if isinstance(b, MetaInst)]
    bb_blocks = [b for b in blocks if isinstance(b, BasicBlockNode)]

    # BRANCH_START and BRANCH_END are standalone MetaInsts (not inside BasicBlockNode.insts).
    assert any(m.type == "BRANCH_START" and m.info.get("compare_reg") == "r_sel"
               for m in meta_blocks)
    assert any(m.type == "BRANCH_END" for m in meta_blocks)

    # All non-meta elements are BasicBlockNodes.
    assert all(isinstance(b, (BasicBlockNode, MetaInst)) for b in blocks)
    assert len(bb_blocks) > 0

    # No BasicBlockNode.insts should contain MetaInst.
    assert all(not any(isinstance(i, MetaInst) for i in bb.insts) for bb in bb_blocks)

    # Both case instructions should appear in the flattened output.
    all_insts = []
    for bb in bb_blocks:
        all_insts.extend(bb.insts)
    assert case_0_inst in all_insts
    assert case_1_inst in all_insts


def test_unlink_inserts_labels_and_strips_p_addr():
    linker = IRLinker()
    prog_list = [
        {"CMD": "REG_WR", "DST": "r0", "SRC": "imm", "LIT": "#1", "P_ADDR": 0},
        {"CMD": "JUMP", "LABEL": "end", "P_ADDR": 1},
    ]
    labels = {"start": "&0", "end": "&2"}

    logical_insts = linker.unlink(
        prog_list,
        labels,
        meta_infos=[
            {"kind": "label", "name": "start", "p_addr": 0},
            {"kind": "label", "name": "end", "p_addr": 2},
        ],
    )

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

    logical_insts = linker.unlink(
        prog_list,
        labels,
        meta_infos=[
            {"kind": "label", "name": "first", "p_addr": 0},
            {"kind": "label", "name": "second", "p_addr": 0},
        ],
    )

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
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser
    from zcu_tools.program.v2.ir.linker import IRLinker
    
    linker = IRLinker()
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

    insts = linker.unlink(prog_list, labels, meta_infos)
    lexer = IRLexer()
    parser = IRParser(pmem_size=2048)
    blocks = lexer.lex(insts)
    root = parser.parse(blocks)

    assert len(root.insts) == 1
    loop = root.insts[0]
    assert isinstance(loop, IRLoop)
    assert loop.name == "loop1"
    assert loop.counter_reg == "r1"
    assert loop.n == 5
