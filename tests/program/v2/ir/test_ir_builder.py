import pytest
from zcu_tools.program.v2.ir.instructions import (
    BaseInst,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.linker import IRLinker
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode, IRBranch, IRLoop
from zcu_tools.program.v2.ir.operands import Immediate, Register, SrcKeyword


def test_instruction_parses_jump_label_to_jumpinst():
    from zcu_tools.program.v2.ir.labels import Label

    Label("target")
    inst = BaseInst.from_dict({"CMD": "JUMP", "LABEL": "target"})

    assert isinstance(inst, JumpInst)
    assert str(inst.label) == "target"
    assert inst.if_cond is None


def test_branch_lower_produces_basic_blocks():
    """Verify IRParser lowers IRBranch to a well-formed BasicBlockNode sequence."""
    from zcu_tools.program.v2.ir.factory import IRParser

    case_0_inst = RegWriteInst(dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1))
    case_1_inst = RegWriteInst(dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(2))

    case_0 = BlockNode(insts=[BasicBlockNode(insts=[case_0_inst])])
    case_1 = BlockNode(insts=[BasicBlockNode(insts=[case_1_inst])])

    branch = IRBranch(name="sel", compare_reg=Register("r_sel"), cases=[case_0, case_1])
    blocks = IRParser().unparse(BlockNode(insts=[branch]))

    meta_blocks = [b for b in blocks if isinstance(b, MetaInst)]
    bb_blocks = [b for b in blocks if isinstance(b, BasicBlockNode)]

    # BRANCH_START and BRANCH_END are standalone MetaInsts (not inside BasicBlockNode.insts).
    assert any(
        m.type == "BRANCH_START" and m.info.get("compare_reg") == "r_sel"
        for m in meta_blocks
    )
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

    case_meta = [m for m in meta_blocks if m.type.startswith("BRANCH_CASE_")]
    assert [m.type for m in case_meta] == [
        "BRANCH_CASE_START",
        "BRANCH_CASE_END",
        "BRANCH_CASE_START",
        "BRANCH_CASE_END",
    ]
    assert [m.name for m in case_meta] == ["0", "0", "1", "1"]


def test_branch_roundtrip_preserves_cases():
    from zcu_tools.program.v2.ir.factory import IRParser

    bb_0: BasicBlockNode = BasicBlockNode(
        insts=[RegWriteInst(dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1))]
    )
    bb_1: BasicBlockNode = BasicBlockNode(
        insts=[RegWriteInst(dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(2))]
    )
    case_0 = BlockNode(insts=[bb_0])
    case_1 = BlockNode(insts=[bb_1])
    root = BlockNode(
        insts=[
            IRBranch(name="sel", compare_reg=Register("r_sel"), cases=[case_0, case_1])
        ]
    )

    parser = IRParser()
    rebuilt = parser.parse(parser.unparse(root))

    branch = rebuilt.insts[0]
    assert isinstance(branch, IRBranch)
    assert branch.compare_reg == Register("r_sel")
    assert len(branch.cases) == 2
    assert isinstance(branch.cases[0], BlockNode)
    c0_bb = branch.cases[0].insts[0]
    assert isinstance(c0_bb, BasicBlockNode)
    c0_inst = c0_bb.insts[0]
    assert isinstance(c0_inst, RegWriteInst)
    assert str(c0_inst.lit) == "#1"
    assert isinstance(branch.cases[1], BlockNode)
    c1_bb = branch.cases[1].insts[0]
    assert isinstance(c1_bb, BasicBlockNode)
    c1_inst = c1_bb.insts[0]
    assert isinstance(c1_inst, RegWriteInst)
    assert str(c1_inst.lit) == "#2"
    assert rebuilt.insts == [branch]


def test_branch_roundtrip_strips_synthetic_small_pmem_jump_and_end_label():
    from zcu_tools.program.v2.ir.factory import IRParser

    case_0 = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r0"), src=SrcKeyword.IMM, lit=Immediate(1)
                    )
                ]
            )
        ]
    )
    case_1 = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)
                    )
                ]
            )
        ]
    )
    root = BlockNode(
        insts=[
            IRBranch(name="sel", compare_reg=Register("r_sel"), cases=[case_0, case_1])
        ]
    )

    rebuilt = IRParser(pmem_size=512).parse(IRParser(pmem_size=512).unparse(root))

    assert len(rebuilt.insts) == 1
    branch = rebuilt.insts[0]
    assert isinstance(branch, IRBranch)
    first_case = branch.cases[0]
    assert isinstance(first_case, BlockNode)
    assert len(first_case.insts) == 1
    only_block = first_case.insts[0]
    assert isinstance(only_block, BasicBlockNode)
    assert only_block.branch is None


def test_branch_roundtrip_strips_synthetic_big_pmem_jump_and_end_label():
    from zcu_tools.program.v2.ir.factory import IRParser

    case_0 = BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
    case_1 = BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
    case_2 = BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
    root = BlockNode(
        insts=[
            IRBranch(
                name="sel",
                compare_reg=Register("r_sel"),
                cases=[case_0, case_1, case_2],
            )
        ]
    )

    parser = IRParser(pmem_size=4096)
    rebuilt = parser.parse(parser.unparse(root))

    assert len(rebuilt.insts) == 1
    branch = rebuilt.insts[0]
    assert isinstance(branch, IRBranch)
    for case in branch.cases[:-1]:
        assert isinstance(case, BlockNode)
        assert len(case.insts) == 1
        case_head = case.insts[0]
        assert isinstance(case_head, BasicBlockNode)
        assert case_head.branch is None


def test_basic_block_rejects_metainst_in_insts():
    with pytest.raises(ValueError, match="MetaInst"):
        BasicBlockNode(insts=[MetaInst(type="BRANCH_START", name="bad")])  # type: ignore[list-item]


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
    assert loop.counter_reg == Register("r1")
    assert loop.n == 5
