from __future__ import annotations

from typing import Any

from zcu_tools.program.v2.ir.instructions import NopInst, RegWriteInst
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode, IRLoop
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    Register,
    SrcKeyword,
)


def test_structural_loop_roundtrip():
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser
    from zcu_tools.program.v2.ir.linker import IRLinker

    # This test verifies that we can build a structural IR from instructions,
    # and unbuild it back to instructions.

    prog_list: list[dict[str, Any]] = [
        {"CMD": "REG_WR", "DST": "r1", "SRC": "imm", "LIT": "#0"},
        {"CMD": "TEST", "OP": "r1 - #5", "UF": "0"},
        {"CMD": "JUMP", "LABEL": "loop1_end", "IF": "NS"},
        {"CMD": "NOP"},
        {"CMD": "REG_WR", "DST": "r1", "SRC": "op", "OP": "r1+#1"},
        {"CMD": "JUMP", "LABEL": "loop1_start"},
    ]
    for i, p in enumerate(prog_list):
        p["P_ADDR"] = i

    meta_infos = [
        {
            "kind": "meta",
            "type": "LOOP_START",
            "name": "loop1",
            "info": {"counter_reg": "r1", "n": 5},
            "p_addr": 0,
        },
        {"kind": "label", "name": "loop1_start", "p_addr": 0},
        {
            "kind": "meta",
            "type": "LOOP_BODY_START",
            "name": "loop1",
            "info": {},
            "p_addr": 3,
        },
        {
            "kind": "meta",
            "type": "LOOP_BODY_END",
            "name": "loop1",
            "info": {},
            "p_addr": 5,
        },
        {"kind": "label", "name": "loop1_end", "p_addr": 6},
        {
            "kind": "meta",
            "type": "LOOP_END",
            "name": "loop1",
            "info": {},
            "p_addr": 6,
        },
    ]

    linker = IRLinker()
    lexer = IRLexer()
    parser = IRParser(pmem_size=1024)

    labels = {"loop1_start": "&0", "loop1_end": "&6"}
    insts = linker.unlink(prog_list, labels, meta_infos)
    blocks = lexer.lex(insts)
    root = parser.parse(blocks)

    # Verify IR structure
    assert len(root.insts) == 1
    loop = root.insts[0]
    assert isinstance(loop, IRLoop)
    assert loop.name == "loop1"
    assert loop.counter_reg == Register("r1")
    assert loop.n == 5
    # start_label / end_label are no longer captured during parse (generated at lower() time).

    # Unbuild (emits instructions)
    opt_blocks = parser.unparse(root)
    opt_insts = lexer.flatten(opt_blocks)
    opt_prog_list, *_, cursor = linker.link(opt_insts)

    # Verify emitted instructions (no markers).
    # IRLoop.emit() (do-while + guard, constant n: no guard) outputs:
    # REG_WR(init), Label(start), [BODY... including i+1], JUMP(start, IF+OP), Label(end)

    expected_cmds = ["REG_WR"]  # Init counter
    expected_cmds.append("NOP")  # Body
    expected_cmds.append("REG_WR")  # counter += 1
    expected_cmds.append("JUMP")  # Cond back-edge (IF=S, OP=counter-n)

    # Note: Labels are extracted into a dictionary when creating binprog, but `emit()` outputs them as LabelInst dicts.
    cmds = [inst.get("CMD") for inst in opt_prog_list if "CMD" in inst]
    expected_cmds.insert(3, "TEST")
    assert cmds == expected_cmds
    assert cursor.final_p_addr == 5
    assert cursor.final_line == 7


def test_nested_loop_unparse_parse_roundtrip():
    """A nested IRLoop survives unparse → parse → unparse unchanged.

    The outer loop body holds an inner IRLoop; unparse must emit both
    structures' markers, and parse must reconstruct the same nesting so a
    second unparse produces an identical chunk stream.
    """
    from zcu_tools.program.v2.ir.factory import IRParser
    from zcu_tools.program.v2.ir.instructions import MetaInst

    inner = IRLoop(
        name="inner",
        counter_reg=Register("r2"),
        n=2,
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )
    outer = IRLoop(
        name="outer",
        counter_reg=Register("r1"),
        n=3,
        body=BlockNode(insts=[inner]),
    )
    root = BlockNode(insts=[outer])

    parser = IRParser(pmem_size=1024)

    chunks1 = parser.unparse(root)
    reparsed = parser.parse(chunks1)

    # The reconstructed tree must still be outer(IRLoop) → inner(IRLoop).
    assert len(reparsed.insts) == 1
    re_outer = reparsed.insts[0]
    assert isinstance(re_outer, IRLoop)
    assert re_outer.name == "outer"
    assert isinstance(re_outer.body, BlockNode)
    re_inner = next(c for c in re_outer.body.insts if isinstance(c, IRLoop))
    assert re_inner.name == "inner"

    # A second unparse must yield the same structural marker sequence.
    chunks2 = IRParser(pmem_size=1024).unparse(reparsed)
    markers1 = [c.type for c in chunks1 if isinstance(c, MetaInst)]
    markers2 = [c.type for c in chunks2 if isinstance(c, MetaInst)]
    assert markers1 == markers2
    assert markers1.count("LOOP_START") == 2


def test_irloop_emit_uses_s15_jump_for_large_pmem():
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser
    from zcu_tools.program.v2.ir.linker import IRLinker

    bb_nop: BasicBlockNode = BasicBlockNode(insts=[NopInst()])
    bb_inc: BasicBlockNode = BasicBlockNode(
        insts=[
            RegWriteInst(
                dst=Register("r1"),
                src=SrcKeyword.OP,
                op=AluExpr(Register("r1"), AluOp.ADD, Immediate(1)),
            )
        ]
    )
    loop_node: IRLoop = IRLoop(
        name="big",
        counter_reg=Register("r1"),
        n=5,
        body=BlockNode(insts=[bb_nop, bb_inc]),
    )
    root = BlockNode(insts=[loop_node])

    lexer = IRLexer()
    parser = IRParser(pmem_size=4096)
    linker = IRLinker()

    insts = lexer.flatten(parser.unparse(root))
    prog_list, *_, cursor = linker.link(insts)

    # Constant n: no guard. Body already contains counter += 1; then cond back-edge.
    # Big-pmem path: back-edge target loaded into s15, then JUMP s15.
    cmds = [inst.get("CMD") for inst in prog_list]
    assert cmds == ["REG_WR", "NOP", "REG_WR", "REG_WR", "TEST", "JUMP"]

    init = prog_list[0]
    assert init["DST"] == "r1"
    assert init["LIT"] == "#0"

    incr = prog_list[2]
    assert incr["DST"] == "r1"
    assert incr["OP"] == "r1 + #1"

    write_label = prog_list[3]
    assert write_label["CMD"] == "REG_WR"
    assert write_label["DST"] == "s15"
    assert write_label["SRC"] == "label"
    assert write_label["LABEL"] == "big_start"

    test_inst = prog_list[4]
    assert test_inst["CMD"] == "TEST"
    assert test_inst["OP"] == "r1 - #5"
    assert test_inst["UF"] == "1"

    back_jump = prog_list[5]
    assert back_jump["CMD"] == "JUMP"
    assert back_jump["ADDR"] == "s15"
    assert back_jump["IF"] == "S"
    assert "LABEL" not in back_jump

    assert cursor.final_p_addr == 6
