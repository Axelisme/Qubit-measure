from typing import Any
from unittest.mock import MagicMock

from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.ir.instructions import NopInst
from zcu_tools.program.v2.ir.node import BlockNode, InstNode, IRLoop, RootNode
from zcu_tools.program.v2.ir.pipeline import make_default_pipeline


def test_structural_loop_roundtrip():
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
            "p_addr": 4,
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

    prog = MagicMock()
    prog.tproccfg = {"pmem_size": 1024}
    builder = IRBuilder(prog)
    root = builder.build(prog_list, {}, meta_infos)

    # Verify IR structure
    assert len(root.insts) == 1
    loop = root.insts[0]
    assert isinstance(loop, IRLoop)
    assert loop.name == "loop1"
    assert loop.counter_reg == "r1"
    assert loop.n == 5
    # start_label / end_label are no longer captured during parse (generated at lower() time).

    # Unbuild (emits instructions)
    opt_insts, *_, cursor = builder.unbuild(root)

    # Verify emitted instructions (no markers).
    # IRLoop.emit() (do-while + guard, constant n: no guard) outputs:
    # REG_WR(init), Label(start), [BODY...], REG_WR(i+1), JUMP(start, IF+OP), Label(end)

    expected_cmds = ["REG_WR"]  # Init counter
    expected_cmds.append("NOP")  # Body
    expected_cmds.append("REG_WR")  # counter += 1
    expected_cmds.append("JUMP")  # Cond back-edge (IF=NS, OP=counter-n)

    # Note: Labels are extracted into a dictionary when creating binprog, but `emit()` outputs them as LabelInst dicts.
    cmds = [inst.get("CMD") for inst in opt_insts if "CMD" in inst]
    assert cmds == expected_cmds
    assert cursor.final_p_addr == 4
    assert cursor.final_line == 6


def test_pipeline_roundtrip_with_normalization():
    # Test that the pipeline preserves well-formed loops
    # Use a large trip count to avoid automatic unrolling (default max is 16)
    loop = IRLoop(name="r", counter_reg="c", n=100)
    root = RootNode(insts=[loop])

    pipeline = make_default_pipeline(pmem_capacity=8192)
    # Actually wait, make_default_pipeline takes pmem_capacity, not config.
    pipeline.config.enable_unroll_loop = False

    out_ir, _ctx = pipeline(root)

    # Check that it's still well-formed
    assert isinstance(out_ir.insts[0], IRLoop)
    out_loop = out_ir.insts[0]
    assert out_loop.counter_reg == "c"
    assert out_loop.n == 100


def test_irloop_emit_uses_s15_jump_for_large_pmem():
    root = RootNode(
        insts=[
            IRLoop(
                name="big",
                counter_reg="r1",
                n=5,
                body=BlockNode(insts=[InstNode(NopInst())]),
            )
        ]
    )

    prog = MagicMock()
    prog.tproccfg = {"pmem_size": 4096}
    builder = IRBuilder(prog)
    prog_list, *_, cursor = builder.unbuild(root)

    # Constant n: no guard. Body executes; counter += 1; cond back-edge.
    # Big-pmem path: back-edge target loaded into s15, then JUMP s15.
    cmds = [inst.get("CMD") for inst in prog_list]
    assert cmds == ["REG_WR", "NOP", "REG_WR", "REG_WR", "JUMP"]

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

    back_jump = prog_list[4]
    assert back_jump["CMD"] == "JUMP"
    assert back_jump["ADDR"] == "s15"
    assert back_jump["IF"] == "NS"
    assert back_jump["OP"] == "r1 - #5"
    assert "LABEL" not in back_jump

    assert cursor.final_p_addr == 5
