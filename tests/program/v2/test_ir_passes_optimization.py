from __future__ import annotations

from typing import cast

from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    LabelInst,
    NopInst,
    RegWriteInst,
    TestInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import BlockNode, InstNode, IRLoop, IRNode, RootNode
from zcu_tools.program.v2.ir.passes import (
    DeadLabelEliminationPass,
    DeadWriteEliminationPass,
    UnrollSmallLoopPass,
)
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    make_default_pipeline,
)


def test_unroll_loop_expands_simple_loop_body():
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(
                    insts=[
                        InstNode(TimeInst(c_op="inc_ref", lit="#1")),
                    ]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 3
    assert all(isinstance(item, InstNode) for item in out.insts)
    assert [cast(TimeInst, cast(InstNode, item).inst).lit for item in out.insts] == ["#1", "#1", "#1"]


def test_unroll_loop_expands_loop_with_internal_label():
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(
                    insts=[
                        InstNode(LabelInst(name=Label("inner"))),
                        InstNode(TimeInst(c_op="inc_ref", lit="#1")),
                    ]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 6
    assert all(isinstance(item, InstNode) for item in out.insts)


def test_dead_write_elimination_removes_overwritten_write():
    root = RootNode(
        insts=[
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"})),
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"})),
            InstNode(NopInst()),
        ]
    )

    out = DeadWriteEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 2
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(cast(InstNode, out.insts[0]).inst, RegWriteInst)
    assert getattr(cast(InstNode, out.insts[0]).inst, "extra_args")["LIT"] == "#2"
    assert isinstance(out.insts[1], InstNode)
    assert isinstance(cast(InstNode, out.insts[1]).inst, NopInst)


def test_dead_write_elimination_keeps_write_before_read():
    root = RootNode(
        insts=[
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"})),
            InstNode(TestInst(op="s1 - #1")),
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"})),
        ]
    )

    out = DeadWriteEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 3
    assert [
        getattr(cast(InstNode, item).inst, "extra_args").get("LIT")
        for item in out.insts
        if isinstance(item, InstNode) and isinstance(item.inst, RegWriteInst)
    ] == [
        "#1",
        "#2",
    ]


def test_dead_label_elimination_removes_unreferenced_label():
    root = RootNode(
        insts=[
            InstNode(LabelInst(name=Label("dead"))),
            InstNode(NopInst()),
        ]
    )

    out = DeadLabelEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(cast(InstNode, out.insts[0]).inst, NopInst)


def test_dead_label_elimination_keeps_referenced_label():
    l = Label("keep")
    root = RootNode(
        insts=[
            InstNode(LabelInst(name=l)),
            InstNode(JumpInst(label=l)),
        ]
    )

    out = DeadLabelEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 2
    assert isinstance(cast(InstNode, out.insts[0]).inst, LabelInst)
    assert isinstance(cast(InstNode, out.insts[1]).inst, JumpInst)


def test_dead_label_elimination_keeps_pseudo_labels():
    """Pseudo labels (HERE, NEXT, PREV, SKIP) are assembler tokens, not IR labels.
    DeadLabelEliminationPass must never remove their LabelInst even when unreferenced."""
    for pseudo in ("HERE", "NEXT", "PREV", "SKIP"):
        root = RootNode(
            insts=[
                InstNode(LabelInst(name=Label(pseudo))),
                InstNode(NopInst()),
            ]
        )

        out = DeadLabelEliminationPass().process(
            root, PipeLineContext(config=PipeLineConfig())
        )

        assert len(out.insts) == 2, f"pseudo label {pseudo!r} was incorrectly removed"
        assert isinstance(cast(InstNode, out.insts[0]).inst, LabelInst)


def test_unroll_partial_unroll_produces_loop_plus_remainder():
    """n=20, body_cost=1 -> k=8; expect IRLoop(n=2) + 4 inlined copies."""
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=20,
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    config = PipeLineConfig(max_loop_unroll_count=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    # Should be: one IRLoop (the chunked loop) + 4 remainder InstNodes
    assert len(out.insts) == 5
    assert isinstance(out.insts[0], IRLoop)
    assert cast(IRLoop, out.insts[0]).n == 2  # 20 // 8
    assert all(isinstance(item, InstNode) for item in out.insts[1:])


def test_unroll_partial_unroll_no_remainder():
    """n=16, body_cost=1 -> k=8; expect IRLoop(n=2) with no remainder."""
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=16,
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    config = PipeLineConfig(max_loop_unroll_count=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRLoop)
    assert cast(IRLoop, out.insts[0]).n == 2


def test_unroll_heavy_body_skips_partial_unroll():
    """body_cost >= target_chunk_cost -> k=1; loop should be left unchanged."""
    from zcu_tools.program.v2.ir.instructions import PortWriteInst

    heavy_insts: list[IRNode] = [InstNode(PortWriteInst(dst="0", time="t0")) for _ in range(5)]
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=100,
                body=BlockNode(insts=heavy_insts),
            )
        ]
    )

    # body_cost = 5 * cost_wmem(4) = 20 -> k=1
    config = PipeLineConfig(max_loop_unroll_count=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRLoop)
    assert cast(IRLoop, out.insts[0]).n == 100


def test_unroll_exact_register_hint_fully_expands():
    """range_hint=(3,3) with register n -> treat as constant 3, fully expand."""
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n="r_count",
                range_hint=(3, 3),
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 3
    assert all(isinstance(item, InstNode) for item in out.insts)


def test_unroll_non_exact_register_hint_not_expanded():
    """range_hint=(2,5) is not exact -> loop must not be expanded."""
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n="r_count",
                range_hint=(2, 5),
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRLoop)


def test_unroll_no_hint_register_loop_not_expanded():
    """Register-driven loop with no hint must not be expanded."""
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n="r_count",
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRLoop)


def test_unroll_counter_sensitive_loop_not_expanded():
    """Loop body that reads the counter register must not be unrolled."""
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(
                    insts=[InstNode(TestInst(op="s0 - #1"))]  # reads counter
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRLoop)


def test_unroll_partial_unroll_clones_labels_safely():
    """Partial unroll with internal label must produce unique, non-dangling labels per copy."""
    Label.reset()
    inner = Label.make_new("inner")
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=20,
                body=BlockNode(
                    insts=[
                        InstNode(LabelInst(name=inner)),
                        InstNode(JumpInst(label=inner)),
                        InstNode(TimeInst(c_op="inc_ref", lit="#1")),
                    ]
                ),
            )
        ]
    )

    config = PipeLineConfig(max_loop_unroll_count=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    # Collect all label definitions and jump targets from emitted instructions
    from zcu_tools.program.v2.ir.instructions import Instruction
    emit: list[Instruction] = []
    out.emit(emit)

    defined = {str(inst.name) for inst in emit if isinstance(inst, LabelInst)}
    targets = {str(inst.label) for inst in emit if isinstance(inst, JumpInst) and inst.label}

    # Every jump target must have a corresponding label definition (no dangling)
    internal_targets = {t for t in targets if not t.startswith("loop")}
    assert internal_targets <= defined


def test_default_pipeline_orders_new_passes_first():
    pipeline = make_default_pipeline(pmem_capacity=8192)

    assert [type(pass_).__name__ for pass_ in pipeline.passes] == [
        "UnrollSmallLoopPass",
        "DeadWriteEliminationPass",
        "DeadLabelEliminationPass",
        "ZeroDelayDCEPass",
        "TimedInstructionMergePass",
    ]
