from __future__ import annotations

from zcu_tools.program.v2.ir.instructions import (
    GenericInst,
    JumpInst,
    LabelInst,
    RegWriteInst,
    TestInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.node import BlockNode, InstNode, IRLoop, RootNode
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
    assert [item.inst.lit for item in out.insts] == ["#1", "#1", "#1"]


def test_unroll_loop_expands_loop_with_internal_label():
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(
                    insts=[
                        InstNode(LabelInst(name="inner")),
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
            InstNode(GenericInst(cmd="NOP")),
        ]
    )

    out = DeadWriteEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 2
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(out.insts[0].inst, RegWriteInst)
    assert out.insts[0].inst.extra_args["LIT"] == "#2"
    assert isinstance(out.insts[1], InstNode)
    assert isinstance(out.insts[1].inst, GenericInst)


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
        item.inst.extra_args.get("LIT")
        for item in out.insts
        if isinstance(item.inst, RegWriteInst)
    ] == [
        "#1",
        "#2",
    ]


def test_dead_label_elimination_removes_unreferenced_label():
    root = RootNode(
        insts=[
            InstNode(LabelInst(name="dead")),
            InstNode(GenericInst(cmd="NOP")),
        ]
    )

    out = DeadLabelEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(out.insts[0].inst, GenericInst)


def test_dead_label_elimination_keeps_referenced_label():
    root = RootNode(
        insts=[
            InstNode(LabelInst(name="keep")),
            InstNode(JumpInst(label="keep")),
        ]
    )

    out = DeadLabelEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 2
    assert isinstance(out.insts[0].inst, LabelInst)
    assert isinstance(out.insts[1].inst, JumpInst)


def test_default_pipeline_orders_new_passes_first():
    pipeline = make_default_pipeline(pmem_capacity=8192)

    assert [type(pass_).__name__ for pass_ in pipeline.passes] == [
        "UnrollSmallLoopPass",
        "DeadWriteEliminationPass",
        "DeadLabelEliminationPass",
        "ZeroDelayDCEPass",
        "TimedInstructionMergePass",
    ]
