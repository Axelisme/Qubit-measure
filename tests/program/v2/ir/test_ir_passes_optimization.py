"""Pipeline cooperation, convergence, invariants, and default-pipeline integration."""

from __future__ import annotations

import pytest
from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    NopInst,
    RegWriteInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode, IRLoop, IRNode
from zcu_tools.program.v2.ir.operands import (
    Immediate,
    Register,
    SrcKeyword,
)
from zcu_tools.program.v2.ir.passes.loop import UnrollLoopPass
from zcu_tools.program.v2.ir.pipeline import (
    AbsIRTreePass,
    IRPipeLine,
    PipeLineConfig,
    PipeLineContext,
    make_default_pipeline,
)

from .pass_test_support import (
    flatten_root,
    register_loop,
    run_chunk_passes_on_root,
    transform_only_child,
    walk_basic_blocks,
)

# ---------------------------------------------------------------------------
# UnrollLoopPass + DeadWriteElimination
# ---------------------------------------------------------------------------


def test_unroll_full_expansion_removes_overwritten_writes_in_body():
    """body: REG_WR r1 imm #1. n=2. Full expansion = 2 copies.
    Only the last write to r1 should survive.
    """
    root = BlockNode(
        insts=[
            IRLoop(
                name="L",
                counter_reg=Register("r0"),
                n=2,
                body=BlockNode(
                    insts=[
                        BasicBlockNode(
                            insts=[
                                RegWriteInst(
                                    dst=Register("r1"),
                                    src=SrcKeyword.IMM,
                                    lit=Immediate(1),
                                )
                            ]
                        ),
                    ]
                ),
            )
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=512)
    ctx = PipeLineContext(
        config=pipeline.config, pmem_budget=512, available_regs={"r14"}
    )

    # 1. Unroll
    out = transform_only_child(root, UnrollLoopPass(), ctx)

    # 2. Merge blocks so the linear pass sees the writes together in one block
    from zcu_tools.program.v2.ir.passes.control_flow import BlockMergePass

    out = run_chunk_passes_on_root(out, [BlockMergePass()])

    # 3. Chunk passes (inc DeadWriteElimination)
    out = run_chunk_passes_on_root(out, pipeline.chunk_passes)

    bbs = _collect_all_basic_blocks(out)
    # Should find only one write to r1 across all expanded copies.
    r1_writes = [
        i
        for bb in bbs
        for i in bb.insts
        if isinstance(i, RegWriteInst) and i.dst.name == "r1"
    ]
    assert len(r1_writes) == 1


# ---------------------------------------------------------------------------
# Global Pipeline Integration
# ---------------------------------------------------------------------------


def test_default_pipeline_can_disable_all_optimization_passes():
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser

    """Disabling all pass flags should keep the IR layout unchanged."""
    root = BlockNode(
        insts=[
            BasicBlockNode(
                insts=[
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(1)
                    ),
                    RegWriteInst(
                        dst=Register("r1"), src=SrcKeyword.IMM, lit=Immediate(2)
                    ),
                ]
            ),
            BasicBlockNode(
                labels=[LabelInst(name=Label("dead"))],
                insts=[
                    TimeInst(c_op="inc_ref", lit=Immediate(0)),
                    TimeInst(c_op="inc_ref", lit=Immediate(1)),
                    TimeInst(c_op="inc_ref", lit=Immediate(2)),
                ],
            ),
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=8192)
    pipeline.config.disable_all_opt = True

    lexer = IRLexer()
    parser = IRParser(pmem_size=8192)
    insts = lexer.flatten(parser.unparse(root))

    out_insts, _ctx = pipeline(insts)
    out = parser.parse(lexer.lex(out_insts))
    assert len(out.insts) == 2
    bb0 = out.insts[0]
    assert isinstance(bb0, BasicBlockNode)
    assert isinstance(bb0.insts[0], RegWriteInst)
    assert isinstance(bb0.insts[1], RegWriteInst)
    bb1 = out.insts[1]
    assert isinstance(bb1, BasicBlockNode)
    assert len(bb1.labels) == 1
    assert isinstance(bb1.labels[0], LabelInst)
    time_inst = bb1.insts[0]
    assert isinstance(time_inst, TimeInst)
    assert str(time_inst.lit) == "#0"


def _collect_all_basic_blocks(root: BlockNode) -> list[BasicBlockNode]:
    return list(walk_basic_blocks(root))


def test_pipeline_non_convergence_emits_warning(caplog):
    """The public pipeline warns when an injected pass fails to converge."""
    import logging

    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkListPass,
        ChunkList,
    )

    class OscillatingPass(AbsChunkListPass):
        """Always reports changed=True — simulates an oscillating pass."""

        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            return chunks, True

    ctx = PipeLineContext(config=PipeLineConfig(max_opt_iterations=2), pmem_budget=1024)
    insts = flatten_root(BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]))

    with caplog.at_level(logging.WARNING, logger="zcu_tools.program.v2.ir.pipeline"):
        IRPipeLine(ctx.config, [], [OscillatingPass()])(insts)

    assert any("did not converge" in r.message for r in caplog.records)
    assert any("OscillatingPass" in r.message for r in caplog.records)


def test_pipeline_non_convergence_diagnostic_does_not_mutate_live_chunks(caplog):
    import logging

    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkListPass,
        ChunkList,
    )

    class DiagnosticMutatingPass(AbsChunkListPass):
        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            block = chunks[0]
            assert isinstance(block, BasicBlockNode)
            block.insts.append(NopInst())
            return chunks, True

    ctx = PipeLineContext(config=PipeLineConfig(max_opt_iterations=2), pmem_budget=1024)
    block = BasicBlockNode(insts=[NopInst()])
    insts = flatten_root(BlockNode(insts=[block]))

    with caplog.at_level(logging.WARNING, logger="zcu_tools.program.v2.ir.pipeline"):
        out, _ = IRPipeLine(ctx.config, [], [DiagnosticMutatingPass()])(insts)

    # Two U-shape phases run at most two live iterations each.
    # Diagnostic probes must not add instructions to the emitted program.
    assert out == [NopInst()] * 5
    assert any("DiagnosticMutatingPass" in r.message for r in caplog.records)


def test_pipeline_disable_opt_violation_raises():
    """A ChunkPass that changes word count of a disable_opt block must raise ValueError."""
    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkPass,
        ChunkList,
    )

    class WordCountChangingPass(AbsChunkPass):
        """Appends a NopInst to any disable_opt block — illegal word-count change."""

        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            changed = False
            for chunk in chunks:
                if isinstance(chunk, BasicBlockNode) and chunk.disable_opt:
                    chunk.insts.append(NopInst())
                    changed = True
            return chunks, changed

    import pytest

    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    fixed_bb = BasicBlockNode(insts=[NopInst()], disable_opt=True)
    with pytest.raises(ValueError, match="disable_opt"):
        IRPipeLine(ctx.config, [WordCountChangingPass()], [])(
            flatten_root(BlockNode(insts=[fixed_bb]))
        )


def test_pipeline_disable_opt_deletion_raises():
    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkPass,
        ChunkList,
    )

    class DeletingPass(AbsChunkPass):
        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            return [
                chunk
                for chunk in chunks
                if not (isinstance(chunk, BasicBlockNode) and chunk.disable_opt)
            ], True

    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    fixed_bb = BasicBlockNode(insts=[NopInst()], disable_opt=True)

    with pytest.raises(ValueError, match="disable_opt"):
        IRPipeLine(ctx.config, [DeletingPass()], [])(
            flatten_root(BlockNode(insts=[fixed_bb]))
        )


def test_pipeline_disable_opt_replacement_raises():
    from zcu_tools.program.v2.ir.pipeline import (
        AbsChunkPass,
        ChunkList,
    )

    class ReplacingPass(AbsChunkPass):
        def process(self, chunks: ChunkList, ctx: PipeLineContext):
            return [
                BasicBlockNode(insts=list(chunk.insts), disable_opt=True)
                if isinstance(chunk, BasicBlockNode) and chunk.disable_opt
                else chunk
                for chunk in chunks
            ], True

    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    fixed_bb = BasicBlockNode(insts=[NopInst()], disable_opt=True)

    with pytest.raises(ValueError, match="disable_opt"):
        IRPipeLine(ctx.config, [ReplacingPass()], [])(
            flatten_root(BlockNode(insts=[fixed_bb]))
        )


def test_pipeline_tree_pass_missing_disable_opt_block_raises():
    class DroppingTreePass(AbsIRTreePass):
        def transform(self, node: IRNode, ctx: PipeLineContext):
            if isinstance(node, BlockNode):
                return BlockNode()
            return None

    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    root = BlockNode(insts=[BasicBlockNode(insts=[NopInst()], disable_opt=True)])

    with pytest.raises(ValueError, match="disable_opt"):
        IRPipeLine(ctx.config, [], [], [DroppingTreePass()])(flatten_root(root))


def test_pipeline_lowers_basic_block_root_replacement():
    class ReplacingRootPass(AbsIRTreePass):
        def transform(self, node: IRNode, ctx: PipeLineContext) -> IRNode | None:
            if isinstance(node, BlockNode):
                return BasicBlockNode(insts=[NopInst(), NopInst()])
            return None

    pipeline = IRPipeLine(PipeLineConfig(), [], [], [ReplacingRootPass()])
    try:
        out, _ = pipeline([NopInst()])
    except TypeError as exc:
        if str(exc) != "IRParser._unparse_node: unexpected node type NopInst":
            raise
        raise AssertionError(
            "The pipeline must lower a legal BasicBlockNode root replacement"
        ) from exc

    assert out == [NopInst(), NopInst()]


def test_pipeline_rejects_non_block_loop_body_replacement():
    class ReplacingLoopBodyPass(AbsIRTreePass):
        def transform(self, node: IRNode, ctx: PipeLineContext):
            if (
                isinstance(node, BlockNode)
                and len(node.insts) == 1
                and isinstance(node.insts[0], BasicBlockNode)
            ):
                return BasicBlockNode(insts=[NopInst()])
            return None

    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=3,
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )
    root = BlockNode(insts=[loop])
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)

    with pytest.raises(TypeError, match="IRLoop.body optimization"):
        IRPipeLine(ctx.config, [], [], [ReplacingLoopBodyPass()])(flatten_root(root))


def test_pipeline_chunk_passes_reach_joint_fixed_point():
    """Label removal enables block merge, which enables increment merging."""
    from zcu_tools.program.v2.ir.passes.control_flow import (
        BlockMergePass,
        DeadLabelEliminationPass,
    )
    from zcu_tools.program.v2.ir.passes.timeline import TimedMergePass, ZeroDelayDCEPass

    insts: list[Instruction] = [
        TimeInst(c_op="inc_ref", lit=Immediate(2)),
        LabelInst(name=Label("unused"), can_remove=True),
        TimeInst(c_op="inc_ref", lit=Immediate(0)),
        TimeInst(c_op="inc_ref", lit=Immediate(3)),
        NopInst(),
    ]
    pipeline = IRPipeLine(
        PipeLineConfig(),
        [TimedMergePass(), ZeroDelayDCEPass()],
        [BlockMergePass(), DeadLabelEliminationPass()],
    )
    out, _ = pipeline(insts)
    assert out == [TimeInst(c_op="inc_ref", lit=Immediate(5)), NopInst()]
    again, _ = pipeline(out)
    assert again == out


def test_pipeline_unroll_revisits_generated_dispatch():
    """The tree driver lowers dispatch nodes introduced by loop expansion."""
    from zcu_tools.program.v2.ir.passes.control_flow import DmemDispatchPass

    root = BlockNode(insts=[register_loop("lp", n_reg="r1")])
    pipeline = IRPipeLine(
        PipeLineConfig(pmem_capacity=512, max_unroll_factor=2),
        [],
        [],
        [UnrollLoopPass(), DmemDispatchPass()],
    )
    out, ctx = pipeline(flatten_root(root))
    assert len(ctx.dmem_tables) == 1
    labels = {inst.name for inst in out if isinstance(inst, LabelInst)}
    assert all(label in labels for label in ctx.dmem_tables[0])
    assert any(
        isinstance(inst, JumpInst) and inst.addr == Register("s15") for inst in out
    )
