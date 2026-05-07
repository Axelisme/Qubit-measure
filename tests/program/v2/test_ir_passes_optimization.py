from __future__ import annotations

from typing import cast

from zcu_tools.program.v2.ir.instructions import (
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TestInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRLoop,
    IRNode,
    RootNode,
)
from zcu_tools.program.v2.ir.passes import (
    DeadLabelEliminationPass,
    DeadWriteEliminationLinear,
    UnrollSmallLoopPass,
)
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    _run_linear_passes,
    make_default_pipeline,
)


def _flat_inst_count(root: RootNode) -> int:
    """Count non-meta instructions after flattening BlockNodes and BasicBlockNodes."""
    count = 0
    stack: list[IRNode] = list(root.insts)
    while stack:
        node = stack.pop()
        if isinstance(node, BasicBlockNode):
            for inst in node.insts:
                if not isinstance(inst, (LabelInst, MetaInst)):
                    count += 1
            if node.branch is not None:
                count += 1
        elif isinstance(node, BlockNode):
            stack.extend(node.insts)
    return count


def _flatten_root(root: RootNode) -> list[Instruction]:
    """Flatten a RootNode into a flat instruction list via the linker."""
    from zcu_tools.program.v2.ir.linker import IRLinker
    linker = IRLinker()
    prog_list, labels, meta_infos, _ = linker.link(root)
    logical = linker.unlink(prog_list, labels, meta_infos)
    return logical


def _has_jump_table_blocks(root: RootNode) -> bool:
    """Check if the root contains BasicBlockNode(s) with fix_addr_size=True."""
    stack: list[IRNode] = list(root.insts)
    while stack:
        node = stack.pop()
        if isinstance(node, BasicBlockNode) and node.fix_addr_size:
            return True
        if isinstance(node, BlockNode):
            stack.extend(node.insts)
        elif isinstance(node, IRLoop):
            stack.extend(node.body.insts)
    return False


def _count_fixed_blocks(root: RootNode) -> int:
    """Count BasicBlockNodes with fix_addr_size=True."""
    count = 0
    stack: list[IRNode] = list(root.insts)
    while stack:
        node = stack.pop()
        if isinstance(node, BasicBlockNode) and node.fix_addr_size:
            count += 1
        if isinstance(node, BlockNode):
            stack.extend(node.insts)
        elif isinstance(node, IRLoop):
            stack.extend(node.body.insts)
    return count


# ---------------------------------------------------------------------------
# Dead write / dead label tests (unchanged from prior phases)
# ---------------------------------------------------------------------------


def test_dead_write_elimination_removes_overwritten_write():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"}),
                RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"}),
                NopInst(),
            ]),
        ]
    )

    _run_linear_passes([DeadWriteEliminationLinear()], root)

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2
    assert isinstance(bb.insts[0], RegWriteInst)
    assert cast(RegWriteInst, bb.insts[0]).extra_args["LIT"] == "#2"
    assert isinstance(bb.insts[1], NopInst)


def test_dead_write_elimination_keeps_write_before_read():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"}),
                TestInst(op="s1 - #1"),
                RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"}),
            ]),
        ]
    )

    _run_linear_passes([DeadWriteEliminationLinear()], root)

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 3
    assert [
        cast(RegWriteInst, item).extra_args.get("LIT")
        for item in bb.insts
        if isinstance(item, RegWriteInst)
    ] == [
        "#1",
        "#2",
    ]


def test_dead_label_elimination_removes_unreferenced_label():
    root = RootNode(
        insts=[
            BasicBlockNode(labels=[LabelInst(name=Label("dead"), can_remove=True)], insts=[NopInst()]),
        ]
    )

    out = DeadLabelEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 1
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.labels) == 0
    assert isinstance(bb.insts[0], NopInst)


def test_dead_label_elimination_keeps_referenced_label():
    label = Label("keep")
    root = RootNode(
        insts=[
            BasicBlockNode(labels=[LabelInst(name=label)], branch=JumpInst(label=label)),
        ]
    )

    out = DeadLabelEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 1
    bb = out.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.labels) == 1
    assert isinstance(bb.labels[0], LabelInst)
    assert isinstance(bb.branch, JumpInst)


def test_dead_label_elimination_keeps_pseudo_labels():
    """Pseudo labels (HERE, NEXT, PREV, SKIP) are assembler tokens, not IR labels."""
    for pseudo in ("HERE", "NEXT", "PREV", "SKIP"):
        root = RootNode(
            insts=[
                BasicBlockNode(labels=[LabelInst(name=Label(pseudo))], insts=[NopInst()]),
            ]
        )

        out = DeadLabelEliminationPass().process(
            root, PipeLineContext(config=PipeLineConfig())
        )

        assert len(out.insts) == 1, f"pseudo label {pseudo!r} was incorrectly removed"
        bb = out.insts[0]
        assert isinstance(bb, BasicBlockNode)
        assert len(bb.labels) == 1, f"pseudo label {pseudo!r} was incorrectly removed"
        assert isinstance(bb.labels[0], LabelInst)


# ---------------------------------------------------------------------------
# Unroll tests (Phase 8 — joint slack/budget formula)
# ---------------------------------------------------------------------------
#
# k formula recap:
#   slack    = scheduled_ticks - body_cost
#   k_timing = max_unroll_factor               if slack <= 0
#            = min(ceil(overhead/slack), cap)  otherwise
#   k_budget = pmem_budget // body_size        (if body_size > 0 and budget set)
#   k        = min(k_timing, k_budget)
#
# Default config: cost_default=1, cost_wmem=4, cost_dmem=4, cost_jump_flush=4,
#                 max_unroll_factor=8, pmem_budget=None.
# loop_overhead = 2*cost_default + cost_jump_flush = 6.


def test_unroll_full_expansion_when_n_le_k():
    """n=3 body=inc_ref #1: scheduled=1, body_cost=1, slack=0 → k_timing=8;
    no budget → k=8. n(=3) <= k → fully expand into BlockNode of 3 copies
    plus one loop-counter init write."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # Expansion: counter init + 3*(body + increment) = 1 + 3*2 = 7 insts.
    assert _flat_inst_count(out) == 7
    # Flatten all BasicBlockNode insts from the output
    flat_insts: list[Instruction] = []
    for node in out.insts:
        if isinstance(node, BlockNode):
            for child in node.insts:
                if isinstance(child, BasicBlockNode):
                    flat_insts.extend(child.insts)
        elif isinstance(node, BasicBlockNode):
            flat_insts.extend(node.insts)
    # [0]: REG_WR counter imm #0 (init)
    assert isinstance(flat_insts[0], RegWriteInst)
    assert cast(RegWriteInst, flat_insts[0]).src == "imm"
    # Remaining: alternating TimeInst(body) and RegWriteInst(increment)
    assert isinstance(flat_insts[1], TimeInst)
    assert isinstance(flat_insts[2], RegWriteInst)  # counter += 1
    assert isinstance(flat_insts[3], TimeInst)
    assert isinstance(flat_insts[4], RegWriteInst)
    assert isinstance(flat_insts[5], TimeInst)
    assert isinstance(flat_insts[6], RegWriteInst)


def test_unroll_full_expansion_keeps_counter_init_for_counter_dependent_body():
    """When body reads/increments loop counter, full expansion must still emit
    the counter init (`REG_WR counter imm #0`) before the first body copy."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r0",
                n=1,
                body=BlockNode(
                    insts=[BasicBlockNode(insts=[RegWriteInst(dst="r1", src="op", op="r0")])]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # Full expansion: counter init BB + body BB + increment BB = 3 BasicBlockNodes
    assert len(out.insts) == 3
    init_bb = out.insts[0]
    assert isinstance(init_bb, BasicBlockNode)
    assert len(init_bb.insts) > 0
    assert isinstance(init_bb.insts[0], RegWriteInst)
    init = cast(RegWriteInst, init_bb.insts[0])
    assert init.dst == "r0"
    assert init.src == "imm"
    assert init.lit == "#0"


def test_unroll_full_expansion_preserves_internal_label():
    """Internal labels are deepcopied per body copy with unique names.
    Full expansion also keeps one counter-init write."""
    Label.reset()
    inner = Label.make_new("inner")
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(
                    insts=[
                        BasicBlockNode(
                            labels=[LabelInst(name=inner)],
                            insts=[TimeInst(c_op="inc_ref", lit="#1")],
                        ),
                    ]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # init(1) + 3*(TimeInst + increment)(2 each) = 1 + 3*2 = 7
    # LabelInsts are not counted (they occupy no pmem).
    assert _flat_inst_count(out) == 7


def test_unroll_partial_unroll_produces_loop_plus_remainder():
    """body=inc_ref #1: scheduled=1, body_cost=1, slack=0 → k_timing=cap=8.
    pmem_budget=8, body_size=1 → k_budget=8 → k=8.
    n=20: iters=2, remainder=4 → BlockNode containing IRLoop(n=16) + 4 copies."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=20,
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    config = PipeLineConfig(pmem_budget=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    # Top level should contain a single BlockNode
    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    # IRLoop(n=16) + 4 remainder copies, each with body+increment = 2 insts → 1 + 4*2 = 9
    assert len(block.insts) == 9
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 16  # (20 // 8) * 8
    # Remainder: each copy is a BasicBlockNode
    assert all(isinstance(item, BasicBlockNode) for item in block.insts[1:])


def test_unroll_partial_unroll_no_remainder():
    """n=16, k=8 → IRLoop(n=16) wrapped in BlockNode, no remainder."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=16,
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    config = PipeLineConfig(pmem_budget=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert len(block.insts) == 1
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 16


def test_unroll_partial_unroll_loop_bound_uses_full_unrolled_iterations():
    """For n=99 and k=8, the generated partial-unroll loop must stop at 96,
    then execute 3 remainder copies inline."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r0",
                n=99,
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    config = PipeLineConfig(pmem_budget=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 96
    # 3 remainder copies, each with body + increment = 2 insts
    assert len(block.insts[1:]) == 6


def test_default_pipeline_can_disable_all_optimization_passes():
    """Disabling all pass flags should keep the IR layout unchanged."""
    Label.reset()
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"}),
                RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"}),
            ]),
            BasicBlockNode(labels=[LabelInst(name=Label.make_new("dead"))], insts=[
                TimeInst(c_op="inc_ref", lit="#0"),
                TimeInst(c_op="inc_ref", lit="#1"),
                TimeInst(c_op="inc_ref", lit="#2"),
            ]),
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=8192)
    pipeline.config.disable_all_opt = True

    out, _ctx = pipeline(root)

    assert len(out.insts) == 2
    bb0 = cast(BasicBlockNode, out.insts[0])
    assert isinstance(bb0.insts[0], RegWriteInst)
    assert isinstance(bb0.insts[1], RegWriteInst)
    bb1 = cast(BasicBlockNode, out.insts[1])
    assert len(bb1.labels) == 1
    assert isinstance(bb1.labels[0], LabelInst)
    assert isinstance(bb1.insts[0], TimeInst)
    assert cast(TimeInst, bb1.insts[0]).lit == "#0"
    assert isinstance(bb1.insts[1], TimeInst)
    assert cast(TimeInst, bb1.insts[1]).lit == "#1"
    assert isinstance(bb1.insts[2], TimeInst)
    assert cast(TimeInst, bb1.insts[2]).lit == "#2"


def test_unroll_no_scheduled_ticks_uses_zero_delay_budget():
    """Body has no inc_ref delay → scheduled_ticks=0, so k uses slack<=0 branch."""
    Label.reset()
    heavy_insts: list[IRNode] = [
        BasicBlockNode(insts=[PortWriteInst(dst="0", time="t0")]) for _ in range(5)
    ]
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

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], BlockNode)
    block = cast(BlockNode, out.insts[0])
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 96  # (100 // 8) * 8
    # remainder=4, body_size=5, each copy has body+increment → 4*(5+1)
    assert len(block.insts) == 1 + 4 * 6


def test_unroll_dynamic_delay_only_body_uses_zero_delay_budget():
    """Body with only dynamic inc_ref → scheduled=0, so k uses slack<=0 branch."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=20,
                body=BlockNode(
                    insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", r1="r_delay")])]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], BlockNode)
    block = cast(BlockNode, out.insts[0])
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 16  # (20 // 8) * 8
    # remainder=4, each copy has body+increment = 2 insts
    assert len(block.insts) == 1 + 4 * 2  # remainder=4


def test_unroll_mixed_literal_and_dynamic_delay_uses_literal_budget():
    """Mixed literal+dynamic inc_ref still treats the dynamic term as 0.
    Under the current large default jump cost, k hits the cap so n=4 fully
    expands."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=4,
                body=BlockNode(
                    insts=[BasicBlockNode(insts=[
                        TimeInst(c_op="inc_ref", lit="#5"),
                        TimeInst(c_op="inc_ref", r1="r_dyn"),
                    ])]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # counter init + 4 copies * (1 body_bb + 1 increment_bb) = 1 + 4*2 = 9 BasicBlockNodes
    # body_bb has 2 insts (TimeInst x2); total flat inst count = 1 + 4*(2+1) = 13
    assert _flat_inst_count(out) == 13


def test_unroll_exact_register_hint_fully_expands():
    """range_hint=(3,3) → treated as constant 3, fully expand into BlockNode
    with one counter-init write."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n="r_count",
                range_hint=(3, 3),
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # counter init + 3*(body + increment) = 1 + 3*2 = 7
    assert _flat_inst_count(out) == 7


def test_unroll_non_exact_register_hint_emits_jump_table():
    """range_hint=(2,5): non-exact register loop now goes through Phase 8D jump table.

    body=inc_ref #1 → scheduled=1, body_cost=1, slack=0 → k_timing=cap=8.
    body_size=1; floor_pow2(8) = 8 → 8 fixed BasicBlockNodes (one per entry).
    """
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n="r_count",
                range_hint=(2, 5),
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # Should produce BasicBlockNode sequence with fix_addr_size=True.
    # k=8 entry blocks + 2 back-edge blocks = 10 fixed blocks.
    assert _has_jump_table_blocks(out)
    assert _count_fixed_blocks(out) == 10


def test_unroll_no_hint_register_loop_emits_jump_table():
    """Register-driven loop with no hint also goes to jump-table dispatch."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n="r_count",
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert _has_jump_table_blocks(out)


def test_unroll_counter_sensitive_loop_still_uses_unroll_k_rules():
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(
                    insts=[BasicBlockNode(insts=[
                        TestInst(op="s0 - #1"),
                        TimeInst(c_op="inc_ref", lit="#1"),
                    ])]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # Full expansion: counter init BB + 3*(body BB + increment BB) = 1 + 3*2 = 7 BasicBlockNodes
    # body BB has 2 insts (TestInst + TimeInst); flat inst count = 1 + 3*(2+1) = 10
    assert _flat_inst_count(out) == 10


def test_unroll_partial_unroll_clones_labels_safely():
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
                        BasicBlockNode(
                            labels=[LabelInst(name=inner)],
                            branch=JumpInst(label=inner),
                        ),
                        BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")]),
                    ]
                ),
            )
        ]
    )

    config = PipeLineConfig(pmem_budget=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    emit = _flatten_root(out)

    defined = {str(inst.name) for inst in emit if isinstance(inst, LabelInst)}
    targets = {
        str(inst.label) for inst in emit if isinstance(inst, JumpInst) and inst.label
    }

    internal_targets = {t for t in targets if not t.startswith("loop")}
    assert internal_targets <= defined


def test_unroll_budget_caps_k_below_timing():
    """body_size=1, pmem_budget=3 → k_budget=3. k_timing would be 8 (cap).
    k=min(8,3)=3. n=9 → iters=3, remainder=0 → IRLoop(n=9)."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=9,
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    config = PipeLineConfig(pmem_budget=3)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 9


def test_unroll_max_factor_caps_k():
    """max_unroll_factor=2 caps k regardless of slack/budget. n=10 → iters=5
    and unrolled loop stop bound stays at 10."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=10,
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    config = PipeLineConfig(max_unroll_factor=2)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 10


def test_unroll_post_order_recurses_into_inner_loop_first():
    """Outer loop wraps an inner loop with n=2; both are constant. Inner unrolls
    first (post-order), then outer is evaluated against the rewritten body."""
    Label.reset()
    inner = IRLoop(
        name="inner",
        counter_reg="s1",
        n=2,
        body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
    )
    outer = IRLoop(
        name="outer",
        counter_reg="s0",
        n=2,
        body=BlockNode(insts=[inner]),
    )
    root = RootNode(insts=[outer])

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # Inner full expansion: init_inner + 2*(body + incr) = 1 + 4 = 5
    # Outer full expansion: init_outer + 2*(inner_block + incr_outer) = 1 + 2*6 = 13
    assert _flat_inst_count(out) == 13


def test_unroll_cpmg_style_body_triggers():
    """CPMG-style: 3 PortWrite + inc_ref #14.
    With the current large default jump cost, k reaches the cap 8.
    n=50 > k=8 → iters=6, remainder=2.
    body_size = 3 (PortWrite) + 1 (TimeInst) = 4.
    """
    Label.reset()
    body_insts: list[IRNode] = [
        BasicBlockNode(insts=[PortWriteInst(dst="0", time="t0")]),
        BasicBlockNode(insts=[PortWriteInst(dst="1", time="t0")]),
        BasicBlockNode(insts=[PortWriteInst(dst="2", time="t0")]),
        BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#14")]),
    ]
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=50,
                body=BlockNode(insts=body_insts),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 48  # (50 // 8) * 8
    # Remainder: 50 % 8 = 2, each copy: body(4 BasicBlockNodes) + increment(1) = 5 items.
    assert len(block.insts) == 1 + 2 * 5


# ---------------------------------------------------------------------------
# Phase 8E integration tests
# ---------------------------------------------------------------------------


def test_unroll_register_driven_k_forced_to_power_of_two():
    """With the current large default jump cost, k_raw reaches the cap 8 and
    remains 8 after floor_pow2()."""
    Label.reset()
    body_insts: list[IRNode] = [
        BasicBlockNode(insts=[PortWriteInst(dst="0", time="t0")]),
        BasicBlockNode(insts=[PortWriteInst(dst="1", time="t0")]),
        BasicBlockNode(insts=[PortWriteInst(dst="2", time="t0")]),
        BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#14")]),
    ]
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=body_insts),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # k=8, body has 4 blocks each → 8*4=32 entry blocks + 2 back-edge blocks = 34 fixed blocks.
    assert _has_jump_table_blocks(out)
    assert _count_fixed_blocks(out) == 34


def test_unroll_register_driven_jump_table_structure():
    """Verify the jump-table blocks have correct structure and conditional jumps."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
            )
        ]
    )

    config = PipeLineConfig(max_unroll_factor=2)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    # k=2: 2 entry blocks + 2 back-edge blocks = 4 fixed blocks.
    assert _count_fixed_blocks(out) == 4

    emit = _flatten_root(out)
    cond_jumps = [
        inst for inst in emit if isinstance(inst, JumpInst) and inst.if_cond is not None
    ]
    # n==0 guard: JUMP -if(Z) -op(n - #0)
    assert any(j.if_cond == "Z" and j.op == "r_n - #0" for j in cond_jumps)
    # back_edge exit: JUMP -if(NS) -op(i - n)
    assert any(j.if_cond == "NS" and j.op == "r_i - r_n" for j in cond_jumps)
    # prologue r==0 jump: JUMP -if(Z) -op(i - #0)
    assert any(j.if_cond == "Z" and j.op == "r_i - #0" for j in cond_jumps)


def test_unroll_register_driven_body_with_no_words_falls_back():
    """A body that flattens to body_words == 0 must fall back to no-unroll.

    Bare LabelInst contributes 0 to flat size and 0 to body_cost.
    """
    Label.reset()
    inner = Label.make_new("inner")
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=[BasicBlockNode(labels=[LabelInst(name=inner)])]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # Falls back: original IRLoop unchanged.
    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRLoop)


def test_unroll_register_driven_dispatch_too_long_falls_back():
    """If body_words requires a shift-add sequence longer than max_dispatch_words,
    register-driven unroll must fall back to no-unroll."""
    # stride = body_words + 1 = 0xFF requires 8 adds + 7 shifts = 15 ops.
    # Use 0xFD NOPs plus one TIME word => body_words = 0xFE => stride = 0xFF.
    Label.reset()
    body_insts: list[IRNode] = [BasicBlockNode(insts=[NopInst()]) for _ in range(0xFD)]
    body_insts.append(BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1000")]))
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=body_insts),
            )
        ]
    )

    config = PipeLineConfig(max_dispatch_words=4)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRLoop)


def test_unroll_nested_register_driven_inner_unrolls_first():
    """Post-order recursion: the inner register-driven loop is rewritten to
    jump-table blocks before the outer loop is evaluated.

    After inner is rewritten, its body blocks include branches (loop guard,
    back-edge), so the outer loop cannot be jump-table unrolled (branches in
    body disqualify it). The inner unroll is still confirmed via fix_addr_size.
    """
    Label.reset()
    inner = IRLoop(
        name="inner",
        counter_reg="r_j",
        n="r_m",
        body=BlockNode(insts=[BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit="#1")])]),
    )
    outer = IRLoop(
        name="outer",
        counter_reg="r_i",
        n="r_n",
        body=BlockNode(insts=[inner]),
    )
    root = RootNode(insts=[outer])

    config = PipeLineConfig(max_unroll_factor=2)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    # Inner loop is rewritten to jump-table blocks (fix_addr_size=True present).
    # Outer loop falls back to no-unroll because inner's lowered blocks contain
    # branches (guard, back-edge) which disqualify jump-table body merging.
    assert _has_jump_table_blocks(out)


def test_default_pipeline_structure():
    from zcu_tools.program.v2.ir.pipeline import IRPipeLine
    pipeline = make_default_pipeline(pmem_capacity=8192)

    assert isinstance(pipeline, IRPipeLine)
    # Shared linear passes: ZeroDCE + TimedMerge + DeadWrite
    assert isinstance(pipeline.linear_passes, list)
    assert [type(p).__name__ for p in pipeline.linear_passes] == [
        "ZeroDelayDCELinear",
        "TimedMergeLinear",
        "DeadWriteEliminationLinear",
    ]
    # IR passes
    assert [type(p).__name__ for p in pipeline.ir_passes] == [
        "UnrollSmallLoopPass",
        "DeadLabelEliminationPass",
        "BranchEliminationPass",
        "BlockMergePass",
    ]



# ---------------------------------------------------------------------------
# Phase 3: DeadWriteElimination on BasicBlockNode path
# ---------------------------------------------------------------------------

def test_dead_write_elimination_removes_overwritten_write_in_basic_block():
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[
                RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"}),
                RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"}),
            ]),
        ]
    )

    _run_linear_passes([DeadWriteEliminationLinear()], root)

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 1
    assert bb.insts[0].extra_args.get("LIT") == "#2"


def test_dead_write_elimination_nop_pads_fixed_basic_block():
    r1 = RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"})
    r2 = RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"})
    root = RootNode(
        insts=[
            BasicBlockNode(insts=[r1, r2], fix_addr_size=True),
        ]
    )

    _run_linear_passes([DeadWriteEliminationLinear()], root)

    bb = root.insts[0]
    assert isinstance(bb, BasicBlockNode)
    assert len(bb.insts) == 2  # stride preserved
    assert isinstance(bb.insts[0], NopInst)  # dead write replaced with NOP
    assert isinstance(bb.insts[1], RegWriteInst)
