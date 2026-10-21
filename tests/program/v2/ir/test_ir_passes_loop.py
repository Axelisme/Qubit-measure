"""Loop unrolling, remainder jumps, and register-driven dispatch contracts."""

from __future__ import annotations

from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    NopInst,
    RegWriteInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRLoop,
    IRNode,
)
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    Register,
    SrcKeyword,
)
from zcu_tools.program.v2.ir.passes.loop import UnrollLoopPass
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext

from .pass_test_support import (
    flatten_root,
    register_loop,
    transform_only_child,
    walk_basic_blocks,
)


def _config(**kwargs) -> PipeLineConfig:
    return PipeLineConfig(**kwargs)


def _counter_update(reg: str) -> BasicBlockNode:
    """Standard counter increment for tests."""
    return BasicBlockNode(
        insts=[
            RegWriteInst(
                dst=Register(reg),
                src=SrcKeyword.OP,
                op=AluExpr(Register(reg), AluOp.ADD, Immediate(1)),
            )
        ]
    )


def test_unroll_full_expansion_keeps_counter_init_for_counter_dependent_body():
    """When body reads/increments loop counter, full expansion must still emit
    the counter init (`REG_WR counter imm #0`) before the first body copy."""
    root = BlockNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg=Register("r0"),
                n=1,
                body=BlockNode(
                    insts=[
                        BasicBlockNode(
                            insts=[
                                RegWriteInst(
                                    dst=Register("r1"),
                                    src=SrcKeyword.OP,
                                    op=AluExpr(Register("r0"), AluOp.NONE),
                                )
                            ]
                        ),
                        _counter_update("r0"),
                    ]
                ),
            )
        ]
    )

    out = transform_only_child(
        root,
        UnrollLoopPass(),
        PipeLineContext(config=_config(), pmem_budget=512),
    )

    # Full expansion: the IRLoop is replaced by a BlockNode(insts=[init_bb, ...])
    assert len(out.insts) == 1
    expanded = out.insts[0]
    assert isinstance(expanded, BlockNode)
    assert len(expanded.insts) == 3
    init_bb = expanded.insts[0]
    assert isinstance(init_bb, BasicBlockNode)
    assert len(init_bb.insts) > 0
    init = init_bb.insts[0]
    assert isinstance(init, RegWriteInst)
    assert init.dst.name == "r0"


def test_unroll_register_driven_jump_table_structure():
    """UnrollLoopPass emits an IRDispatch node (not pre-lowered stubs) for the
    dispatch table; stubs are produced later by the pipeline fallback."""
    from zcu_tools.program.v2.ir.node import IRDispatch

    root = BlockNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg=Register("r0"),
                n=Register("r1"),
                body=BlockNode(
                    insts=[
                        BasicBlockNode(
                            insts=[TimeInst(c_op="inc_ref", lit=Immediate(1))]
                        )
                    ]
                ),
            )
        ]
    )

    config = _config(max_unroll_factor=2)
    out = transform_only_child(
        root,
        UnrollLoopPass(),
        PipeLineContext(config=config, pmem_budget=512, available_regs={"r14"}),
    )

    # UnrollLoopPass output should contain an IRDispatch node (not yet lowered).
    def _collect_dispatch(node: IRNode) -> list[IRDispatch]:
        if isinstance(node, IRDispatch):
            return [node]
        if isinstance(node, BlockNode):
            result = []
            for child in node.insts:
                result.extend(_collect_dispatch(child))
            return result
        return []

    dispatch_nodes = _collect_dispatch(out)
    assert len(dispatch_nodes) == 1
    assert len(dispatch_nodes[0].target_labels) == 2

    emit = flatten_root(out)
    cond_jumps = [
        inst for inst in emit if isinstance(inst, JumpInst) and inst.if_cond is not None
    ]
    # n==0 guard: JUMP -if(Z) -op(n - #0)
    assert any(j.if_cond == "Z" for j in cond_jumps)


# ---------------------------------------------------------------------------
# UnrollLoopPass — _unroll_partial big-jump paths (8.2)
# ---------------------------------------------------------------------------


def _nop_loop(name: str, n: int, counter: str = "r0") -> IRLoop:
    """IRLoop with a single NopInst body (size=1, cost=1, scheduled_ticks=0)."""
    return IRLoop(
        name=name,
        counter_reg=Register(counter),
        n=n,
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )


def _empty_const_loop(name: str, n: int) -> IRLoop:
    return IRLoop(
        name=name,
        counter_reg=Register("r0"),
        n=n,
        body=BlockNode(insts=[]),
    )


def _ctx_partial(pmem_capacity: int, pmem_budget: int = 10) -> PipeLineContext:
    """Context that forces k_final=pmem_budget (body_size=1) via pmem_budget control.
    slack=-1 < 0 → k_timing=max_unroll_factor=32; k_budget=pmem_budget//1=pmem_budget.
    k_final = min(32, pmem_budget).
    """
    return PipeLineContext(
        config=PipeLineConfig(pmem_capacity=pmem_capacity),
        pmem_budget=pmem_budget,
    )


def _collect_reg_write_s15(root: BlockNode) -> list[RegWriteInst]:
    """Collect all RegWriteInst with dst=s15 from the flat basic-block tree."""
    result = []
    for bb in walk_basic_blocks(root):
        for inst in bb.insts:
            if isinstance(inst, RegWriteInst) and inst.dst == Register("s15"):
                result.append(inst)
    return result


def test_partial_unroll_suffixes_start_label_when_name_is_already_allocated():
    loop = _nop_loop("loop", n=100)
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=512, pmem_budget=10)
    ctx.allocated_names.add("loop_unrolled_start")

    out = transform_only_child(root, UnrollLoopPass(), ctx)
    label_names = [
        str(label.name) for bb in walk_basic_blocks(out) for label in bb.labels
    ]

    assert "loop_unrolled_start_0" in label_names
    assert "loop_unrolled_start" not in label_names


def test_partial_unroll_small_pmem_no_remainder_label_jump():
    """Partial unroll (no remainder), small pmem: back-edge uses label-mode JUMP."""
    # n=100, pmem_budget=10 → k=10, remainder=0, small pmem
    loop = _nop_loop("lp", n=100)
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=512, pmem_budget=10)
    out = transform_only_child(root, UnrollLoopPass(), ctx)

    s15_writes = _collect_reg_write_s15(out)
    assert len(s15_writes) == 0  # no s15 in small-pmem mode

    # All back-edge jump BBs have label-mode JumpInst
    back_jumps = [
        bb.branch
        for bb in walk_basic_blocks(out)
        if isinstance(getattr(bb, "branch", None), JumpInst)
        and bb.branch is not None
        and bb.branch.if_cond == "S"
    ]
    assert len(back_jumps) >= 1
    assert all(j.label is not None for j in back_jumps)


def test_partial_unroll_big_pmem_no_remainder_s15_jump():
    """Partial unroll (no remainder), big pmem: back-edge uses s15 indirect jump."""
    loop = _nop_loop("lp", n=100)
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=4096, pmem_budget=10)
    out = transform_only_child(root, UnrollLoopPass(), ctx)

    s15_writes = _collect_reg_write_s15(out)
    # back-edge writes s15 in big-pmem mode
    assert len(s15_writes) >= 1


def test_partial_unroll_with_remainder_small_pmem():
    """Partial unroll with remainder (n=103, k=10), small pmem: init_bb uses label JUMP."""
    loop = _nop_loop("lp", n=103)  # remainder = 3
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=512, pmem_budget=10)
    out = transform_only_child(root, UnrollLoopPass(), ctx)

    s15_writes = _collect_reg_write_s15(out)
    assert len(s15_writes) == 0  # no s15 in small-pmem mode


def test_partial_unroll_synthetic_labels_are_removable():
    loop = _nop_loop("lp", n=103)
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=512, pmem_budget=10)
    out = transform_only_child(root, UnrollLoopPass(), ctx)

    synthetic_labels = [
        label
        for bb in walk_basic_blocks(out)
        for label in bb.labels
        if str(label.name).startswith(
            ("lp_remainder_entry", "lp_unrolled_start", "lp_unrolled_end")
        )
    ]

    assert synthetic_labels
    assert all(label.can_remove for label in synthetic_labels)


def test_partial_unroll_empty_body_remainder_label_is_removable():
    loop = _empty_const_loop("lp", n=103)
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=512, pmem_budget=10)
    out = transform_only_child(root, UnrollLoopPass(), ctx)

    remainder_labels = [
        label
        for bb in walk_basic_blocks(out)
        for label in bb.labels
        if str(label.name).startswith("lp_remainder_entry")
    ]

    assert remainder_labels
    assert all(label.can_remove for label in remainder_labels)


def test_partial_unroll_with_remainder_big_pmem():
    """Partial unroll with remainder (n=103, k=10), big pmem: init_bb uses s15 jump."""
    loop = _nop_loop("lp", n=103)  # remainder = 3
    root = BlockNode(insts=[loop])
    ctx = _ctx_partial(pmem_capacity=4096, pmem_budget=10)
    out = transform_only_child(root, UnrollLoopPass(), ctx)

    s15_writes = _collect_reg_write_s15(out)
    # init_bb (for remainder skip) + back-edge both write s15
    assert len(s15_writes) >= 2


# ---------------------------------------------------------------------------
# UnrollLoopPass — _maybe_build_jump_table early-return paths (8.2)
# ---------------------------------------------------------------------------


def _reg_loop_empty(name: str) -> IRLoop:
    """Register-driven IRLoop with an empty body (body_size=0 → skip jump table)."""
    return IRLoop(
        name=name,
        counter_reg=Register("r0"),
        n=Register("n_reg"),
        body=BlockNode(insts=[]),
    )


def test_jump_table_body_size_zero_returns_none():
    """IRLoop with empty body → body_size=0 → _maybe_build_jump_table returns None (no transform)."""
    loop = _reg_loop_empty("lp")
    root = BlockNode(insts=[loop])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1024,
        available_regs={"r14"},
    )
    out = transform_only_child(root, UnrollLoopPass(), ctx)
    # No IRDispatch should appear — loop was not transformed
    from zcu_tools.program.v2.ir.node import IRDispatch

    dispatches = [n for n in _walk_all_nodes(out) if isinstance(n, IRDispatch)]
    assert len(dispatches) == 0


def test_jump_table_no_available_regs_returns_none():
    """No available_regs → _maybe_build_jump_table returns None (cannot allocate scratch)."""
    loop = register_loop("lp")
    root = BlockNode(insts=[loop])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1024,
        available_regs=set(),  # empty
    )
    out = transform_only_child(root, UnrollLoopPass(), ctx)
    from zcu_tools.program.v2.ir.node import IRDispatch

    dispatches = [n for n in _walk_all_nodes(out) if isinstance(n, IRDispatch)]
    assert len(dispatches) == 0


def test_jump_table_k_raw_le_1_returns_none():
    """k_raw <= 1 → skip jump table.  Force by using pmem_budget=1 (k_budget=1//1=1)."""
    loop = register_loop("lp")
    root = BlockNode(insts=[loop])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1,  # k_budget = 1//1 = 1 → k_raw=min(max_unroll,1)=1 → skip
        available_regs={"r14"},
    )
    out = transform_only_child(root, UnrollLoopPass(), ctx)
    from zcu_tools.program.v2.ir.node import IRDispatch

    dispatches = [n for n in _walk_all_nodes(out) if isinstance(n, IRDispatch)]
    assert len(dispatches) == 0


def test_jump_table_normal_path_produces_dispatch():
    """Normal register-driven path: k≥2 → BlockNode with IRDispatch inside."""
    loop = register_loop("lp")
    root = BlockNode(insts=[loop])
    ctx = PipeLineContext(
        config=PipeLineConfig(pmem_capacity=512),
        pmem_budget=1024,
        available_regs={"r14"},
    )
    out = transform_only_child(root, UnrollLoopPass(), ctx)
    from zcu_tools.program.v2.ir.node import IRDispatch

    dispatches = [n for n in _walk_all_nodes(out) if isinstance(n, IRDispatch)]
    assert len(dispatches) >= 1


def _walk_all_nodes(node):
    """Walk all IRNodes recursively."""
    yield node
    if isinstance(node, BlockNode):
        for child in node.insts:
            yield from _walk_all_nodes(child)
    elif isinstance(node, IRLoop):
        yield from _walk_all_nodes(node.body)
    elif isinstance(node, IRBranch):
        for case in node.cases:
            yield from _walk_all_nodes(case)
