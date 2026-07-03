"""Tests for ir/passes/control_flow/unreachable.py: UnreachableEliminationPass."""

from __future__ import annotations

from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
)
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode
from zcu_tools.program.v2.ir.operands import AluExpr, AluOp, Immediate, Register
from zcu_tools.program.v2.ir.passes.control_flow.unreachable import (
    UnreachableEliminationPass,
)
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext


def _label(name: str) -> Label:
    return Label(name)


def _lref(name: str) -> LabelRef:
    return LabelRef(Label(name))


def _ctx():
    return PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)


def _run(root: BlockNode) -> tuple[BlockNode, bool]:
    parser = IRParser()
    chunks = parser.unparse(root)
    result_chunks, changed = UnreachableEliminationPass().process(chunks, _ctx())
    return parser.parse(result_chunks), changed


def _nop_block(labels=None, branch=None):
    return BasicBlockNode(labels=labels or [], insts=[NopInst()], branch=branch)


def _unconditional_jump(target: str) -> JumpInst:
    return JumpInst(label=_lref(target))


def _conditional_jump(target: str) -> JumpInst:
    return JumpInst(
        label=_lref(target),
        if_cond="Z",
        op=AluExpr(lhs=Register("r0"), op=AluOp.SUB, rhs=Immediate(0)),
    )


# ---------------------------------------------------------------------------
# Basic: dead block after unconditional jump is removed
# ---------------------------------------------------------------------------


def test_unreachable_dead_block_removed():
    exit_lbl = _label("exit")
    root = BlockNode(
        insts=[
            BasicBlockNode(branch=_unconditional_jump("exit")),
            _nop_block(),  # dead: no labels
            BasicBlockNode(labels=[LabelInst(name=exit_lbl)]),
        ]
    )
    out, changed = _run(root)

    assert changed is True
    blocks = [n for n in out.insts if isinstance(n, BasicBlockNode)]
    for bb in blocks:
        assert bb.insts != [NopInst()] or bb.labels  # dead NopInst block gone


def test_unreachable_labelled_block_after_jump_survives():
    exit_lbl = _label("exit")
    root = BlockNode(
        insts=[
            BasicBlockNode(branch=_unconditional_jump("exit")),
            BasicBlockNode(labels=[LabelInst(name=exit_lbl)], insts=[NopInst()]),
        ]
    )
    out, changed = _run(root)

    assert changed is False
    # The labelled block must still be present
    blocks = [n for n in out.insts if isinstance(n, BasicBlockNode)]
    labelled = [bb for bb in blocks if bb.labels]
    assert any(bb.labels[0].name == exit_lbl for bb in labelled)


def test_unreachable_labelled_disable_opt_fallthrough_clears_dead_mode():
    entry_lbl = _label("entry")
    exit_lbl = _label("exit")
    root = BlockNode(
        insts=[
            BasicBlockNode(branch=_unconditional_jump("exit")),
            BasicBlockNode(
                labels=[LabelInst(name=entry_lbl)],
                insts=[NopInst()],
                disable_opt=True,
            ),
            _nop_block(),
            BasicBlockNode(labels=[LabelInst(name=exit_lbl)]),
        ]
    )
    out, changed = _run(root)

    assert changed is False
    blocks = [n for n in out.insts if isinstance(n, BasicBlockNode)]
    assert len(blocks) == 4
    assert blocks[1].disable_opt is True
    assert blocks[1].labels[0].name == entry_lbl


def test_unreachable_labelled_block_with_unconditional_branch_reenters_dead_mode():
    entry_lbl = _label("entry")
    exit_lbl = _label("exit")
    root = BlockNode(
        insts=[
            BasicBlockNode(branch=_unconditional_jump("entry")),
            BasicBlockNode(
                labels=[LabelInst(name=entry_lbl)],
                branch=_unconditional_jump("exit"),
            ),
            _nop_block(),
            BasicBlockNode(labels=[LabelInst(name=exit_lbl)]),
        ]
    )
    out, changed = _run(root)

    assert changed is True
    blocks = [n for n in out.insts if isinstance(n, BasicBlockNode)]
    assert len(blocks) == 3
    assert blocks[1].labels[0].name == entry_lbl
    assert all(not (bb.insts == [NopInst()] and not bb.labels) for bb in blocks)


# ---------------------------------------------------------------------------
# MetaInst preserved in dead region
# ---------------------------------------------------------------------------


def test_unreachable_meta_inst_preserved_in_dead_region():
    exit_lbl = _label("end")
    meta = MetaInst(type="LOOP_BODY_END", name="lp", info={})
    parser = IRParser()
    chunks_before = parser.unparse(
        BlockNode(
            insts=[
                BasicBlockNode(branch=_unconditional_jump("end")),
                _nop_block(),  # dead normal block
                BasicBlockNode(labels=[LabelInst(name=exit_lbl)]),
            ]
        )
    )
    # Insert meta between jump and dead block (at chunk level)
    insert_pos = 1
    chunks_before.insert(insert_pos, meta)
    result_chunks, changed = UnreachableEliminationPass().process(chunks_before, _ctx())

    assert changed is True
    assert meta in result_chunks


# ---------------------------------------------------------------------------
# Conditional jump does NOT trigger dead mode
# ---------------------------------------------------------------------------


def test_unreachable_conditional_jump_keeps_following_block():
    target_lbl = _label("tgt")
    root = BlockNode(
        insts=[
            BasicBlockNode(branch=_conditional_jump("tgt")),
            _nop_block(),  # NOT dead — conditional jump
            BasicBlockNode(labels=[LabelInst(name=target_lbl)]),
        ]
    )
    out, changed = _run(root)

    assert changed is False
    # All three blocks survive
    blocks = [n for n in out.insts if isinstance(n, BasicBlockNode)]
    assert len(blocks) == 3


# ---------------------------------------------------------------------------
# Empty root — no crash
# ---------------------------------------------------------------------------


def test_unreachable_empty_block_list_no_crash():
    root = BlockNode(insts=[])
    out, changed = _run(root)
    assert changed is False
    assert isinstance(out, BlockNode)
