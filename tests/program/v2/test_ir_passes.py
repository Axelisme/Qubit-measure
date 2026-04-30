"""Tests for IR optimization passes (Phase 3R)."""

from __future__ import annotations

import pytest
from zcu_tools.program.v2.ir.nodes import (
    IRBranch,
    IRCondJump,
    IRDelay,
    IRJump,
    IRLabel,
    IRLoop,
    IRNop,
    IRPulse,
    IRRegOp,
    IRSendReadoutConfig,
    IRSeq,
    RegOp,
)
from zcu_tools.program.v2.ir.pass_base import PassConfig
from zcu_tools.program.v2.ir.passes import make_default_pipeline


def test_unroll_short_loop() -> None:
    root = IRSeq(
        body=(
            IRLoop(name="l0", n=3, body=IRDelay(0.01)),
            IRDelay(0.2),
        )
    )
    new_root, _ = make_default_pipeline(PassConfig())(root)
    assert isinstance(new_root, IRSeq)
    assert len(new_root.body) == 1
    # Unrolled (3x0.01), flattened, then fused with trailing 0.2.
    assert isinstance(new_root.body[0], IRDelay)
    assert new_root.body[0].t == pytest.approx(0.23)


def test_skip_unroll_large_loop_count() -> None:
    root = IRSeq(body=(IRLoop(name="l0", n=32, body=IRDelay(0.01)),))
    config = PassConfig(max_unroll_iters=16)
    new_root, ctx = make_default_pipeline(config)(root)
    assert isinstance(new_root, IRSeq)
    assert isinstance(new_root.body[0], IRLoop)
    assert any("skip unroll loop 'l0'" in msg for msg in ctx.diagnostics)


def test_fuse_adjacent_delays_with_same_tag_only() -> None:
    root = IRSeq(
        body=(
            IRDelay(0.1, tag="a"),
            IRDelay(0.2, tag="a"),
            IRDelay(0.3, tag="b"),
        )
    )
    new_root, _ = make_default_pipeline(PassConfig())(root)
    assert isinstance(new_root, IRSeq)
    assert len(new_root.body) == 2
    assert isinstance(new_root.body[0], IRDelay)
    assert isinstance(new_root.body[1], IRDelay)
    assert new_root.body[0].t == pytest.approx(0.3)
    assert new_root.body[0].tag == "a"
    assert new_root.body[1].t == 0.3
    assert new_root.body[1].tag == "b"


def test_remove_zero_delays_before_fusion() -> None:
    root = IRSeq(
        body=(
            IRDelay(0.0),
            IRDelay(0.1),
            IRDelay(0.0),
            IRDelay(0.2),
        )
    )
    new_root, _ = make_default_pipeline(PassConfig())(root)
    assert isinstance(new_root, IRSeq)
    assert len(new_root.body) == 1
    assert isinstance(new_root.body[0], IRDelay)
    assert new_root.body[0].t == pytest.approx(0.3)


def test_unroll_skipped_when_counter_register_referenced() -> None:
    body = IRSeq(
        body=(
            IRRegOp(dst="l0", lhs="l0", op=RegOp.ADD, rhs=1),
            IRDelay(0.01),
        )
    )
    root = IRSeq(body=(IRLoop(name="l0", n=2, body=body),))
    new_root, ctx = make_default_pipeline(PassConfig(enable_fusion=False))(root)
    # Loop is preserved because body references the loop's counter register.
    assert isinstance(new_root, IRSeq)
    assert len(new_root.body) == 1
    loop = new_root.body[0]
    assert isinstance(loop, IRLoop)
    assert loop.name == "l0"
    assert any(
        "skip unroll loop 'l0': body references counter register" in msg
        for msg in ctx.diagnostics
    )


def test_branch_arms_kept_unaligned_after_pipeline() -> None:
    root = IRBranch(
        compare_reg="sel",
        arms=(
            IRSeq(body=(IRDelay(0.1),)),
            IRSeq(body=(IRDelay(0.1), IRDelay(0.2))),
        ),
    )
    new_root, _ = make_default_pipeline(
        PassConfig(enable_fusion=False)
    )(root)
    assert isinstance(new_root, IRBranch)
    # AlignBranchDispatch has been removed; branch arms keep original shape.
    # FlattenSeq unwraps single-element IRSeq arms to their sole child.
    assert isinstance(new_root.arms[0], IRDelay)
    assert isinstance(new_root.arms[1], IRSeq)
    assert len(new_root.arms[1].body) == 2


def test_reorder_pulse_like_nodes_by_t_within_segment() -> None:
    root = IRSeq(
        body=(
            IRPulse(ch="0", pulse_name="p_late", t=0.3),
            IRSendReadoutConfig(ch="0", pulse_name="cfg_early", t=0.1),
            IRPulse(ch="0", pulse_name="p_mid", t=0.2),
        )
    )
    new_root, _ = make_default_pipeline(PassConfig())(root)
    assert isinstance(new_root, IRSeq)
    assert isinstance(new_root.body[0], IRSendReadoutConfig)
    assert isinstance(new_root.body[1], IRPulse)
    assert isinstance(new_root.body[2], IRPulse)
    assert new_root.body[1].pulse_name == "p_mid"
    assert new_root.body[2].pulse_name == "p_late"


def test_reorder_pulse_like_nodes_does_not_cross_barrier() -> None:
    root = IRSeq(
        body=(
            IRPulse(ch="0", pulse_name="p_late", t=0.3),
            IRCondJump(target="L0", arg1="r0", test="S", op="-", arg2=1),
            IRSendReadoutConfig(ch="0", pulse_name="cfg_early", t=0.1),
        )
    )
    new_root, _ = make_default_pipeline(PassConfig())(root)
    assert isinstance(new_root, IRSeq)
    assert isinstance(new_root.body[0], IRPulse)
    assert isinstance(new_root.body[1], IRCondJump)
    assert isinstance(new_root.body[2], IRSendReadoutConfig)


def test_validate_invariants_reports_undefined_label_target() -> None:
    root = IRSeq(body=(IRJump(target="L_missing"),))
    _, ctx = make_default_pipeline(PassConfig())(root)
    assert any("undefined jump target label:" in msg for msg in ctx.diagnostics)


def test_validate_invariants_accepts_defined_label_target() -> None:
    root = IRSeq(body=(IRJump(target="L0"), IRLabel(name="L0")))
    _, ctx = make_default_pipeline(PassConfig())(root)
    assert all("undefined jump target label" not in msg for msg in ctx.diagnostics)
