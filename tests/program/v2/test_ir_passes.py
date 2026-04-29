"""Tests for IR optimization passes (Phase 3R)."""

from __future__ import annotations

import pytest
from zcu_tools.program.v2.ir.nodes import IRBranch, IRDelay, IRLoop, IRNop, IRSeq
from zcu_tools.program.v2.ir.pass_base import PassConfig
from zcu_tools.program.v2.ir.passes import make_default_pipeline


def test_unroll_short_loop() -> None:
    root = IRSeq(
        body=(
            IRLoop(name="l0", n=3, body=IRDelay(0.01)),
            IRDelay(0.2),
        )
    )
    new_root, _ = make_default_pipeline(PassConfig(min_body_us=0.05))(root)
    assert isinstance(new_root, IRSeq)
    assert len(new_root.body) == 1
    # Unrolled (3x0.01), flattened, then fused with trailing 0.2.
    assert isinstance(new_root.body[0], IRDelay)
    assert new_root.body[0].t == pytest.approx(0.23)


def test_skip_unroll_large_loop_count() -> None:
    root = IRSeq(body=(IRLoop(name="l0", n=32, body=IRDelay(0.01)),))
    config = PassConfig(min_body_us=0.05, extra={"max_unroll_iters": 16})
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
    new_root, _ = make_default_pipeline(PassConfig(min_body_us=0.0))(root)
    assert isinstance(new_root, IRSeq)
    assert len(new_root.body) == 2
    assert isinstance(new_root.body[0], IRDelay)
    assert isinstance(new_root.body[1], IRDelay)
    assert new_root.body[0].t == pytest.approx(0.3)
    assert new_root.body[0].tag == "a"
    assert new_root.body[1].t == 0.3
    assert new_root.body[1].tag == "b"


def test_align_branch_dispatch_pads_short_arms() -> None:
    root = IRBranch(
        compare_reg="sel",
        arms=(
            IRSeq(body=(IRDelay(0.1),)),
            IRSeq(body=(IRDelay(0.1), IRDelay(0.2))),
        ),
    )
    new_root, _ = make_default_pipeline(
        PassConfig(min_body_us=0.0, enable_fusion=False)
    )(root)
    assert isinstance(new_root, IRBranch)
    assert isinstance(new_root.arms[0], IRSeq)
    assert isinstance(new_root.arms[1], IRSeq)
    # First arm was padded to match second arm cost.
    assert isinstance(new_root.arms[0].body[-1], IRNop)
    assert len(new_root.arms[0].body) == len(new_root.arms[1].body)
