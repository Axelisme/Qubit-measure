"""Unit tests for IR passes."""

import pytest
from .nodes import (
    IRBranch,
    IRCondJump,
    IRDelay,
    IRDelayAuto,
    IRLabel,
    IRJump,
    IRLoop,
    IRNop,
    IRPulse,
    IRSeq,
)
from .passes import EstimateDurations, FreshLabels
from .pass_base import PassCtx, PassConfig


class TestFreshLabels:
    """Tests for FreshLabels pass."""

    def test_rename_single_label(self):
        pass_obj = FreshLabels()
        node = IRLabel(name="loop_start")
        result = pass_obj(node)
        assert isinstance(result, IRLabel)
        assert result.name.startswith("_label_")

    def test_rename_label_and_jump(self):
        pass_obj = FreshLabels()
        node = IRSeq(body=(
            IRLabel(name="start"),
            IRJump(target="start"),
        ))
        result = pass_obj(node)
        assert isinstance(result, IRSeq)
        label_node = result.body[0]
        jump_node = result.body[1]
        assert isinstance(label_node, IRLabel)
        assert isinstance(jump_node, IRJump)
        assert label_node.name == jump_node.target
        assert label_node.name.startswith("_label_")

    def test_multiple_labels(self):
        pass_obj = FreshLabels()
        node = IRSeq(body=(
            IRLabel(name="start"),
            IRLabel(name="end"),
        ))
        result = pass_obj(node)
        label1 = result.body[0]
        label2 = result.body[1]
        assert label1.name != label2.name

    def test_condjump_target_renamed(self):
        pass_obj = FreshLabels()
        node = IRSeq(body=(
            IRLabel(name="exit"),
            IRCondJump(target="exit", arg1="r0", test="Z"),
        ))
        result = pass_obj(node)
        label_node = result.body[0]
        condjump_node = result.body[1]
        assert label_node.name == condjump_node.target


class TestEstimateDurations:
    """Tests for EstimateDurations pass (new schema: t field, only IRDelay advances ref_t)."""

    def test_pulse_duration_zero(self):
        """IRPulse does not advance ref_t: duration 0."""
        pass_obj = EstimateDurations()
        node = IRPulse(ch="0", pulse_name="p", t=0.5)
        result = pass_obj(node)
        assert isinstance(result, IRPulse)
        assert result.meta.duration == pytest.approx(0.0)

    def test_delay_numeric(self):
        pass_obj = EstimateDurations()
        node = IRDelay(t=1.5)
        result = pass_obj(node)
        assert isinstance(result, IRDelay)
        assert result.meta.duration == pytest.approx(1.5)

    def test_delay_qickparam(self):
        from qick.asm_v2 import QickParam
        pass_obj = EstimateDurations()
        param = QickParam(start=1.0, spans={"x": 0.5})
        node = IRDelay(t=param)
        result = pass_obj(node)
        assert result.meta.duration is None

    def test_delay_auto_always_none(self):
        pass_obj = EstimateDurations()
        node = IRDelayAuto(t=0.0)
        result = pass_obj(node)
        assert isinstance(result, IRDelayAuto)
        assert result.meta.duration is None

    def test_seq_sum_delays(self):
        pass_obj = EstimateDurations()
        node = IRSeq(body=(
            IRDelay(t=1.0),
            IRDelay(t=2.0),
            IRDelay(t=1.5),
        ))
        result = pass_obj(node)
        assert result.meta.duration == pytest.approx(4.5)

    def test_seq_with_pulse_and_delay(self):
        """Pulse contributes 0; only Delay contributes."""
        pass_obj = EstimateDurations()
        node = IRSeq(body=(
            IRPulse(ch="0", pulse_name="p", t=0.5),
            IRDelay(t=2.0),
        ))
        result = pass_obj(node)
        assert result.meta.duration == pytest.approx(2.0)

    def test_seq_with_delay_auto_becomes_none(self):
        pass_obj = EstimateDurations()
        node = IRSeq(body=(
            IRDelay(t=1.0),
            IRDelayAuto(t=0.0),
            IRDelay(t=1.5),
        ))
        result = pass_obj(node)
        assert result.meta.duration is None

    def test_loop_duration(self):
        pass_obj = EstimateDurations()
        node = IRLoop(name="loop1", n=3, body=IRDelay(t=2.0))
        result = pass_obj(node)
        assert result.meta.duration == pytest.approx(6.0)

    def test_nested_loop(self):
        pass_obj = EstimateDurations()
        inner = IRLoop(name="inner", n=2, body=IRDelay(t=1.0))
        outer = IRLoop(name="outer", n=3, body=inner)
        result = pass_obj(outer)
        assert result.meta.duration == pytest.approx(6.0)

    def test_empty_seq(self):
        pass_obj = EstimateDurations()
        node = IRSeq(body=())
        result = pass_obj(node)
        assert result.meta.duration == pytest.approx(0.0)

    def test_branch_max_duration(self):
        pass_obj = EstimateDurations()
        node = IRBranch(
            compare_reg="r0",
            arms=(IRDelay(t=1.0), IRDelay(t=3.0)),
        )
        result = pass_obj(node)
        assert result.meta.duration == pytest.approx(3.0)


class TestPassIntegration:
    def test_fresh_labels_then_estimate_durations(self):
        ir = IRSeq(body=(
            IRLabel(name="loop_start"),
            IRLoop(
                name="main",
                n=2,
                body=IRSeq(body=(
                    IRDelay(t=1.0),
                    IRCondJump(target="loop_start", arg1="r0", test="Z"),
                )),
            ),
        ))

        fresh_pass = FreshLabels()
        ir_fresh = fresh_pass(ir)

        estimate_pass = EstimateDurations()
        ir_estimated = estimate_pass(ir_fresh)

        assert isinstance(ir_estimated, IRSeq)
        label_node = ir_estimated.body[0]
        loop_node = ir_estimated.body[1]
        assert isinstance(label_node, IRLabel)
        assert isinstance(loop_node, IRLoop)

        loop_body = loop_node.body
        assert isinstance(loop_body, IRSeq)
        condjump = loop_body.body[1]
        assert condjump.target == label_node.name

        # Loop: 2 * (1.0 + 0) = 2.0
        assert loop_node.meta.duration == pytest.approx(2.0)
        # Seq: label(0) + loop(2.0) = 2.0
        assert ir_estimated.meta.duration == pytest.approx(2.0)
