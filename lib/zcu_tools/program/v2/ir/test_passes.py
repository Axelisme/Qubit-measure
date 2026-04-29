"""Unit tests for IR passes."""

import pytest
from .nodes import (
    IRMeta, IRLabel, IRJump, IRCondJump, IRSeq, IRLoop, IRDelay, IRPulse, IRNop
)
from .passes import FreshLabels, EstimateDurations
from .pass_base import PassCtx, PassConfig


class TestFreshLabels:
    """Tests for FreshLabels pass."""

    def test_rename_single_label(self):
        """FreshLabels should rename a single label."""
        pass_obj = FreshLabels()
        node = IRLabel(name="loop_start")
        result = pass_obj(node)
        assert isinstance(result, IRLabel)
        assert result.name.startswith("_label_")

    def test_rename_label_and_jump(self):
        """FreshLabels should rename label and update jump target."""
        pass_obj = FreshLabels()
        # Build: label "start" followed by jump to "start"
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
        # Label and jump target should have same fresh name
        assert label_node.name == jump_node.target
        assert label_node.name.startswith("_label_")

    def test_multiple_labels(self):
        """FreshLabels should assign unique names to different labels."""
        pass_obj = FreshLabels()
        node = IRSeq(body=(
            IRLabel(name="start"),
            IRLabel(name="end"),
        ))
        result = pass_obj(node)
        label1 = result.body[0]
        label2 = result.body[1]
        assert label1.name != label2.name
        assert label1.name.startswith("_label_")
        assert label2.name.startswith("_label_")

    def test_condjump_target_renamed(self):
        """FreshLabels should rename conditional jump targets."""
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
    """Tests for EstimateDurations pass."""

    def test_delay_numeric_duration(self):
        """EstimateDurations should set duration for numeric delays."""
        pass_obj = EstimateDurations()
        node = IRDelay(duration=1.5)
        result = pass_obj(node)
        assert isinstance(result, IRDelay)
        assert result.meta.duration == 1.5

    def test_delay_string_duration(self):
        """EstimateDurations should leave duration None for QickParam expressions."""
        pass_obj = EstimateDurations()
        node = IRDelay(duration="2 * some_param")
        result = pass_obj(node)
        assert isinstance(result, IRDelay)
        assert result.meta.duration is None

    def test_seq_sum_durations(self):
        """IRSeq duration should be sum of child durations."""
        pass_obj = EstimateDurations()
        node = IRSeq(body=(
            IRDelay(duration=1.0),
            IRDelay(duration=2.0),
            IRDelay(duration=1.5),
        ))
        result = pass_obj(node)
        assert isinstance(result, IRSeq)
        # Sum: 1.0 + 2.0 + 1.5 = 4.5
        assert result.meta.duration == pytest.approx(4.5)

    def test_seq_with_none_duration(self):
        """IRSeq with any None child duration should have None duration."""
        pass_obj = EstimateDurations()
        node = IRSeq(body=(
            IRDelay(duration=1.0),
            IRDelay(duration="param"),  # None
            IRDelay(duration=1.5),
        ))
        result = pass_obj(node)
        assert isinstance(result, IRSeq)
        assert result.meta.duration is None

    def test_loop_duration(self):
        """IRLoop duration should be n * body_duration."""
        pass_obj = EstimateDurations()
        node = IRLoop(
            name="loop1",
            n=3,
            body=IRDelay(duration=2.0)
        )
        result = pass_obj(node)
        assert isinstance(result, IRLoop)
        # Duration: 3 * 2.0 = 6.0
        assert result.meta.duration == pytest.approx(6.0)

    def test_nested_loop(self):
        """Nested loops should have durations computed correctly."""
        pass_obj = EstimateDurations()
        inner = IRLoop(name="inner", n=2, body=IRDelay(duration=1.0))
        outer = IRLoop(name="outer", n=3, body=inner)
        result = pass_obj(outer)
        assert isinstance(result, IRLoop)
        # Outer: 3 * (2 * 1.0) = 6.0
        assert result.meta.duration == pytest.approx(6.0)

    def test_pulse_duration(self):
        """IRPulse duration should include pre and post delays."""
        pass_obj = EstimateDurations()
        node = IRPulse(
            ch="ch0", pulse_name="pulse1", pre_delay=0.5, advance=0.8
        )
        result = pass_obj(node)
        assert isinstance(result, IRPulse)
        # Duration: 0.5 + 0.3 = 0.8 (actual pulse length not in IR)
        assert result.meta.duration == pytest.approx(0.8)

    def test_empty_seq(self):
        """Empty IRSeq should have zero duration."""
        pass_obj = EstimateDurations()
        node = IRSeq(body=())
        result = pass_obj(node)
        assert isinstance(result, IRSeq)
        assert result.meta.duration == pytest.approx(0.0)


class TestPassIntegration:
    """Integration tests for pass pipeline."""

    def test_fresh_labels_then_estimate_durations(self):
        """Test running two passes in sequence."""
        # Build a simple IR with label and loop
        ir = IRSeq(body=(
            IRLabel(name="loop_start"),
            IRLoop(
                name="main",
                n=2,
                body=IRSeq(body=(
                    IRDelay(duration=1.0),
                    IRCondJump(target="loop_start", arg1="r0", test="Z"),
                ))
            ),
        ))

        # Run FreshLabels
        fresh_pass = FreshLabels()
        ir_fresh = fresh_pass(ir)

        # Run EstimateDurations
        estimate_pass = EstimateDurations()
        ir_estimated = estimate_pass(ir_fresh)

        # Verify structure
        assert isinstance(ir_estimated, IRSeq)
        label_node = ir_estimated.body[0]
        loop_node = ir_estimated.body[1]
        assert isinstance(label_node, IRLabel)
        assert isinstance(loop_node, IRLoop)

        # Verify fresh label is used in jump
        loop_body = loop_node.body
        assert isinstance(loop_body, IRSeq)
        condjump = loop_body.body[1]
        assert condjump.target == label_node.name

        # Verify durations
        # Loop: 2 * (1.0 + 0) = 2.0
        assert loop_node.meta.duration == pytest.approx(2.0)
        # Seq: label (0) + loop (2.0) = 2.0
        assert ir_estimated.meta.duration == pytest.approx(2.0)
