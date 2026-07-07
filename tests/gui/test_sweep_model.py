from __future__ import annotations

import pytest
from zcu_tools.gui.app.main import (
    CenteredSweepEditor,
    CenteredSweepValue,
    EvalValue,
    SweepEditor,
    SweepValue,
)


def test_canonicalize_derives_step_from_numeric_bounds() -> None:
    value = SweepEditor.canonicalize(SweepValue(0.0, 1.0, 5, step=999.0))

    assert value.step == pytest.approx(0.25)


def test_update_step_derives_expts_and_canonical_step() -> None:
    value = SweepEditor.update_step(SweepValue(0.0, 1.0, 11, step=0.1), 0.2)

    assert value.expts == 6
    assert value.step == pytest.approx(0.2)


def test_update_expts_derives_step() -> None:
    value = SweepEditor.update_expts(SweepValue(0.0, 1.0, 11, step=0.1), 5)

    assert value.step == pytest.approx(0.25)


def test_update_step_preserves_unresolved_expression_axis() -> None:
    value = SweepValue(
        start=EvalValue(expr="missing", resolved=None),
        stop=1.0,
        expts=11,
        step=0.1,
    )

    assert SweepEditor.update_step(value, 0.2) == value


def test_sweep_value_rejects_zero_expts() -> None:
    with pytest.raises(ValueError, match="expts"):
        SweepValue(0.0, 1.0, 0)


def test_centered_canonicalize_derives_step_from_span() -> None:
    value = CenteredSweepEditor.canonicalize(
        CenteredSweepValue(center=0.0, span=10.0, expts=5, step=999.0)
    )

    assert value.step == pytest.approx(2.5)


def test_centered_update_step_derives_expts_and_canonical_step() -> None:
    value = CenteredSweepEditor.update_step(
        CenteredSweepValue(center=0.0, span=10.0, expts=11, step=1.0), 2.0
    )

    assert value.expts == 6
    assert value.step == pytest.approx(2.0)


def test_centered_update_span_preserves_center() -> None:
    value = CenteredSweepEditor.update_span(
        CenteredSweepValue(center=EvalValue("r_f"), span=10.0, expts=11), 20.0
    )

    assert isinstance(value.center, EvalValue)
    assert value.center.expr == "r_f"
    assert value.span == pytest.approx(20.0)
    assert value.step == pytest.approx(2.0)


def test_centered_sweep_value_rejects_negative_span() -> None:
    with pytest.raises(ValueError, match="span"):
        CenteredSweepValue(0.0, -1.0, 11)
