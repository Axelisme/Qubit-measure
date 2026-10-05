from __future__ import annotations

import pytest
from zcu_tools.experiment.utils import make_sweep


@pytest.mark.parametrize(
    (
        "start",
        "stop",
        "expts",
        "step",
        "expected_stop",
        "expected_step",
        "expected_count",
    ),
    [
        pytest.param(1.0, 9.0, 5, None, 9.0, 2.0, 5, id="endpoints-count"),
        pytest.param(9.0, 1.0, 5, None, 1.0, -2.0, 5, id="descending-count"),
        pytest.param(0.0, 1.0, 4, None, 1.0, 1 / 3, 4, id="fractional-step"),
        pytest.param(1.0, 10.0, None, 2.0, 9.0, 2.0, 5, id="count-truncates-span"),
        pytest.param(9.0, 0.0, None, -2.0, 1.0, -2.0, 5, id="descending-span"),
        pytest.param(1.0, None, 5, 2.0, 9.0, 2.0, 5, id="count-step"),
        pytest.param(1.0, 100.0, 5, 2.0, 9.0, 2.0, 5, id="explicit-stop-ignored"),
        pytest.param(9.0, 100.0, 5, -2.0, 1.0, -2.0, 5, id="descending-stop-ignored"),
        pytest.param(3.0, None, 1, None, 3.0, 0.0, 1, id="single-count-only"),
        pytest.param(3.0, 3.0, 1, None, 3.0, 0.0, 1, id="single-endpoints"),
        pytest.param(3.0, None, 1, 0.0, 3.0, 0.0, 1, id="single-explicit-step"),
        pytest.param(3.0, 10.0, 1, 0.0, 3.0, 0.0, 1, id="single-stop-ignored"),
        pytest.param(3.0, 3.0, None, 0.0, 3.0, 0.0, 1, id="single-inferred-count"),
    ],
)
def test_make_sweep_preserves_inference_and_direction(
    start: float,
    stop: float | None,
    expts: int | None,
    step: float | None,
    expected_stop: float,
    expected_step: float,
    expected_count: int,
) -> None:
    sweep = make_sweep(start, stop, expts, step)

    assert sweep.start == start
    assert sweep.stop == pytest.approx(expected_stop)
    assert sweep.step == pytest.approx(expected_step)
    assert sweep.expts == expected_count


@pytest.mark.parametrize(
    (
        "start",
        "stop",
        "expts",
        "step",
        "expected_start",
        "expected_stop",
        "expected_step",
        "expected_count",
    ),
    [
        pytest.param(1.8, 9.8, 3, None, 1, 9, 4, 3, id="infer-before-truncation"),
        pytest.param(-1.8, None, 3, -2.9, -1, -5, -2, 3, id="truncate-toward-zero"),
        pytest.param(1.8, 9.9, None, 2.9, 1, 5, 2, 3, id="count-before-truncation"),
        pytest.param(1.8, None, 1, 0.9, 1, 1, 0, 1, id="single-truncated-step"),
        pytest.param(1.8, None, 1, None, 1, 1, 0, 1, id="single-truncated-start"),
    ],
)
def test_force_int_truncates_before_recomputing_stop(
    start: float,
    stop: float | None,
    expts: int | None,
    step: float | None,
    expected_start: int,
    expected_stop: int,
    expected_step: int,
    expected_count: int,
) -> None:
    sweep = make_sweep(start, stop, expts, step, force_int=True)

    assert sweep.start == expected_start
    assert sweep.stop == expected_stop
    assert sweep.step == expected_step
    assert sweep.expts == expected_count


@pytest.mark.parametrize(
    ("stop", "expts", "step", "force_int", "message"),
    [
        pytest.param(None, None, None, False, "Not enough information", id="no-inputs"),
        pytest.param(
            2.0, None, None, False, "Not enough information", id="missing-count-step"
        ),
        pytest.param(
            None, None, 1.0, False, "Not enough information", id="missing-stop-count"
        ),
        pytest.param(
            None, 2, None, False, "Not enough information", id="missing-stop-step"
        ),
        pytest.param(
            2.0, None, 0.0, False, "stop must equal start", id="zero-step-span"
        ),
        pytest.param(
            2.0, 1, None, False, "stop must equal start", id="single-inferred-span"
        ),
        pytest.param(None, 1, 1.0, False, "step must be 0", id="single-nonzero-step"),
        pytest.param(None, 1, 1e-13, False, "step must be 0", id="single-tiny-step"),
        pytest.param(
            None, 2, 0.0, False, "step must not be zero", id="multiple-zero-step"
        ),
        pytest.param(
            None, 0, 1.0, False, "expts must be greater than 0", id="zero-count"
        ),
        pytest.param(
            None, -1, 1.0, False, "expts must be greater than 0", id="negative-count"
        ),
        pytest.param(
            -1.0, None, 1.0, False, "expts must be greater than 0", id="wrong-direction"
        ),
        pytest.param(
            None, 3, 0.9, True, "step must not be zero", id="truncated-zero-step"
        ),
    ],
)
def test_invalid_sweeps_fail_explicitly(
    stop: float | None,
    expts: int | None,
    step: float | None,
    force_int: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_sweep(0.0, stop, expts, step, force_int=force_int)
