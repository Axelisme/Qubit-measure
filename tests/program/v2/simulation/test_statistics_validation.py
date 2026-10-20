"""Borrowed compact statistics are validated by content and always detached."""

import re
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.core.simulation import (
    AcquireRequest,
    ChannelStatistics,
    ReadoutWindow,
    RoundStatistics,
    SamplingIdentity,
)
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2.simulation.signals import StatisticsCopier


@pytest.fixture
def compact_round():
    env = white_noise(std_adc=0, seed=17)
    request = AcquireRequest(
        "integrated",
        (1000,),
        0,
        (ReadoutWindow(0, 0, 100, 100.0, (0.0,)),),
        1,
        0,
        "validation",
        reps=1000,
    )
    record = RoundStatistics(
        np.array([400, 600], dtype=np.int64),
        (
            ChannelStatistics(
                0,
                np.array([[20.0, -30.0]]),
                np.array([[[1.0, 2.0], [3.0, 4.0]]]),
                np.eye(2),
            ),
        ),
        SamplingIdentity(17, (0, 0, 1)),
    )
    env.soc.binding.execute = lambda _: record
    env.soc.binding.begin()
    env.soc.install_request(request)
    try:
        yield env.soc, request, record
    finally:
        env.soc.release_request()
        env.soc.binding.end()
        env.devices.close_all_devices()


def assert_detached(actual: RoundStatistics, expected: RoundStatistics):
    pairs: list[
        tuple[
            NDArray[np.int64] | NDArray[np.float64],
            NDArray[np.int64] | NDArray[np.float64],
        ]
    ] = [(actual.state_counts, expected.state_counts)]
    for got, wanted in zip(actual.channels, expected.channels, strict=True):
        assert got.channel == wanted.channel
        pairs.extend(
            [
                (got.sums, wanted.sums),
                (got.conditional_sums, wanted.conditional_sums),
                (got.shot_covariance, wanted.shot_covariance),
            ]
        )
    for got, wanted in pairs:
        assert got.dtype == wanted.dtype
        assert got.shape == wanted.shape
        assert got.tobytes() == wanted.tobytes()
        assert not np.shares_memory(got, wanted)
    assert actual.raw_identity == expected.raw_identity


def mutate_layout(
    record: RoundStatistics, kind: str
) -> tuple[RoundStatistics, str | None]:
    ch = record.channels[0]
    error = None
    if kind == "negative_counts":
        record.state_counts[:] = [-1, 1001]
        error = "state_counts"
    elif kind == "wrong_total":
        record.state_counts[0] = 0
        error = "state_counts"
    elif kind == "counts_dtype":
        record = replace(record, state_counts=record.state_counts.astype(float))
        error = "state_counts"
    elif kind == "wrong_channel":
        record = replace(record, channels=(replace(ch, channel=1),))
        error = "channels"
    elif kind == "duplicate_channels":
        record = replace(record, channels=(ch, ch))
        error = "channels"
    elif kind == "wrong_shape":
        record = replace(record, channels=(replace(ch, sums=np.zeros((2, 2))),))
        error = "statistics shape"
    return record, error


@pytest.mark.parametrize(
    "kind",
    [
        "valid",
        "negative_counts",
        "wrong_total",
        "counts_dtype",
        "wrong_channel",
        "duplicate_channels",
        "wrong_shape",
        "nan_sums",
        "infinite_conditional",
        "conditional_overflow",
        "asymmetric_covariance",
        "negative_covariance",
        "dynamic_overflow",
        "changed_valid_conditional",
        "changed_valid_covariance",
        "valid_again",
    ],
)
def test_statistics_contract_after_successful_round(compact_round, kind):
    soc, _, record = compact_round
    first = soc.run_round()
    assert isinstance(first, RoundStatistics)
    assert_detached(first, record)
    ch = record.channels[0]
    changed, error = mutate_layout(record, kind)
    if kind == "nan_sums":
        ch.sums[0, 0] = np.nan
        error = "finite values"
    elif kind == "infinite_conditional":
        ch.conditional_sums[0, 0, 0] = np.inf
        error = "finite values"
    elif kind == "conditional_overflow":
        ch.conditional_sums[0, 0, 0] = 2.0**63
        error = "int64"
    elif kind == "asymmetric_covariance":
        ch.shot_covariance[0, 1] = 1.0
        error = "symmetric positive semidefinite"
    elif kind == "negative_covariance":
        ch.shot_covariance[0, 0] = -1.0
        error = "symmetric positive semidefinite"
    elif kind == "dynamic_overflow":
        ch.sums[0, 0] = 1000 * 2.0**63
        error = "int64"
    elif kind == "changed_valid_conditional":
        ch.conditional_sums[0, 0, 0] = 7.0
    elif kind == "changed_valid_covariance":
        ch.shot_covariance[:] = [[2.0, 0.3], [0.3, 1.0]]
    soc.binding.execute = lambda _: changed
    if error is not None:
        with pytest.raises(ValueError, match=error):
            soc.run_round()
    else:
        second = soc.run_round()
        assert isinstance(second, RoundStatistics)
        assert_detached(second, changed)
        assert not np.shares_memory(first.channels[0].sums, second.channels[0].sums)


def numeric_round() -> tuple[AcquireRequest, RoundStatistics]:
    request = AcquireRequest(
        "integrated",
        (2, 4),
        1,
        tuple(ReadoutWindow(ch, 0, 8, 100.0, (0.0,)) for ch in (3, 1)),
        17,
        0,
        "validation",
        reps=4,
    )
    first = ChannelStatistics(
        3,
        np.array([[[2.0, -6.0]], [[10.0, -14.0]]]),
        np.zeros((2, 1, 2, 2), dtype=np.float64),
        np.eye(2, dtype=np.float64),
    )
    record = RoundStatistics(
        np.array([[1, 3], [2, 2]], dtype=np.int64),
        (replace(first, channel=1), first),
        SamplingIdentity(17, (0, 0, 1)),
    )
    return request, record


def statistics_cases():
    request, record = numeric_round()
    first = record.channels[1]
    counts_error = "backend state_counts do not match request point layout/reps"
    shape_error = "invalid ADC 3 statistics shape or finite values"
    range_error = "ADC sums must be finite and representable as int64"
    covariance_error = "shot_covariance must be symmetric positive semidefinite"
    cases = [pytest.param(request, record, None, id="valid")]
    cases.append(
        pytest.param(
            replace(request, reps=0),
            replace(record, state_counts=np.zeros_like(record.state_counts)),
            range_error,
            id="zero-reps",
        )
    )
    for name, counts in (
        ("negative-count", np.array([[-1, 5], [2, 2]], dtype=np.int64)),
        ("wrong-total", record.state_counts + 1),
        ("float-count", record.state_counts.astype(np.float64)),
        ("wrong-count-shape", record.state_counts[..., None]),
        ("empty-states", record.state_counts[..., :0]),
    ):
        cases.append(
            pytest.param(
                request, replace(record, state_counts=counts), counts_error, id=name
            )
        )
    for name, value, error in (
        ("nan", np.nan, shape_error),
        ("inf", np.inf, shape_error),
        ("overflow", 2.0**63 * request.reps, range_error),
        ("negative-overflow", -(2.0**64) * request.reps, range_error),
        ("negative-endpoint", -(2.0**63) * request.reps, None),
        ("upper-safe", np.nextafter(2.0**63, 0) * request.reps, None),
        ("zero", 0.0, None),
    ):
        changed = replace(first, sums=np.full_like(first.sums, value))
        cases.append(
            pytest.param(
                request,
                replace(record, channels=(record.channels[0], changed)),
                error,
                id=f"sums-{name}",
            )
        )
    for name, channel, error in (
        ("sums-float32", replace(first, sums=first.sums.astype(np.float32)), None),
        ("sums-strided", replace(first, sums=first.sums[..., ::-1]), None),
        (
            "conditional-nan",
            replace(
                first, conditional_sums=np.full_like(first.conditional_sums, np.nan)
            ),
            shape_error,
        ),
        (
            "conditional-overflow",
            replace(
                first, conditional_sums=np.full_like(first.conditional_sums, 2.0**63)
            ),
            range_error,
        ),
        (
            "conditional-shape",
            replace(first, conditional_sums=first.conditional_sums[..., 0]),
            shape_error,
        ),
        (
            "covariance-negative",
            replace(first, shot_covariance=-np.eye(2)),
            covariance_error,
        ),
        (
            "covariance-nan",
            replace(first, shot_covariance=np.full((2, 2), np.nan)),
            shape_error,
        ),
        (
            "covariance-shape",
            replace(first, shot_covariance=np.ones((1, 1))),
            shape_error,
        ),
    ):
        cases.append(
            pytest.param(
                request,
                replace(record, channels=(record.channels[0], channel)),
                error,
                id=name,
            )
        )
    cases.append(
        pytest.param(
            request,
            replace(record, channels=record.channels[:1]),
            "backend ADC channels do not match request",
            id="missing-channel",
        )
    )
    return cases


@pytest.mark.parametrize("acquire_request, record, error", statistics_cases())
def test_copier_numeric_contract_and_detach(
    acquire_request: AcquireRequest, record: RoundStatistics, error: str | None
):
    copier = StatisticsCopier()
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(2):
            if error is not None:
                with pytest.raises(ValueError, match=re.escape(error)) as caught:
                    copier.copy(acquire_request, record)
                assert str(caught.value) == error
            else:
                copied = copier.copy(acquire_request, record)
                assert [channel.channel for channel in copied.channels] == [3, 1]
                ordered = replace(record, channels=tuple(reversed(record.channels)))
                assert_detached(copied, ordered)


@pytest.mark.parametrize("nonfinite_sums", [False, True])
def test_copier_preserves_finite_then_conditional_then_range_error_order(
    nonfinite_sums,
):
    request, record = numeric_round()
    first = record.channels[1]
    changed = replace(
        first,
        sums=np.full_like(first.sums, np.nan if nonfinite_sums else 2.0**65),
        shot_covariance=-np.eye(2),
    )
    error = (
        "invalid ADC 3 statistics shape or finite values"
        if nonfinite_sums
        else "shot_covariance must be symmetric positive semidefinite"
    )
    with pytest.raises(ValueError, match=re.escape(error)) as caught:
        StatisticsCopier().copy(
            request, replace(record, channels=(record.channels[0], changed))
        )
    assert str(caught.value) == error


def test_copier_checks_each_new_request_layout():
    request, record = numeric_round()
    copier = StatisticsCopier()
    reversed_request = replace(request, readouts=tuple(reversed(request.readouts)))
    more_reps = replace(request, reps=8, loop_dims=(2, 8))
    for active, borrowed in (
        (request, record),
        (reversed_request, record),
        (more_reps, replace(record, state_counts=record.state_counts * 2)),
        (request, record),
    ):
        copied = copier.copy(active, borrowed)
        by_channel = {ch.channel: ch for ch in borrowed.channels}
        expected = replace(
            borrowed,
            channels=tuple(by_channel[w.channel] for w in active.readouts),
        )
        assert_detached(copied, expected)
    with pytest.raises(ValueError, match="point layout/reps"):
        copier.copy(more_reps, record)


def test_copier_handles_readonly_strided_borrowed_arrays():
    request, record = numeric_round()
    counts = record.state_counts[:, ::-1]
    channels = tuple(replace(ch, sums=ch.sums[..., ::-1]) for ch in record.channels)
    borrowed = replace(record, state_counts=counts, channels=channels)
    for ch in borrowed.channels:
        ch.sums.setflags(write=False)
    counts.setflags(write=False)
    copied = StatisticsCopier().copy(request, borrowed)
    assert_detached(copied, replace(borrowed, channels=tuple(reversed(channels))))
    copied.state_counts[:] = 0
    copied.channels[0].sums[:] = 42
    assert counts.tobytes() == record.state_counts[:, ::-1].tobytes()
    assert borrowed.channels[1].sums.tobytes() != copied.channels[0].sums.tobytes()


@pytest.fixture
def eigen_calls(monkeypatch):
    calls = []
    original = np.linalg.eigvalsh

    def observe(covariance, *args, **kwargs):
        calls.append(covariance.copy())
        return original(covariance, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvalsh", observe)
    return calls


def test_equal_content_reuses_validation_across_rounds_and_requests(
    compact_round,
    eigen_calls,
):
    soc, request, record = compact_round
    soc.run_round()
    equal_content = deepcopy(record)
    soc.binding.execute = lambda _: equal_content
    soc.run_round()
    soc.release_request()
    soc.install_request(request)
    soc.run_round()
    assert len(eigen_calls) == 1
    # Dynamic values remain live on a static-certificate hit.
    equal_content.state_counts[:] = [500, 500]
    equal_content.channels[0].sums[:] = [[42.0, -21.0]]
    result = soc.run_round()
    assert isinstance(result, RoundStatistics)
    assert_detached(result, equal_content)
    assert len(eigen_calls) == 1


def test_certificate_is_not_shared_between_socs(compact_round, eigen_calls):
    soc, request, record = compact_round
    soc.run_round()
    other = white_noise(std_adc=0, seed=17)
    other.soc.binding.execute = lambda _: record
    other.soc.binding.begin()
    other.soc.install_request(request)
    try:
        other.soc.run_round()
    finally:
        other.soc.release_request()
        other.soc.binding.end()
        other.devices.close_all_devices()
    assert len(eigen_calls) == 2


def test_invalid_static_content_is_never_certified(compact_round, eigen_calls):
    soc, _, record = compact_round
    record.channels[0].shot_covariance[:] = -np.eye(2)
    for _ in range(2):
        with pytest.raises(ValueError, match="symmetric positive semidefinite"):
            soc.run_round()
    assert len(eigen_calls) == 2


def test_dtype_is_part_of_certificate(compact_round):
    soc, _, record = compact_round
    record.channels[0].conditional_sums[:] = -0.0
    soc.run_round()
    # Same shape and bytes, but uint64 interprets -0.0 as 2**63.
    changed = replace(
        record.channels[0],
        conditional_sums=record.channels[0].conditional_sums.view(np.uint64),
    )
    soc.binding.execute = lambda _: replace(record, channels=(changed,))
    with pytest.raises(ValueError, match="int64"):
        soc.run_round()


def test_certificate_residency_is_bounded(compact_round, eigen_calls):
    soc, _, record = compact_round
    counts = np.zeros(2048, dtype=np.int64)
    counts[0] = 1000
    conditional = np.zeros((1, 2048, 2), dtype=np.float64)
    large = replace(
        record,
        state_counts=counts,
        channels=(replace(record.channels[0], conditional_sums=conditional),),
    )
    soc.binding.execute = lambda _: large
    soc.run_round()
    soc.run_round()
    assert len(eigen_calls) == 1
    # Four distinct 32 KiB payloads cannot all fit in the SoC's 64 KiB budget.
    for value in (1.0, 2.0, 3.0):
        conditional[:] = value
        soc.run_round()
    conditional[:] = 0.0
    soc.run_round()
    assert len(eigen_calls) == 5
