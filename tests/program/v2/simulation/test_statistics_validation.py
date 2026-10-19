"""Borrowed compact statistics are validated by content and always detached."""

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
