"""Compact rounds preserve QICK layouts and defer shot creation until needed."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import (
    ChannelStatistics,
    RoundRequest,
    RoundSignals,
    RoundStatistics,
    SamplingIdentity,
)
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program.v2.simulation.support import raw_of, request_of


def make_program(env, *, rounds=2):
    return ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=1, rounds=rounds),
        modules=[
            PulseReadoutCfg(
                gen_ch=ch,
                ro_ch=ch,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=env.soccfg.cycles2us(8, ro_ch=ch),
            ).build(f"readout_{ch}_{read}")
            for read in range(2)
            for ch in (1, 0)
        ],
        sweep=[("outer", 2), ("inner", 3)],
    )


def statistics(request: RoundRequest) -> RoundStatistics:
    acquire = request.acquire
    point_dims = tuple(
        n for axis, n in enumerate(acquire.loop_dims) if axis != acquire.avg_level
    )
    counts = np.full((*point_dims, 1), acquire.reps, dtype=np.int64)
    channels = []
    # Backend order differs from declaration order. Every point and read differs.
    for window in reversed(acquire.readouts):
        means = np.empty((*point_dims, window.reads, 1, 2))
        for point in np.ndindex(point_dims):
            for read in range(window.reads):
                means[(*point, read, 0)] = [
                    8
                    * (
                        10 * window.channel
                        + 3 * point[0]
                        + point[1]
                        + read
                        + request.index
                    ),
                    -8 * (read + 1),
                ]
        channels.append(
            ChannelStatistics(
                window.channel,
                means[..., 0, :] * acquire.reps,
                means,
                np.zeros((2, 2)),
            )
        )
    return RoundStatistics(
        counts,
        tuple(channels),
        SamplingIdentity(acquire.root_seed, (acquire.sequence, request.index, 1)),
    )


def test_averaged_statistics_preserve_layout_offsets_hooks_and_deferred_raw():
    env = white_noise(std_adc=0, n_readouts=2, seed=19)
    prog = make_program(env)
    records = []
    hooks = []

    def execute(request):
        result = statistics(request)
        records.append(result)
        return result

    def hook(count, data, stop):
        # acc_buf is QICK's public raw buffer; averaged calls must not allocate it.
        assert prog.acc_buf is None
        hooks.append((count, [d.copy() for d in data]))

    env.soc.binding.execute = execute
    for ro in prog.soccfg["readouts"]:
        ro["iq_offset"] = 0.25
    averaged = prog.acquire(env.soc, progress=False, round_hook=hook)
    assert averaged is not None
    expected = np.array(
        [
            [
                [[10.25, -1.25], [11.25, -1.25], [12.25, -1.25]],
                [[13.25, -1.25], [14.25, -1.25], [15.25, -1.25]],
            ],
            [
                [[11.25, -2.25], [12.25, -2.25], [13.25, -2.25]],
                [[14.25, -2.25], [15.25, -2.25], [16.25, -2.25]],
            ],
        ]
    )
    np.testing.assert_array_equal(averaged[0], expected)
    np.testing.assert_array_equal(averaged[1][..., 0], expected[..., 0] - 10)
    assert [count for count, _ in hooks] == [1, 2]
    np.testing.assert_array_equal(hooks[0][1][0][..., 0], expected[..., 0] - 0.5)
    np.testing.assert_array_equal(hooks[1][1][0], averaged[0])
    request = request_of(env)
    deferred = records[-1].materialize_raw(request.reps, request.avg_level)
    raw = raw_of(prog)
    assert raw[0].dtype == np.int64
    for ch, value in zip((1, 0), raw, strict=True):
        np.testing.assert_array_equal(value, deferred[ch])
    np.testing.assert_array_equal(raw_of(prog)[0], raw[0])
    assert len(records) == 2  # getters must not rerun the backend


@pytest.mark.parametrize("processing", ["threshold", "classification", "tracker"])
def test_shot_processing_uses_materialized_statistics(processing):
    env = white_noise(std_adc=0, n_readouts=2, seed=29)
    prog = make_program(env, rounds=1)
    records = []
    tracked = []

    class Tracker:
        def update(self, points):
            tracked.append(points.copy())

    def execute(request):
        result = statistics(request)
        records.append(result)
        return result

    env.soc.binding.execute = execute
    kwargs = {}
    if processing == "threshold":
        kwargs = {"threshold": 12.5}
    elif processing == "classification":
        kwargs = {"g_center": 10 - 1j, "e_center": 11 - 1j, "ge_radius": 0.1}
    else:
        kwargs = {"trackers": [Tracker(), Tracker()]}
    result = prog.acquire(env.soc, progress=False, **kwargs)
    assert result is not None
    request = request_of(env)
    raw = raw_of(prog)
    expected_raw = records[0].materialize_raw(request.reps, request.avg_level)
    np.testing.assert_array_equal(raw[0], expected_raw[1])
    if processing == "threshold":
        np.testing.assert_array_equal(result[0][0, ..., 0], [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(result[0][..., 1], 0)
    elif processing == "classification":
        np.testing.assert_array_equal(result[0][0, 0], [[1, 0], [0, 1], [0, 0]])
    else:
        assert len(tracked) == 2
        expected = np.moveaxis(expected_raw[1], [-2, request.avg_level], [0, -2]) / 8
        np.testing.assert_array_equal(tracked[0], expected)
        np.testing.assert_array_equal(result[0][0, 0, 0], [10, -1])


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "extra",
        "duplicate",
        "sums_shape",
        "conditional_shape",
        "counts_shape",
        "counts_total",
        "nonfinite",
        "covariance",
        "overflow",
    ],
)
def test_invalid_statistics_raise_at_acquire_without_materializing(invalid):
    env = white_noise(std_adc=0, n_readouts=2)
    prog = make_program(env, rounds=1)

    def execute(request):
        result = statistics(request)
        ch = result.channels[0]
        if invalid == "missing":
            return replace(result, channels=result.channels[:1])
        if invalid == "extra":
            return replace(result, channels=(*result.channels, replace(ch, channel=2)))
        if invalid == "duplicate":
            return replace(result, channels=(*result.channels, ch))
        if invalid == "counts_shape":
            return replace(result, state_counts=np.full((6, 1), 4, dtype=np.int64))
        if invalid == "counts_total":
            return replace(result, state_counts=result.state_counts + 1)
        if invalid == "sums_shape":
            ch = replace(ch, sums=ch.sums[..., 0])
        elif invalid == "conditional_shape":
            ch = replace(ch, conditional_sums=ch.conditional_sums[..., :0, :])
        elif invalid == "nonfinite":
            ch = replace(ch, sums=np.full_like(ch.sums, np.nan))
        elif invalid == "covariance":
            ch = replace(ch, shot_covariance=-np.eye(2))
        elif invalid == "overflow":
            ch = replace(
                ch,
                conditional_sums=np.full_like(ch.conditional_sums, 2.0**63),
                sums=np.full_like(ch.sums, 4 * 2.0**63),
            )
        return replace(result, channels=(ch, *result.channels[1:]))

    env.soc.binding.execute = execute
    with pytest.raises(
        ValueError, match="channels|state_counts|statistics|covariance|int64"
    ):
        prog.acquire(env.soc, progress=False)


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_binding_rejects_result_type_mismatch(mode, monkeypatch):
    env = white_noise(std_adc=0, n_readouts=2)
    prog = make_program(env, rounds=1)
    prog.acquire(env.soc, progress=False)
    request = replace(request_of(env), mode=mode)
    monkeypatch.setattr(
        env.soc.binding,
        "execute",
        statistics if mode == "decimated" else lambda request: RoundSignals(()),
    )
    with pytest.raises(ValueError, match="mode|result|Round"):
        env.soc.binding.sample_round(request, 0)
