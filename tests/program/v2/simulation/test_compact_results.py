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
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import IntegratedRound

from tests.program._acquisition_soc import AcquisitionSoc
from tests.program._profiles import load_soccfg
from tests.program._timeline import compile_modules
from tests.program.v2.simulation.support import FixedRawShots, raw_of, request_of


def make_program(env):
    return compile_modules(
        env.soccfg,
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
        reps=1,
    )


def statistics(request: RoundRequest) -> IntegratedRound:
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
    summary = RoundStatistics(
        counts,
        tuple(channels),
        SamplingIdentity(acquire.root_seed, (acquire.sequence, request.index, 1)),
    )
    return IntegratedRound(
        summary,
        FixedRawShots({ch.channel: ch.conditional_sums[..., 0, :] for ch in channels}),
    )


def test_summed_materializes_retained_shots_matching_hardware(monkeypatch):
    env = white_noise(
        load_soccfg(n_readouts=2),
        std_adc=0,
        seed=19,
        routes=tuple((ch, ch) for ch in range(2)),
    )
    prog = make_program(env)
    hardware = make_program(env)
    soc = AcquisitionSoc(env.soccfg.get_cfg())
    iq = np.arange(1, 25, dtype=np.float64).reshape(2, 3, 2, 2)
    raw = {0: (iq * 8).astype(np.int64), 1: ((iq + 24) * 8).astype(np.int64)}
    executed = []
    materialized = []
    received = []

    class Raw:
        def materialize(self, reps, avg_level):
            materialized.append((reps, avg_level))
            return {ch: value[None].copy() for ch, value in raw.items()}

    def execute(request):
        executed.append(request.index)
        # Compact IQ totals intentionally disagree with the retained shots.
        return replace(statistics(request), raw=Raw())

    def processor(values):
        received.append(values[1].copy())
        # One output combines both ADCs while retaining their different reads.
        return np.stack(
            [
                values[0][:, :, 0, 0],
                values[1][:, :, 0, 0],
                values[1][:, :, 1, 0] - values[1][:, :, 1, 1],
            ],
            axis=-1,
        ).sum(axis=1)

    env.soc.binding.execute = execute
    monkeypatch.setattr(
        soc,
        "poll_data",
        lambda: [(6, ([raw[ch].reshape(-1, 2) for ch in (1, 0)], {}))],
    )
    for ro in env.soccfg["readouts"]:
        ro["iq_offset"] = 7.0
    try:
        expected = hardware.acquire_summed(
            soc, processor=processor, progress=False, rounds=2
        )
        actual = prog.acquire_summed(
            env.soc, processor=processor, progress=False, rounds=2
        )
        assert expected is not None and actual is not None
        assert actual.shape == (2, 3, 3)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[..., 0], [[2, 10, 18], [26, 34, 42]])
        np.testing.assert_array_equal(
            actual[0], [[2, 50, -2], [10, 58, -2], [18, 66, -2]]
        )
        assert len(received) == 4
        for values in received:
            assert values.dtype == np.float64
            np.testing.assert_array_equal(values, (iq + 24).reshape(6, 1, 2, 2))
        for values, ch in zip(raw_of(prog), (1, 0), strict=True):
            assert values.dtype == np.int64
            np.testing.assert_array_equal(values, raw[ch][None])
        raw_of(prog)
        assert executed == [0, 1]
        assert materialized == [(1, 0), (1, 0)]
        # An ordinary call still uses the independent compact IQ summary.
        ordinary = prog.acquire(env.soc, progress=False, remove_offset=False)
        np.testing.assert_array_equal(ordinary[0][0, 0, 0], [10, -1])
        assert executed == [0, 1, 0]
        assert materialized == [(1, 0), (1, 0)]
    finally:
        env.devices.close_all_devices()


def test_averaged_statistics_preserve_layout_offsets_hooks_and_deferred_raw():
    env = white_noise(
        load_soccfg(n_readouts=2),
        std_adc=0,
        seed=19,
        routes=tuple((ch, ch) for ch in range(2)),
    )
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
    averaged = prog.acquire(
        env.soc,
        progress=False,
        round_hook=hook,
        rounds=2,
    )
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
    deferred = records[-1].raw.materialize(request.reps, request.avg_level)
    raw = raw_of(prog)
    assert raw[0].dtype == np.int64
    for ch, value in zip((1, 0), raw, strict=True):
        np.testing.assert_array_equal(value, deferred[ch])
    np.testing.assert_array_equal(raw_of(prog)[0], raw[0])
    for ch, value in zip((1, 0), raw, strict=True):
        np.testing.assert_array_equal(
            env.soc.get_accumulated(ch, 0, value.size // 2), value.reshape(-1, 2)
        )
    assert len(records) == 2  # program and SoC getters must not rerun the backend


@pytest.mark.parametrize("processing", ["threshold", "tracker"])
def test_shot_processing_uses_materialized_statistics(processing):
    env = white_noise(
        load_soccfg(n_readouts=2),
        std_adc=0,
        seed=29,
        routes=tuple((ch, ch) for ch in range(2)),
    )
    prog = make_program(
        env,
    )
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
    if processing == "threshold":
        result = prog.acquire(env.soc, progress=False, threshold=12.5)
    else:
        result = prog.acquire(env.soc, progress=False, trackers=[Tracker(), Tracker()])
    assert result is not None
    request = request_of(env)
    raw = raw_of(prog)
    expected_raw = records[0].raw.materialize(request.reps, request.avg_level)
    np.testing.assert_array_equal(raw[0], expected_raw[1])
    if processing == "threshold":
        np.testing.assert_array_equal(result[0][0, ..., 0], [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(result[0][..., 1], 0)
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
    env = white_noise(
        load_soccfg(n_readouts=2),
        std_adc=0,
        seed=None,
        routes=tuple((ch, ch) for ch in range(2)),
    )
    prog = make_program(
        env,
    )

    def execute(request):
        delivery = statistics(request)
        result = delivery.statistics
        ch = result.channels[0]
        if invalid == "missing":
            return replace(
                delivery, statistics=replace(result, channels=result.channels[:1])
            )
        if invalid == "extra":
            return replace(
                delivery,
                statistics=replace(
                    result, channels=(*result.channels, replace(ch, channel=2))
                ),
            )
        if invalid == "duplicate":
            return replace(
                delivery, statistics=replace(result, channels=(*result.channels, ch))
            )
        if invalid == "counts_shape":
            return replace(
                delivery,
                statistics=replace(
                    result, state_counts=np.full((6, 1), 4, dtype=np.int64)
                ),
            )
        if invalid == "counts_total":
            return replace(
                delivery,
                statistics=replace(result, state_counts=result.state_counts + 1),
            )
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
        return replace(
            delivery, statistics=replace(result, channels=(ch, *result.channels[1:]))
        )

    env.soc.binding.execute = execute
    with pytest.raises(
        ValueError, match="channels|state_counts|statistics|covariance|int64"
    ):
        prog.acquire(env.soc, progress=False)


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_binding_rejects_result_type_mismatch(mode, monkeypatch):
    env = white_noise(
        load_soccfg(n_readouts=2),
        std_adc=0,
        seed=None,
        routes=tuple((ch, ch) for ch in range(2)),
    )
    prog = make_program(
        env,
    )
    prog.acquire(env.soc, progress=False)
    request = replace(request_of(env), mode=mode)
    monkeypatch.setattr(
        env.soc.binding,
        "execute",
        statistics if mode == "decimated" else lambda request: RoundSignals(()),
    )
    with pytest.raises(ValueError, match="mode|result|Round"):
        env.soc.binding.sample_round(request, 0)
