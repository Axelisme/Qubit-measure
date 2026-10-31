"""Explicit environments exercise the real QICK acquisition/packing pipeline."""

import importlib
import importlib.util
from threading import Event

import numpy as np
import pytest
from zcu_tools.cfg_model import LinearClassifier
from zcu_tools.core.simulation import (
    ChannelStatistics,
    RoundStatistics,
    SamplingIdentity,
)
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2.builder import TimelineBuilder
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.read_axes import IQ, Classify, shot_sums, split_sums
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.program.v2.timeline import TimelineProgramV2
from zcu_tools.simulate.evolution import Cancelled


def environment(**kwargs):
    name = "zcu_tools.experiment.simulation"
    assert importlib.util.find_spec(name) is not None, (
        "explicit environment API required"
    )
    return importlib.import_module(name).white_noise(**kwargs)


def program(env, *, reps=2, channels=(0,), samples=16, reads=1, sweep=None):
    return ModularProgramV2(
        env.soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=ch,
                ro_ch=ch,
                nqz=1,
                freq=100.0,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=env.soccfg.cycles2us(samples, ro_ch=ch),
            ).build(f"readout_{ch}_{read}")
            for read in range(reads)
            for ch in channels
        ],
        reps=reps,
        sweep=sweep,
    )


def test_builder_program_acquires_and_sums_tagged_readouts():
    env = environment(std_adc=0, seed=123)
    try:
        builder = TimelineBuilder(
            env.soccfg, reps=3, initial_delay=0, relax_delay=0.1
        )
        builder.add_loop("point", 2)
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=env.soccfg.cycles2us(16, ro_ch=0),
        ).build("readout")
        builder.emit(readout, tag="herald")
        builder.emit(readout, tag="signal")
        prog = builder.build()

        averaged = prog.acquire(env.soc, progress=False)
        assert averaged[0].shape == (2, 2, 2)
        np.testing.assert_array_equal(averaged[0], 0)
        result = prog.acquire_summed(
            env.soc,
            processor=shot_sums(prog, [IQ("herald"), IQ("signal")]),
            rounds=2,
            progress=False,
        )
        expected = np.zeros((2, 3, 3))
        expected[:, 0, 0] = 6
        np.testing.assert_array_equal(result, expected)
        request = env.soc.binding.last_request
        assert request is not None
        assert request.point_names == ("point",)
        assert request.reps == 3
        (window,) = request.materialize_point(0).readouts
        assert window.samples == 16
        assert len(window.trigger_us) == 2
    finally:
        env.devices.close_all_devices()


def test_summed_simulation_materializes_joint_shots_and_retains_completed_rounds():
    env = environment(std_adc=0, seed=123, n_gens=2, n_readouts=2)
    prog = program(env, reps=3, channels=(1, 0), sweep=[("point", 2)])
    inputs = []
    hooks = []

    def processor(iq):
        inputs.append(iq)
        return np.full((2, 2, 3), 3.0, dtype=np.float64)

    def hook(count, sums, stop):
        hooks.append(sums.copy())
        if count == 2:
            stop.set()

    result = prog.acquire_summed(
        env.soc, processor=processor, rounds=3, round_hook=hook, progress=False
    )
    np.testing.assert_array_equal(result, np.full((2, 2, 3), 6.0))
    np.testing.assert_array_equal(hooks[0], np.full((2, 2, 3), 3.0))
    assert len(inputs) == 2
    assert list(inputs[0]) == [1, 0]
    for values in inputs[0].values():
        assert values.shape == (2, 3, 1, 2)
        assert values.dtype == np.float64
        np.testing.assert_array_equal(values, 0)
    stopped = Event()
    stopped.set()
    with pytest.raises(StoppedPartialAcquireError):
        prog.acquire_summed(
            env.soc, processor=processor, cancel_flag=stopped, progress=False
        )


@pytest.mark.parametrize("completed", [False, True])
def test_summed_simulation_discards_interrupted_round(completed):
    env = environment(std_adc=0, seed=123)
    prog = program(env)
    execute = env.soc.binding.execute
    stop = Event()
    hooks = []

    def source(request):
        if request.index == int(completed):
            stop.set()
            raise Cancelled()
        return execute(request)

    env.soc.binding.execute = source
    try:

        def acquire():
            return prog.acquire_summed(
                env.soc,
                processor=lambda iq: np.full((1, 3), 2.0),
                rounds=3,
                progress=False,
                cancel_flag=stop,
                round_hook=lambda count, data, flag: hooks.append(data.copy()),
            )

        if completed:
            np.testing.assert_array_equal(acquire(), [2, 2, 2])
            np.testing.assert_array_equal(hooks, [[2, 2, 2]])
        else:
            with pytest.raises(StoppedPartialAcquireError):
                acquire()
            assert hooks == []
    finally:
        env.devices.close_all_devices()


def raw_of(prog):
    raw = prog.get_raw()
    if raw is None:
        raise AssertionError("completed acquisition must retain raw buffers")
    return raw


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_native_timeline_acquire_uses_compiled_effects(mode):
    env = environment(std_adc=0, seed=123, routes=((0, 0),))

    class NativeProgram(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0, "probe", style="const", freq=100, phase=30, gain=0.25, length=0.1
            )
            self.declare_readout(0, length=env.soccfg.cycles2us(16, ro_ch=0))
            self.add_readoutconfig(0, "ro", freq=100, gen_ch=0)

        def _body(self, cfg):
            self.send_readoutconfig(0, "ro", t=0)
            self.pulse(0, "probe", t=0, tag="native-probe")
            self.trigger([0], t=0)

    try:
        prog = NativeProgram(env.soccfg, reps=2, initial_delay=0, relax_delay=0.1)
        if mode == "integrated":
            result = prog.acquire(env.soc, progress=False)
            assert result[0].shape == (1, 2)
        else:
            result = prog.acquire_decimated(env.soc, progress=False)
            assert result[0].shape == (2, 16, 2)
        np.testing.assert_array_equal(result[0], 0)
        raw = raw_of(prog)[0]
        assert raw.shape == (2, 1, 2)
        assert raw.dtype == np.int64
        np.testing.assert_array_equal(raw, 0)
        request = env.soc.binding.last_request
        assert request is not None
        assert request.mode == mode
        assert request.reps == 2
        assert request.sequence == 0
        assert len(env.soc.binding.rounds) == 1
        point = request.materialize_point(0)
        (event,) = point.events
        assert event.source == "native-probe"
        assert event.frequency_mhz == pytest.approx(100, abs=1e-5)
        assert event.gain == pytest.approx(0.25, abs=1e-4)
        (readout,) = point.readouts
        assert readout.channel == 0
        assert readout.generator_channels == (0,)
        assert readout.samples == 16
        assert readout.frequency_mhz == pytest.approx((100,), abs=1e-5)
    finally:
        env.devices.close_all_devices()


def test_explicit_zero_noise_preserves_shapes_and_raw():
    env = environment(std_adc=0, seed=123)
    prog = program(
        env,
    )
    result = prog.acquire(
        env.soc,
        progress=False,
        rounds=2,
    )
    assert result is not None
    assert result[0].shape == (1, 2)
    np.testing.assert_array_equal(result[0], 0)
    raw = prog.get_raw()
    assert raw is not None
    assert raw[0].shape == (2, 1, 2)
    assert raw[0].dtype == np.int64
    dec = prog.acquire_decimated(
        env.soc,
        progress=False,
        rounds=2,
    )
    assert dec is not None
    assert dec[0].shape == (2, 16, 2)
    np.testing.assert_array_equal(dec[0], 0)


@pytest.mark.parametrize("edge_counting", [False, True])
def test_integrated_normalization_round_buffers_and_delayed_raw(edge_counting):
    from zcu_tools.core.simulation import (
        ChannelStatistics,
        RoundStatistics,
        SamplingIdentity,
    )
    from zcu_tools.program.v2.simulation import IntegratedRound

    from tests.program.v2.simulation.support import FixedRawShots

    env = environment(std_adc=0, seed=13, n_gens=2, n_readouts=2)
    for ch in (0, 1):
        env.soccfg["readouts"][ch]["iq_offset"] = 0.125 * (ch + 1)
    prog = program(env, reps=3, channels=(1, 0), samples=7)
    for ro in prog.ro_chs.values():
        ro["edge_counting"] = edge_counting
    sums = np.array([[13.0, -29.0]])

    def source(request):
        statistics = RoundStatistics(
            np.array([3], dtype=np.int64),
            tuple(
                ChannelStatistics(
                    ch, sums * (ch + 1), np.zeros((1, 1, 2)), np.zeros((2, 2))
                )
                for ch in (0, 1)
            ),
            SamplingIdentity(13, (request.acquire.sequence, request.index, 1)),
        )
        return IntegratedRound(
            statistics, FixedRawShots({ch: sums * (ch + 1) / 3 for ch in (0, 1)})
        )

    env.soc.binding.execute = source
    hooks = []
    try:
        for remove_offset in (True, False, True):
            hooks.clear()
            actual = prog.acquire(
                env.soc,
                progress=False,
                remove_offset=remove_offset,
                round_hook=lambda count, data, stop: hooks.append((count, data)),
                rounds=3,
            )
            expected = []
            for ch in (1, 0):
                avg = sums * (ch + 1) / 3
                if not edge_counting:
                    avg /= 7
                    if remove_offset:
                        avg -= 0.125 * (ch + 1)
                expected.append(avg)
            rounds = prog.get_rounds()
            assert rounds is not None and len(rounds) == 3
            for completed in rounds:
                for got, wanted in zip(completed, expected, strict=True):
                    assert got.tobytes() == wanted.tobytes()
            # QICK summarizes by a NumPy mean over the completed rounds.
            for got, wanted in zip(actual, expected, strict=True):
                reference = np.stack([wanted] * 3).mean(axis=0)
                assert got.tobytes() == reference.tobytes()
            assert [count for count, _ in hooks] == [1, 2, 3]
            raw = raw_of(prog)
            assert all(value.shape == (3, 1, 2) for value in raw)
            for first, second in zip(raw, raw_of(prog), strict=True):
                assert first.tobytes() == second.tobytes()
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("cancel_round", [0, 1])
def test_mid_round_stop_discards_chunks_and_preserves_completed_rounds(cancel_round):
    from zcu_tools.simulate.evolution import Cancelled

    env = environment(std_adc=0, seed=13)
    prog = program(
        env,
    )
    stop = Event()
    completed_chunks = []
    execute = env.soc.binding.execute

    def source(request):
        cancel = getattr(request, "cancel", None)
        for chunk in range(5):
            if request.index == cancel_round and chunk == 2:
                stop.set()
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            completed_chunks.append((request.index, chunk))
        return execute(request)

    env.soc.binding.execute = source
    try:
        if cancel_round == 0:
            with pytest.raises(StoppedPartialAcquireError):
                prog.acquire(
                    env.soc,
                    progress=False,
                    cancel_flag=stop,
                    rounds=3,
                )
            assert prog.get_raw() is None
        else:
            result = prog.acquire(
                env.soc,
                progress=False,
                cancel_flag=stop,
                rounds=3,
            )
            np.testing.assert_array_equal(result[0], 0)
            rounds = prog.get_rounds()
            assert rounds is not None and len(rounds) == 1
            np.testing.assert_array_equal(raw_of(prog)[0], 0)
        assert completed_chunks == [
            *((0, chunk) for chunk in range(5) if cancel_round == 1),
            (cancel_round, 0),
            (cancel_round, 1),
        ]
        # Cancellation releases admission and does not leak into the next acquire.
        env.soc.binding.execute = execute
        prog.acquire(
            env.soc,
            progress=False,
            rounds=3,
        )
        resumed = prog.get_rounds()
        assert resumed is not None and len(resumed) == 3
    finally:
        env.devices.close_all_devices()


def assert_conditional_sums(value, completed):
    """Check counts and weighted IQ, including the completed zero-count point."""
    expected = (
        [[[1, 2, -2], [2, 180, -180]], [[0, 0, 0], [3, 270, -270]]]
        if completed == 1
        else [[[3, 24, -24], [3, 270, -270]], [[0, 0, 0], [6, 540, -540]]]
    )
    np.testing.assert_array_equal(value, expected)
    with np.errstate(all="raise"):
        counts, means = split_sums(value)
    np.testing.assert_array_equal(counts[:, 0], [1 if completed == 1 else 3, 0])
    assert means[0, 0] == (2 - 2j if completed == 1 else 8 - 8j)
    assert np.isnan(means[1, 0].real) and np.isnan(means[1, 0].imag)


@pytest.fixture
def conditional_environment():
    env = environment(std_adc=0, seed=13)
    try:
        yield env
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("cancel_at", [None, 0, 1, "hook"])
def test_summed_conditional_hooks_means_and_cancellation(
    cancel_at, conditional_environment
):
    env = conditional_environment
    prog = program(env, reps=3, reads=2, sweep=[("point", 2)])
    length = prog.ro_chs[0]["length"]
    literals = np.array(
        [
            [
                [[[-1, 0], [2, -2]], [[1, 0], [90, -90]]],
                [[[1, 0], [90, -90]], [[1, 0], [90, -90]]],
                [[[1, 0], [90, -90]], [[1, 0], [90, -90]]],
            ],
            [
                [[[-1, 0], [8, -8]], [[1, 0], [90, -90]]],
                [[[-1, 0], [14, -14]], [[1, 0], [90, -90]]],
                [[[1, 0], [90, -90]], [[1, 0], [90, -90]]],
            ],
        ],
        dtype=np.int64,
    )
    stop = Event()
    executed = []
    materialized = []
    hooks = []
    requests = []
    processor = shot_sums(
        prog,
        [
            Classify(
                "readout_0_0", classifier=LinearClassifier(angle=0.0, threshold=0.0)
            ),
            IQ("readout_0_1"),
        ],
    )

    def execute(request):
        executed.append(request.index)
        requests.append(request.acquire)
        if request.index == cancel_at:
            stop.set()
            raise Cancelled()

        class Raw:
            def materialize(self, reps, avg_level):
                materialized.append(request.index)
                return {0: literals[request.index].copy() * length}

        summary = RoundStatistics(
            np.full((2, 1), 3, dtype=np.int64),
            (
                ChannelStatistics(
                    0, np.zeros((2, 2, 2)), np.zeros((2, 2, 1, 2)), np.zeros((2, 2))
                ),
            ),
            SamplingIdentity(13, (request.acquire.sequence, request.index, 1)),
        )
        return IntegratedRound(summary, Raw())

    def hook(count, data, flag):
        hooks.append((count, data.copy()))
        with pytest.raises(RuntimeError, match="active|reentr"):
            prog.acquire_summed(env.soc, processor=processor, progress=False)
        if cancel_at == "hook":
            flag.set()

    env.soc.binding.execute = execute
    if cancel_at == 0:
        with pytest.raises(StoppedPartialAcquireError):
            prog.acquire_summed(
                env.soc,
                processor=processor,
                progress=False,
                rounds=2,
                cancel_flag=stop,
                round_hook=hook,
            )
        assert hooks == []
        assert prog.get_raw() is None
        assert materialized == []
        assert executed == [0]
    else:
        result = prog.acquire_summed(
            env.soc,
            processor=processor,
            progress=False,
            rounds=2,
            cancel_flag=stop,
            round_hook=hook,
        )
        assert result is not None
        completed = 2 if cancel_at is None else 1
        assert [count for count, _ in hooks] == list(range(1, completed + 1))
        for count, values in hooks:
            assert_conditional_sums(values, count)
        assert_conditional_sums(result, completed)
        np.testing.assert_array_equal(raw_of(prog)[0], literals[completed - 1] * length)
        assert materialized == list(range(completed))
        assert executed == ([0] if cancel_at == "hook" else [0, 1])
        assert all(request is requests[0] for request in requests)
    # Stop/failure and reentry must release admission for the next call.
    stop.clear()
    cancel_at = "hook"
    retried = prog.acquire_summed(env.soc, processor=processor, progress=False)
    assert retried is not None
    assert_conditional_sums(retried, 1)


def test_gaussian_sum_scaling_and_reproducible_independent_acquires():
    # 12000 independent quadratures: mean/std bounds exceed six sampling SEs.
    env = environment(std_adc=20, seed=41)
    prog = program(env, reps=6000, samples=9)
    prog.acquire(env.soc, progress=False)
    first = raw_of(prog)[0].copy()
    assert abs(first.mean()) < 2
    assert 58 < first.std() < 62  # sqrt(9) * 20 ADC sum units
    prog.acquire(env.soc, progress=False)
    assert not np.array_equal(first, raw_of(prog)[0])
    repeat = environment(std_adc=20, seed=41)
    replay = program(repeat, reps=6000, samples=9)
    replay.acquire(repeat.soc, progress=False)
    np.testing.assert_array_equal(first, raw_of(replay)[0])


def test_decimated_companion_raw_uses_same_trace_and_adc_order():
    env = environment(std_adc=3, seed=91, n_gens=2, n_readouts=2)
    prog = program(env, channels=(1, 0), reps=3, samples=7)
    traces = prog.acquire_decimated(env.soc, progress=False)
    assert traces is not None
    raw = raw_of(prog)
    for trace, accumulated in zip(traces, raw, strict=True):
        np.testing.assert_array_equal(
            accumulated[:, 0, :], np.rint(trace.sum(axis=-2)).astype(np.int64)
        )
    assert not np.array_equal(traces[0], traces[1])
    for ch, trace in zip((1, 0), traces, strict=True):
        np.testing.assert_array_equal(
            env.soc.get_decimated(ch, 0, 21), trace.reshape(-1, 2)
        )
        np.testing.assert_array_equal(
            env.soc.get_decimated(ch, 0, 21), trace.reshape(-1, 2)
        )


@pytest.mark.parametrize(
    "mode, options",
    [
        ("acquire", {}),
        ("acquire_decimated", {}),
        ("acquire_summed", {"processor": lambda iq: iq[0][:, :, 0].sum(axis=1)}),
    ],
)
def test_step_rounds_rejected_before_consuming_sequence(mode, options):
    env = environment(seed=11)
    prog = program(env)
    with pytest.raises(ValueError, match="step_rounds"):
        getattr(prog, mode)(env.soc, progress=False, step_rounds=True, **options)
    getattr(prog, mode)(env.soc, progress=False, **options)
    assert env.soc.binding.last_request.sequence == 0


def test_reentry_does_not_corrupt_original_rounds_and_stop_consumes_sequence():
    env = environment(seed=21)
    prog = program(
        env,
    )
    observed = []

    def hook(count, data, stop):
        observed.append(count)
        with pytest.raises(RuntimeError, match="active|reentr"):
            prog.acquire_decimated(env.soc, progress=False)
        if count == 2:
            stop.set()

    prog.acquire(
        env.soc,
        progress=False,
        round_hook=hook,
        rounds=3,
    )
    assert observed == [1, 2]
    rounds = prog.get_rounds()
    assert rounds is not None and len(rounds) == 2
    prog.acquire(
        env.soc,
        progress=False,
        rounds=3,
    )
    assert env.soc.binding.last_request.sequence == 1
    stop = Event()
    stop.set()
    with pytest.raises(StoppedPartialAcquireError):
        prog.acquire(
            env.soc,
            progress=False,
            cancel_flag=stop,
            rounds=3,
        )
    prog.acquire(
        env.soc,
        progress=False,
        rounds=3,
    )
    assert env.soc.binding.last_request.sequence == 3


@pytest.mark.parametrize("std_adc", [-1, float("inf"), float("nan")])
def test_invalid_std_rejected(std_adc):
    with pytest.raises(ValueError, match="std_adc"):
        environment(std_adc=std_adc)


def test_unbound_channel_rejected():
    env = environment(n_gens=2, n_readouts=2, routes=((0, 0),))
    prog = program(env, channels=(1,))
    with pytest.raises(
        ValueError,
        match=r"ADC 1 read 0 point 0: no wired generator route.*candidates=\[\].*wired=\[\].*time=\[",
    ):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


def test_numeric_source_round_snapshots_packing_and_fixed_quantized_controls():
    from zcu_tools.core.simulation import (
        ChannelStatistics,
        EnvironmentSnapshot,
        RoundStatistics,
        SamplingIdentity,
    )

    env = environment(std_adc=0, seed=12, n_gens=2, n_readouts=2)
    prog = program(env, reps=1, channels=(1, 0), samples=4)
    other = program(env, reps=1)
    readout = prog.modules[0]
    if not isinstance(readout, PulseReadout):
        raise AssertionError("synthetic program must provide its declared readout")
    value = 0.0
    seen = []

    def snapshot():
        return EnvironmentSnapshot(
            "synthetic-v1", "retained-model-v1", (("bias", value),)
        )

    from zcu_tools.program.v2.simulation import IntegratedRound

    from tests.program.v2.simulation.support import FixedRawShots

    def source(request):
        seen.append(request)
        bias = request.snapshot.values[0][1]
        # Deliberately reverse backend channel order; totals use ADC sum units.
        channels = []
        for ch in (0, 1):
            sums = np.rint(
                np.array([[10 * (1 - ch) + bias + 0.5, 10 * (1 - ch) + bias + 1.5]])
            )
            channels.append(
                ChannelStatistics(ch, sums, sums[:, None, :], np.zeros((2, 2)))
            )
        statistics = RoundStatistics(
            np.array([1], dtype=np.int64),
            tuple(channels),
            SamplingIdentity(12, (request.index,)),
        )
        return IntegratedRound(
            statistics, FixedRawShots({ch.channel: ch.sums for ch in channels})
        )

    def hook(count, data, stop):
        nonlocal value
        value = 2.0
        with pytest.raises(RuntimeError, match="active|reentr"):
            other.acquire(env.soc, progress=False)
        # Acquiring does not recompile live cfg changes between rounds.
        readout.cfg.gain = 0.9

    env.soc.binding.capture_snapshot = snapshot
    env.soc.binding.execute = source
    averaged = prog.acquire(
        env.soc,
        progress=False,
        round_hook=hook,
        rounds=2,
    )
    assert averaged is not None
    np.testing.assert_array_equal(raw_of(prog)[0], [[[2, 4]]])
    np.testing.assert_array_equal(raw_of(prog)[1], [[[12, 14]]])
    np.testing.assert_array_equal(averaged[0], [[0.25, 0.75]])
    np.testing.assert_array_equal(averaged[1], [[2.75, 3.25]])
    assert seen[0].acquire is seen[1].acquire
    assert [r.snapshot.values for r in seen] == [(("bias", 0.0),), (("bias", 2.0),)]
    assert [r.index for r in seen] == [0, 1]
    maxv = env.soccfg.get_maxv(1)
    assert seen[0].acquire.materialize_point(0).events[0].gain == int(0.1 * maxv) / maxv


@pytest.mark.parametrize(
    "invalid",
    [
        "nan",
        "infinity",
        "overflow",
        "underflow",
        "shape",
        "missing",
        "extra",
        "duplicate",
    ],
)
def test_invalid_backend_signals_fail_public_acquire_and_allow_retry(invalid):
    from zcu_tools.core.simulation import ChannelSignals, RoundSignals

    env = environment(std_adc=0, seed=45)
    prog = program(env, reps=1, samples=4)
    source = env.soc.binding.execute

    def malformed(request):
        shape = (1, 1, 4, 2)
        value = np.zeros(shape)
        if invalid in ("nan", "infinity", "overflow", "underflow"):
            value.flat[0] = {
                "nan": np.nan,
                "infinity": np.inf,
                "overflow": 2.0**63,
                "underflow": -(2.0**63) - 2048,
            }[invalid]
        elif invalid == "shape":
            value = np.zeros(shape[1:])
        channels = (ChannelSignals(0, value),)
        if invalid == "missing":
            channels = ()
        elif invalid == "duplicate":
            channels = channels + channels
        elif invalid == "extra":
            channels = channels + (ChannelSignals(1, value),)
        return RoundSignals(channels)

    env.soc.binding.execute = malformed
    acquire = prog.acquire_decimated
    reason = (
        "finite|int64"
        if invalid in ("nan", "infinity", "overflow", "underflow")
        else "expected.*got|channels"
    )
    with pytest.raises(ValueError, match=reason):
        acquire(env.soc, progress=False)
    failed_sequence = env.soc.binding.last_request.sequence
    env.soc.binding.execute = source
    result = acquire(env.soc, progress=False)
    np.testing.assert_array_equal(result, 0)
    np.testing.assert_array_equal(raw_of(prog), 0)
    assert env.soc.binding.last_request.sequence == failed_sequence + 1


@pytest.mark.parametrize("ro_length", [0.1, 1.5], ids=["no-carrier", "overlap"])
def test_scalar_readout_trigger_is_relative_to_module_not_pulse_predelay(ro_length):
    env = environment(std_adc=0)
    cfg = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=ro_length,
        pre_delay=1.5,
        trig_offset=0.25,
    )
    prog = ModularProgramV2(
        env.soccfg,
        modules=[cfg.build("readout")],
        reps=1,
    )
    if ro_length == 0.1:
        with pytest.raises(
            ValueError, match=r"ADC 0 read 0.*no pulse carrier in acquisition window"
        ):
            prog.acquire(env.soc, progress=False)
        assert not env.soc.binding.rounds
        return
    prog.acquire(env.soc, progress=False)
    request = env.soc.binding.last_request
    expected_trigger = prog.cycles2us(prog.us2cycles(0.25))
    assert request.readouts[0].trigger_us == (expected_trigger,)
    expected_pulse = prog.cycles2us(prog.us2cycles(1.5))
    assert request.materialize_point(0).events[0].start_us == expected_pulse
