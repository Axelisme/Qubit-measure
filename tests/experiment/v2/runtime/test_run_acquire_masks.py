"""Registered shot masks through Schedule and a simulated timeline program."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.cfg_model import CircleClassifier, LinearClassifier
from zcu_tools.cfg_model import ModuleCfgFactory as CoreModuleCfgFactory
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.simulation import SimulationEnvironment, white_noise
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.program.v2 import IQ, Classify, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import ModuleCfgFactory, PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate.boundary import (
    ChannelStatistics,
    RoundStatistics,
    SamplingIdentity,
)

from tests.program._profiles import load_soccfg
from tests.program.v2._mux_support import mux_soccfg, native_mux


@dataclass(frozen=True)
class LiteralShots:
    """Return independent integer buffers from literal (points, reps, reads, IQ) shots."""

    channels: Mapping[int, NDArray[np.int64]]

    def materialize(self, reps: int, avg_level: int) -> dict[int, NDArray[np.int64]]:
        return {
            ch: np.moveaxis(value, 1, avg_level).copy()
            for ch, value in self.channels.items()
        }


@pytest.fixture
def environment() -> Iterator[SimulationEnvironment]:
    soccfg = load_soccfg(n_gens=2, n_readouts=2)
    soccfg["readouts"][0]["iq_offset"] = 1.0
    env = white_noise(soccfg, std_adc=0, seed=13, routes=((0, 0), (1, 1)))
    try:
        yield env
    finally:
        env.devices.close_all_devices()


def emit_readout(builder: TimelineBuilder, *tags: str, channel: int = 0) -> None:
    readout = PulseReadoutCfg(
        gen_ch=channel,
        nqz=1,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_ch=channel,
        ro_length=0.1,
        freq=100.0,
    ).build(f"readout{channel}")
    for tag in tags:
        builder.emit(readout, tag=tag)
        builder.delay_auto(t=0.05)


def deliver_rounds(
    environment: SimulationEnvironment, *rounds: Mapping[int, NDArray[np.int64]]
) -> None:
    def delivery(
        index: int, channels: Mapping[int, NDArray[np.int64]]
    ) -> IntegratedRound:
        points, reps, _, _ = next(iter(channels.values())).shape
        return IntegratedRound(
            RoundStatistics(
                np.full((points, 1), reps, dtype=np.int64),
                tuple(
                    ChannelStatistics(
                        ch,
                        values.sum(axis=1, dtype=np.float64),
                        np.zeros((points, values.shape[2], 1, 2)),
                        np.zeros((2, 2)),
                    )
                    for ch, values in channels.items()
                ),
                SamplingIdentity(13, (index,)),
            ),
            LiteralShots(channels),
        )

    deliveries = iter(
        delivery(index, channels) for index, channels in enumerate(rounds)
    )
    environment.soc.binding.execute = lambda _: next(deliveries)


def test_herald_reset_masks_terminal_average_and_classification(
    environment: SimulationEnvironment,
) -> None:
    terminal = PulseReadoutCfg(
        gen_ch=0,
        nqz=1,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_ch=0,
        ro_length=0.1,
        post_delay=0.05,
        freq=100.0,
        classifier=LinearClassifier(angle=0.0, threshold=0.0),
    ).build("terminal")
    cfg = ModuleCfgFactory.from_core(
        CoreModuleCfgFactory.from_raw(
            {
                "type": "reset/herald",
                "readout": terminal.cfg.to_dict(),
                "keep": "g",
                "prepare": {
                    "type": "pulse",
                    "ch": 1,
                    "nqz": 1,
                    "freq": 150.0,
                    "gain": 0.1,
                    "waveform": {"style": "const", "length": 0.1},
                },
            }
        ),
        context=None,
    )
    assert cfg.type == "reset/herald"
    builder = TimelineBuilder(environment.soccfg, reps=3)
    builder.add_loop("point", 1)
    builder.emit(cfg.build("herald", terminal=terminal), tag="filter")
    builder.emit(
        PulseCfg(
            ch=1, nqz=1, freq=250.0, gain=0.1, waveform=ConstWaveformCfg(length=0.1)
        ).build("drive")
    )
    builder.emit(terminal, tag="signal")
    program = builder.build()
    normalized = np.array(
        [[[[1, 0], [100, 100]], [[-2, 0], [-4, 3]], [[0, 0], [10, 0]]]],
        dtype=np.int64,
    )
    raw = normalized * program.ro_chs[0]["length"]
    deliver_rounds(environment, {0: raw}, {0: raw})
    with Schedule(SignalBuffer((1,)), scope=Scope("run", "herald")) as schedule:
        averaged = schedule.run_acquire(program, environment.soc, remove_offset=False)
        np.testing.assert_array_equal(averaged, [2 + 1j])
    with Schedule(
        SignalBuffer((1, 2), dtype=np.float64), scope=Scope("run", "herald")
    ) as schedule:
        counts = schedule.run_acquire_summed(
            program, environment.soc, reducer=shot_sums(program, [Classify("signal")])
        )
        np.testing.assert_array_equal(counts, [[1, 1]])
    with pytest.raises(ValueError, match="filter"):
        shot_sums(program, [IQ("filter")])
    request = environment.soc.binding.last_request
    assert request is not None
    timeline = request.materialize_point(0)
    prepare, herald, drive, terminal_pulse = timeline.events
    (window,) = timeline.readouts
    herald_tick, terminal_tick = window.trigger_ticks
    assert (prepare.gen_ch, herald.gen_ch, drive.gen_ch, terminal_pulse.gen_ch) == (
        1,
        0,
        1,
        0,
    )
    assert prepare.start_ticks < herald_tick < drive.start_ticks < terminal_tick


def test_mux_herald_hides_its_reads_after_a_balanced_branch() -> None:
    env = white_noise(mux_soccfg(), std_adc=0, seed=13, routes=((0, 4), (0, 1), (0, 6)))
    try:
        readout = native_mux((4, 1, 6), post_delay=0.1)
        readout.tones[0].classifier = CircleClassifier(
            g_center=0j, e_center=5 + 0j, radius=1.0
        )
        readout.tones[2].classifier = LinearClassifier(angle=0.0, threshold=0.0)
        terminal = readout.build("terminal")
        cfg = ModuleCfgFactory.from_raw(
            {
                "type": "reset/mux_herald",
                "readout": readout,
                "keep": {"0": "g", "2": "e"},
            }
        )
        assert cfg.type == "reset/mux_herald"
        builder = TimelineBuilder(env.soccfg, reps=4)
        builder.add_loop("point", 2)
        with builder.branch(by="point", align_timeline=True) as branch:
            for case in range(2):
                with branch.case(case):
                    builder.emit(terminal)
        builder.emit(cfg.build("herald", terminal=terminal), tag="filter")
        builder.emit(terminal, tag="signal")
        program = builder.build()
        reads = {
            4: [[0, 5], [5, 100], [0, 100], [2, 100]],
            1: [[99, 12], [99, 100], [99, 100], [99, 100]],
            6: [[3, 16], [3, 100], [-1, 100], [3, 100]],
        }
        raw = {}
        for ch, values in reads.items():
            i = np.column_stack((np.full(4, 20), values))
            iq = np.stack((i, np.zeros_like(i)), axis=-1)
            raw[ch] = np.stack((iq, iq)) * program.ro_chs[ch]["length"]
        deliver_rounds(env, raw, raw)
        with Schedule(
            SignalBuffer((3, 2, 2, 2), dtype=np.float64), scope=Scope("run")
        ) as schedule:
            averaged = schedule.run_acquire(
                program,
                env.soc,
                options=RunOptions(converter=np.stack),
                remove_offset=False,
            )
            np.testing.assert_array_equal(
                averaged,
                [
                    [[[5, 0], [5, 0]], [[1.25, 0], [1.25, 0]]],
                    [[[5, 0], [5, 0]], [[3, 0], [3, 0]]],
                    [[[5, 0], [5, 0]], [[4, 0], [4, 0]]],
                ],
            )
        with Schedule(
            SignalBuffer((2, 3, 2), dtype=np.float64), scope=Scope("run")
        ) as schedule:
            counts = schedule.run_acquire_summed(
                program,
                env.soc,
                reducer=shot_sums(
                    program, [Classify("signal", tone=0), Classify("signal", tone=2)]
                ),
            )
            np.testing.assert_array_equal(
                counts, [[[0, 0], [0, 1], [0, 0]], [[0, 0], [0, 1], [0, 0]]]
            )
    finally:
        env.devices.close_all_devices()


def test_run_acquire_averages_only_kept_iq_over_all_shots_and_completed_rounds(
    environment: SimulationEnvironment,
) -> None:
    builder = TimelineBuilder(environment.soccfg, reps=3)
    builder.add_loop("point", 2)
    emit_readout(builder, "filter", "signal")
    builder.register_shot_mask_source({0: [0]}, lambda iq: iq[0][..., 0, 0] == 2)
    program = builder.build()
    normalized = np.array(
        [
            [
                [[[2, 0], [7, 4]], [[-2, 0], [100, 100]], [[-2, 0], [100, 100]]],
                [[[0, 0], [100, 100]], [[0, 0], [100, 100]], [[0, 0], [100, 100]]],
            ],
            [
                [[[2, 0], [4, 7]], [[2, 0], [10, 1]], [[-2, 0], [100, 100]]],
                [[[2, 0], [7, 1]], [[-2, 0], [100, 100]], [[2, 0], [7, 7]]],
            ],
        ],
        dtype=np.int64,
    )
    raw = normalized * program.ro_chs[0]["length"]
    deliver_rounds(environment, {0: raw[0]}, {0: raw[1]})
    updates = []
    signals = SignalBuffer(
        (2,), on_update=lambda data: updates.append(data.copy()), update_interval=1
    )
    with Schedule(signals, scope=Scope("run", "masked")) as schedule:
        result = schedule.run_acquire(
            program,
            environment.soc,
            rounds=3,
            options=RunOptions(stop_condition=lambda: len(updates) == 2),
        )
        np.testing.assert_array_equal(result, [3 + 1.5j, 2 + 1j])
    np.testing.assert_array_equal(
        updates, [[2 + 1j, 0j], [3 + 1.5j, 2 + 1j], [3 + 1.5j, 2 + 1j]]
    )
    retained = program.get_raw()
    assert retained is not None
    np.testing.assert_array_equal(retained[0], raw[1].transpose(1, 0, 2, 3))


def test_run_acquire_intersects_sources_in_order_and_omits_mask_only_adc(
    environment: SimulationEnvironment,
) -> None:
    builder = TimelineBuilder(environment.soccfg, reps=3, reps_innermost=True)
    builder.add_loop("point", 1)
    emit_readout(builder, "filter1", channel=1)
    emit_readout(builder, "filter0", "signal")
    calls = []

    def first(iq: Mapping[int, NDArray[np.float64]]) -> NDArray[np.bool_]:
        calls.append(tuple(iq))
        return iq[1][..., 0, 0] > 0

    def second(iq: Mapping[int, NDArray[np.float64]]) -> NDArray[np.bool_]:
        calls.append(tuple(iq))
        return iq[0][..., 0, 0] > 0

    builder.register_shot_mask_source({1: [0]}, first)
    builder.register_shot_mask_source({0: [0]}, second)
    program = builder.build()
    raw = {
        1: np.array([[[[1, 0]], [[1, 0]], [[-1, 0]]]]) * program.ro_chs[1]["length"],
        0: np.array([[[[1, 0], [6, 3]], [[-1, 0], [90, 90]], [[1, 0], [90, 90]]]])
        * program.ro_chs[0]["length"],
    }
    deliver_rounds(environment, raw)
    with Schedule(SignalBuffer((1,)), scope=Scope("run")) as schedule:
        result = schedule.run_acquire(program, environment.soc, remove_offset=False)
        np.testing.assert_array_equal(result, [2 + 1j])
    assert calls == [(1,), (0,)]


def test_run_acquire_shots_returns_visible_iq_and_the_unapplied_round_mask(
    environment: SimulationEnvironment,
) -> None:
    builder = TimelineBuilder(environment.soccfg, reps=3)
    builder.add_loop("point", 2)
    emit_readout(builder, "first", "filter0", "second")
    emit_readout(builder, "filter1", channel=1)
    first_adc = np.array(
        [
            [
                [[11, 1], [1, 0], [12, 2]],
                [[21, 3], [-1, 0], [22, 4]],
                [[31, 5], [1, 0], [32, 6]],
            ],
            [
                [[41, 7], [-1, 0], [42, 8]],
                [[51, 9], [1, 0], [52, 10]],
                [[61, 11], [1, 0], [62, 12]],
            ],
        ],
        dtype=np.int64,
    )
    second_adc = np.array(
        [[[[1, 0]], [[1, 0]], [[-1, 0]]], [[[1, 0]], [[1, 0]], [[1, 0]]]],
        dtype=np.int64,
    )
    builder.register_shot_mask_source({0: [1]}, lambda iq: iq[0][..., 0, 0] > 0)
    builder.register_shot_mask_source({1: [0]}, lambda iq: iq[1][..., 0, 0] > 0)
    program = builder.build()
    deliver_rounds(environment, {0: first_adc, 1: second_adc})

    with Schedule(scope=Scope("run")) as schedule:
        result = schedule.run_acquire_shots(program, environment.soc)
        assert result is not None
        assert set(result.shots) == {0}
        visible = result.shots[0]
        assert visible.shape == (2, 3, 2, 2)
        assert visible.dtype == np.int64
        np.testing.assert_array_equal(visible, first_adc[:, :, [0, 2], :])
        assert result.mask.dtype == np.bool_
        np.testing.assert_array_equal(
            result.mask, [[True, False, False], [False, True, True]]
        )


def test_run_acquire_shots_returns_an_all_true_mask_without_sources(
    environment: SimulationEnvironment,
) -> None:
    builder = TimelineBuilder(environment.soccfg, reps=2)
    builder.add_loop("point", 1)
    emit_readout(builder, "signal", channel=1)
    program = builder.build()
    raw = np.array([[[[-4, 5]], [[6, -7]]]], dtype=np.int64)
    deliver_rounds(environment, {1: raw})

    with Schedule(scope=Scope("run")) as schedule:
        result = schedule.run_acquire_shots(program, environment.soc)
        assert result is not None
        assert set(result.shots) == {1}
        np.testing.assert_array_equal(result.shots[1], raw)
        np.testing.assert_array_equal(result.mask, np.array([[True, True]]))


@pytest.mark.parametrize(
    ("mode", "reason"),
    [("decimated", "decimated"), ("threshold", "threshold")],
)
def test_run_acquire_rejects_modes_incompatible_with_mask_sources(
    environment: SimulationEnvironment, mode: str, reason: str
) -> None:
    builder = TimelineBuilder(environment.soccfg, reps=2)
    emit_readout(builder, "filter", "signal")
    builder.register_shot_mask_source({0: [0]}, lambda iq: iq[0][..., 0, 0] >= 0)
    program = builder.build()
    options = RunOptions[list[NDArray[np.float64]]](converter=lambda _: np.array(0j))
    with Schedule(SignalBuffer(()), scope=Scope("run")) as schedule:
        run = (
            schedule.run_acquire_decimated
            if mode == "decimated"
            else schedule.run_acquire
        )
        kwargs = {"threshold": 0} if mode == "threshold" else {}
        with pytest.raises(ValueError, match=rf"{reason}.*shot mask source"):
            run(program, environment.soc, options=options, **kwargs)


class _DigitReducer:
    def start(self) -> int:
        return 0

    def add(
        self,
        state: int,
        shots: Mapping[int, NDArray[np.float64]],
        mask: NDArray[np.bool_],
    ) -> int:
        digit = int(shots[0][mask, 0, 0].sum())
        return 10 * state + digit

    def output(self, state: int) -> NDArray[np.float64]:
        return np.array([state**2], dtype=np.float64)


def test_run_acquire_summed_keeps_reducer_state_through_last_completed_round(
    environment: SimulationEnvironment,
) -> None:
    builder = TimelineBuilder(environment.soccfg, reps=2)
    builder.add_loop("point", 1)
    emit_readout(builder, "signal")
    program = builder.build()
    length = program.ro_chs[0]["length"]
    first = np.array([[[[1, 0]], [[2, 0]]]]) * length
    second = np.array([[[[2, 0]], [[3, 0]]]]) * length
    deliver_rounds(environment, {0: first}, {0: second})
    updates = []
    signals = SignalBuffer(
        (1,),
        dtype=np.float64,
        on_update=lambda data: updates.append(data.copy()),
        update_interval=1,
    )
    with Schedule(signals, scope=Scope("run", "reduced")) as schedule:
        result = schedule.run_acquire_summed(
            program,
            environment.soc,
            reducer=_DigitReducer(),
            rounds=3,
            options=RunOptions(stop_condition=lambda: len(updates) == 2),
        )
        np.testing.assert_array_equal(result, [1225.0])
    np.testing.assert_array_equal(updates, [[9.0], [1225.0], [1225.0]])


def test_run_acquire_summed_counts_only_kept_shots_from_visible_reads(
    environment: SimulationEnvironment,
) -> None:
    builder = TimelineBuilder(environment.soccfg, reps=3)
    builder.add_loop("point", 2)
    emit_readout(builder, "filter", "signal")
    builder.register_shot_mask_source({0: [0]}, lambda iq: iq[0][..., 0, 0] > 0)
    program = builder.build()
    raw = (
        np.array(
            [
                [[[1, 0], [2, 1]], [[-1, 0], [99, 99]], [[1, 0], [3, -1]]],
                [[[-1, 0], [99, 99]], [[-1, 0], [99, 99]], [[-1, 0], [99, 99]]],
            ]
        )
        * program.ro_chs[0]["length"]
    )
    deliver_rounds(environment, {0: raw})
    with Schedule(
        SignalBuffer((2, 6), dtype=np.float64), scope=Scope("run")
    ) as schedule:
        result = schedule.run_acquire_summed(
            program, environment.soc, reducer=shot_sums(program, [IQ("signal")])
        )
        np.testing.assert_array_equal(
            result, [[2.0, 5.0, 0.0, 13.0, -1.0, 2.0], [0.0] * 6]
        )
    shots = program.get_shot_result()
    assert shots is not None
    np.testing.assert_array_equal(shots.mask, [[True, False, True], [False] * 3])
