"""Registered shot masks through Schedule and a simulated timeline program."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.simulation import SimulationEnvironment, white_noise
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate.boundary import (
    ChannelStatistics,
    RoundStatistics,
    SamplingIdentity,
)

from tests.program._profiles import load_soccfg


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


@pytest.mark.parametrize("mode", ["shots", "summed"])
def test_unmigrated_run_modes_reject_mask_sources(
    environment: SimulationEnvironment, mode: str
) -> None:
    """Remove each case when that Schedule mode starts accepting shot masks."""
    builder = TimelineBuilder(environment.soccfg, reps=2)
    emit_readout(builder, "filter", "signal")
    builder.register_shot_mask_source({0: [0]}, lambda iq: iq[0][..., 0, 0] >= 0)
    program = builder.build()
    with Schedule(SignalBuffer((), dtype=np.float64), scope=Scope("run")) as schedule:
        if mode == "shots":
            with pytest.raises(
                ValueError, match=r"run_acquire_shots.*shot mask source"
            ):
                schedule.run_acquire_shots(program, environment.soc)
        else:
            with pytest.raises(
                ValueError, match=r"run_acquire_summed.*shot mask source"
            ):
                schedule.run_acquire_summed(
                    program,
                    environment.soc,
                    processor=lambda iq: np.ones(iq[0].shape[0]),
                )
