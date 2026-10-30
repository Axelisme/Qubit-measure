"""Emit programs use the ordinary QICK pipeline, not the legacy simulator frontend."""

from collections.abc import Mapping

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._acquisition_soc import make_acquisition_soc


def _make_readout() -> PulseReadout:
    return PulseReadoutCfg(
        gen_ch=0,
        nqz=1,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_ch=0,
        ro_length=0.1,
        freq=100.0,
    ).build("readout")


class ReadoutTimeline(TimelineProgramV2):
    """Hand-written native program declaring and emitting one receiver module."""

    def _initialize(self, cfg):
        self.readout = _make_readout()
        self.declare(self.readout)

    def _body(self, cfg):
        self.emit(self.readout)


def _sum_iq(iq: Mapping[int, NDArray[np.float64]]) -> NDArray[np.float64]:
    return iq[0][:, :, 0, :].sum(axis=1)


@pytest.mark.parametrize("source", ["handwritten", "builder"])
@pytest.mark.parametrize("mode", ["integrated", "summed", "decimated"])
def test_timeline_acquires_through_schedule_and_real_qick_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    source: str,
) -> None:
    soc, soccfg = make_acquisition_soc()
    if source == "builder":
        builder = TimelineBuilder(soccfg, reps=2)
        builder.emit(_make_readout())
        program = builder.build()
    else:
        program = ReadoutTimeline(soccfg, reps=2)
    samples = program.ro_chs[0]["length"]

    def poll():
        raw = np.tile([3 * samples, 4 * samples], (2, 1)).astype(np.int64)
        return [(2, ([raw], {}))]

    def decimated(ch, address, length):
        return np.tile([3.0, 4.0], (length, 1))

    monkeypatch.setattr(soc, "poll_data", poll)
    monkeypatch.setattr(soc, "get_decimated", decimated)
    shape = (2, samples) if mode == "decimated" else (2,) if mode == "summed" else (1,)
    dtype = np.float64 if mode == "summed" else np.complex128
    updates = []
    buffer = SignalBuffer(
        shape,
        dtype=dtype,
        update_interval=1,
        on_update=lambda data: updates.append(data.copy()),
    )
    expected = np.full(shape, np.nan, dtype=dtype)
    with Schedule({}, buffer, scope=Scope("run")) as schedule:
        if mode == "integrated":
            result = schedule.managed_acquire(program, soc, rounds=2)
            expected = np.array([3 + 4j])
        elif mode == "summed":
            result = schedule.managed_acquire_summed(
                program, soc, rounds=2, processor=_sum_iq
            )
            expected = np.array([12.0, 16.0])
        else:
            result = schedule.managed_acquire_decimated(program, soc, rounds=2)
            expected = np.full((2, samples), 3 + 4j)
        np.testing.assert_array_equal(result, expected)
        assert np.shares_memory(result, buffer.array)
        assert schedule.outcome.status == "completed"
    assert len(updates) == 3
    np.testing.assert_array_equal(updates[-1], expected)
