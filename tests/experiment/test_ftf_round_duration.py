"""FTF environment pacing and its opt-out leave public acquisition unchanged."""

import time

import numpy as np
import pytest
from zcu_tools.experiment.ftf import ftf
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.program.v2.simulation.support import raw_of
from tests.resources._ftf_data import ftf_document

pytestmark = pytest.mark.sim


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_default_paces_every_round_and_builder_zero_preserves_bitwise_results(
    monkeypatch: pytest.MonkeyPatch, mode: str
):
    waits: list[float] = []
    now = [0.0]

    def sleep(duration: float) -> None:
        waits.append(duration)
        now[0] += duration

    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(time, "sleep", sleep)
    # The synthetic settings omit pacing; the builder supplies physical time.
    settings = parse_ftf(ftf_document())
    env = ftf(settings)
    try:
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=6000.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.04),
            ro_length=0.04,
        )
        program = ModularProgramV2(
            env.soccfg, ProgramV2Cfg(reps=2, rounds=2), modules=[readout.build("ro")]
        )
        acquire = program.acquire if mode == "integrated" else program.acquire_decimated
        expected = acquire(env.soc, progress=False)
        expected_raw = raw_of(program)
        assert waits == pytest.approx([2.08e-6, 2.08e-6], abs=10e-9)
        unpaced = ftf(settings, time_scale=0)
        try:
            actual = acquire(unpaced.soc, progress=False)
            assert waits == pytest.approx([2.08e-6, 2.08e-6], abs=10e-9)
            assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
            assert (
                np.asarray(raw_of(program)).tobytes()
                == np.asarray(expected_raw).tobytes()
            )
            assert len(unpaced.soc.binding.rounds) == 2
            assert [r.snapshot for r in unpaced.soc.binding.rounds] == [
                r.snapshot for r in env.soc.binding.rounds
            ]
        finally:
            unpaced.devices.close_all_devices()
    finally:
        env.devices.close_all_devices()


def test_yaml_zero_disables_round_wait(monkeypatch: pytest.MonkeyPatch):
    waits: list[float] = []
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(time, "sleep", waits.append)
    document = ftf_document()
    settings = parse_ftf(document)
    document["numerics"] = {**settings.numerics.model_dump(), "time_scale": 0}
    env = ftf(parse_ftf(document))
    try:
        program = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=2, rounds=2),
            modules=[
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=6000.0,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.04),
                    ro_length=0.04,
                ).build("ro")
            ],
        )
        result = program.acquire(env.soc, progress=False)
        assert result is not None
        assert waits == []
        assert np.isfinite(result).all()
    finally:
        env.devices.close_all_devices()
