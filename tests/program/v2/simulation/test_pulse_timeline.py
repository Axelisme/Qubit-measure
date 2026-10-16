"""Generator and ADC timing observed at the public acquisition boundary."""

import math

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.delay import Delay, Join, SoftDelay
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)

from tests.program.v2.simulation.support import request_of


@pytest.mark.parametrize(
    "waveform",
    [
        ConstWaveformCfg(length=0.25),
        CosineWaveformCfg(length=0.25),
        GaussWaveformCfg(length=0.25, sigma=0.05),
        DragWaveformCfg(length=0.25, sigma=0.05, delta=200, alpha=0.5),
        FlatTopWaveformCfg(length=0.25, raise_waveform=CosineWaveformCfg(length=0.1)),
    ],
    ids=["const", "cosine", "gauss", "drag", "flat_top"],
)
def test_parallel_pulses_retain_envelopes_phase_and_separate_adc_duration(waveform):
    env = white_noise(std_adc=0, seed=4, n_gens=2, routes=((0, 0), (1, 0)))
    pulse = PulseCfg(
        ch=1, nqz=1, freq=80, gain=-0.4, phase=37, waveform=waveform
    ).build("drive")
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.5,
        trig_offset=0.02,
    ).build("readout")
    prog = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=2),
        modules=[
            Delay("origin", 0.5),
            Join(pulse, [SoftDelay("offset", 0.1), readout]),
        ],
    )
    try:
        prog.acquire(env.soc, progress=False)
    except ValueError as exc:
        pytest.fail(f"supported pulse/timing modules must acquire: {exc}")
    request = request_of(env)
    events = request.materialize_point(0).events
    drive = next(event for event in events if event.gen_ch == 1)
    receiver = next(event for event in events if event.gen_ch == 0)
    assert drive.start_us == pytest.approx(0.5, abs=0.003)
    assert receiver.start_us == pytest.approx(0.6, abs=0.005)
    drive_end = max(e.start_us + e.duration_us for e in events if e.gen_ch == 1)
    assert drive_end - drive.start_us == pytest.approx(0.25, abs=0.006)
    assert drive.gain == pytest.approx(-0.4, abs=1 / 32766)
    assert drive.phase_deg == pytest.approx(37, abs=1e-5)
    assert drive.phase_reference_us == receiver.phase_reference_us == 0
    origin = math.ceil(0.5 * 430.08) / 430.08
    offset = math.ceil(0.1 * 430.08) / 430.08
    trigger = origin + round((offset + 0.02) * 430.08) / 430.08
    assert request.materialize_point(0).readouts[0].trigger_us[0] == pytest.approx(
        trigger
    )
    assert request.readouts[0].samples == prog.ro_chs[0]["length"]
    assert (
        request.readouts[0].samples / request.readouts[0].sample_rate_mhz
        > receiver.duration_us
    )
    assert drive.source == "drive"
    if waveform.style == "const":
        assert drive.envelope is None  # unity envelope, signed gain stays separate
    else:
        envelope = drive.envelope
        assert envelope is not None
        assert len(envelope.sample_us) == len(envelope.values)
        assert np.all(np.diff(envelope.sample_us) > 0)
        assert max(abs(v) for v in envelope.values) <= 1.001
        if waveform.style == "drag":
            assert (
                min(v.imag for v in envelope.values)
                < 0
                < max(v.imag for v in envelope.values)
            )
    assert request.materialize_point(0).events == events
    # Frozen request survives later mutation of its originating module cfg.
    if pulse.cfg is None:
        raise AssertionError("drive fixture must provide a configured pulse")
    pulse.cfg.gain = 0.9
    assert drive.gain < 0
