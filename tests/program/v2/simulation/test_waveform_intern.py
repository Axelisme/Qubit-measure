from zcu_tools.program.v2 import TimelineBuilder
from tests.program._timeline import compile_modules
"""Compiled waveform reuse observed through detached request records."""

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import (
    ArbWaveformCfg,
    ConstWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.simulation.compiled import build_compiled_request

from tests.program.v2.simulation.time_observations import envelope_times_us


@pytest.mark.parametrize("kind", ["gauss", "drag", "flat_top", "arb"])
def test_repeated_waveforms_preserve_samples_signed_gain_and_phase(kind, tmp_path):
    if kind == "arb":
        path = tmp_path / "wave.npz"
        np.savez(path, time=[0, 0.04, 0.1], idata=[0, 0.5, 0], qdata=[0.1, 0, -0.2])
        waveform = ArbWaveformCfg(data="wave", path=str(path))
    elif kind == "drag":
        waveform = DragWaveformCfg(length=0.1, sigma=0.02, delta=200, alpha=0.5)
    elif kind == "flat_top":
        waveform = FlatTopWaveformCfg(
            length=0.15, raise_waveform=GaussWaveformCfg(length=0.1, sigma=0.02)
        )
    else:
        waveform = GaussWaveformCfg(length=0.1, sigma=0.02)
    env = white_noise(std_adc=0, seed=4, n_gens=2, routes=((0, 0), (1, 0)))
    candidates = [
        PulseCfg(
            ch=1, nqz=1, freq=80, phase=37 * i, gain=-0.2 * (i + 1), waveform=waveform
        )
        for i in range(2)
    ]
    builder = TimelineBuilder(env.soccfg, reps=2)
    builder.add_loop("axis", 2)
    builder.emit(LoadValue("select", [0, 1], "axis", "selected"))
    with builder.repeat("repeat", 3):
        builder.emit(ComputedPulse("gate", val_reg="selected", pulses=candidates))
    builder.emit(PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=0.1,
        ).build("readout"))
    builder.delay_by_timeline()
    prog = builder.build()
    table = prog.envelopes[1]["envs"]
    assert isinstance(table, dict)
    buffers = list(table.values())
    request = build_compiled_request(
        prog, "integrated", env.soc.binding, 0, routes=env.soc.routes
    )
    events = [e for e in request.materialize_point(1).events if e.source == "gate"]
    assert events
    assert all(e.gain < 0 for e in events)
    assert all(e.phase_deg == pytest.approx(37, abs=1e-5) for e in events)
    data = buffers[0]["data"]
    expected_values = (data[:, 0] + 1j * data[:, 1]) / prog.soccfg.get_maxv(1)
    # Observe playback samples, not template candidate identities or ordering.
    shaped = [e.envelope for e in events if e.envelope is not None]
    assert shaped
    if kind == "flat_top":
        # Each repetition plays both halves of the compiled ramp around a plateau.
        half = len(expected_values) // 2
        np.testing.assert_array_equal(
            np.concatenate([shaped[0].values, shaped[1].values]),
            np.concatenate([expected_values[:half], expected_values[-half:]]),
        )
    else:
        for envelope in shaped:
            np.testing.assert_array_equal(envelope.values, expected_values)
    for event in events:
        if event.envelope is not None:
            np.testing.assert_allclose(
                envelope_times_us(request.time_base, event.envelope),
                np.arange(len(event.envelope.values))
                / (
                    prog.soccfg["gens"][1]["f_fabric"]
                    * prog.soccfg["gens"][1]["samps_per_clk"]
                ),
            )
    original = tuple(request.materialize_point(point) for point in range(2))
    for buffer in buffers:
        buffer["data"][:, 0] = 0
    later = build_compiled_request(
        prog, "integrated", env.soc.binding, 1, routes=env.soc.routes
    )
    changed = tuple(later.materialize_point(point) for point in range(2))
    assert changed != original
    assert tuple(request.materialize_point(point) for point in range(2)) == original
    assert all(
        value.real == 0
        for point in changed
        for event in point.events
        if event.source == "gate" and event.envelope is not None
        for value in event.envelope.values
    )


@pytest.mark.parametrize("difference", ["samples", "clock", "maxv"])
def test_equal_names_do_not_merge_different_compiled_content(difference):
    env = white_noise(std_adc=0, n_gens=3, routes=((0, 0), (1, 0), (2, 0)))
    waveform = GaussWaveformCfg(length=0.1, sigma=0.02)
    prog = compile_modules(
        env.soccfg,
        modules=[
            PulseCfg(ch=ch, nqz=1, freq=80, gain=0.2, waveform=waveform).build(
                f"drive{ch}"
            )
            for ch in (1, 2)
        ]
        + [
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=0.1,
            ).build("readout")
        ],
        reps=1,
    )
    table = prog.envelopes[2]["envs"]
    assert isinstance(table, dict)
    buffers = list(table.values())
    if difference == "samples":
        buffers[0]["data"][:, 1] = 7
    elif difference == "clock":
        prog.soccfg["gens"][2]["f_fabric"] *= 2
    else:
        prog.soccfg["gens"][2]["maxv"] //= 2
    request = build_compiled_request(
        prog, "integrated", env.soc.binding, 0, routes=env.soc.routes
    )
    first, second = [
        event.envelope for event in request.materialize_point(0).events[:2]
    ]
    assert first is not None and second is not None
    assert first != second
    if difference == "clock":
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(
            np.array(envelope_times_us(request.time_base, first)) / 2,
            envelope_times_us(request.time_base, second),
        )
    elif difference == "maxv":
        np.testing.assert_array_equal(
            envelope_times_us(request.time_base, first),
            envelope_times_us(request.time_base, second),
        )
        np.testing.assert_array_equal(np.array(first.values) * 2, second.values)
    else:
        assert any(value.imag != 0 for value in second.values)
