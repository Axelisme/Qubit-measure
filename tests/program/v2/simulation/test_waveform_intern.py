"""Compiled waveform reuse observed through detached request records."""

import hashlib
from dataclasses import fields, is_dataclass

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.control import Repeat
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
from zcu_tools.program.v2.simulation.frontend import build_request


def content_digest(value):
    """Characterize all template fields, ignoring shared object identities."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode())
            digest.update(item.tobytes())
        elif is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


@pytest.mark.parametrize("kind", ["gauss", "drag", "flat_top", "arb"])
def test_repeat_candidates_extract_compiled_content_once(kind, tmp_path, monkeypatch):
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
    prog = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=2),
        modules=[
            LoadValue("select", [0, 1], "axis", "selected"),
            Repeat("repeat", 3).add_content(
                ComputedPulse("gate", val_reg="selected", pulses=candidates)
            ),
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=0.1,
            ).build("readout"),
        ],
        sweep=[("axis", 2)],
    )
    table = prog.envelopes[1]["envs"]
    assert isinstance(table, dict)
    buffers = list(table.values())
    sample_count = len(buffers[0]["data"])
    extractions = []
    original = np.arange

    def observed_arange(*args, **kwargs):
        if args == (sample_count,):
            extractions.append(1)
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "arange", observed_arange)
        request = build_request(prog, "integrated", env.soc.binding, 0)
    # These digests characterize the original lowering, including flat-top
    # ramp/plateau slots, complex samples, candidate order and every point array.
    expected = {
        "gauss": "cf4af2dee86009b7d7a764f2dc1fb11f4223995530d9d35ac59f97ece396e8f8",
        "drag": "843dc2578e9293fc1ac44df45fdd86cef4eeb2bd34d50bf7fee12373dfd14035",
        "flat_top": "67521e134b91ead1aa64a7596e69230d240aeb32e9745ac2c666c356cb818704",
        "arb": "65fd4cd5d58a7894f2fb097d7ac75a8f4bf03210df5fd0e4da11f8e7098ef98d",
    }
    assert content_digest(request.templates) == expected[kind]
    events = request.materialize_point(1).events
    assert events[0].gain < 0
    assert events[0].phase_deg == pytest.approx(37, abs=1e-5)
    assert len(extractions) == 1
    if kind != "flat_top":
        envelopes = [
            e for slot in request.templates[0].events[:-1] for e in slot.envelopes
        ]
        assert all(e is envelopes[0] for e in envelopes)
        envelope = envelopes[0]
        assert envelope is not None
        data = buffers[0]["data"]
        expected_values = (data[:, 0] + 1j * data[:, 1]) / prog.soccfg.get_maxv(1)
        np.testing.assert_array_equal(envelope.values, expected_values)
    # A later preparation must read current compiled bytes, even with the same
    # channel/name. Previous records remain detached from those mutable buffers.
    before = content_digest(request.templates)
    for buffer in buffers:
        buffer["data"][:, 0] = 0
    later = build_request(prog, "integrated", env.soc.binding, 1)
    assert content_digest(later.templates) != before
    assert content_digest(request.templates) == before


@pytest.mark.parametrize("difference", ["samples", "clock", "maxv"])
def test_equal_names_do_not_merge_different_compiled_content(difference):
    env = white_noise(std_adc=0, n_gens=3, routes=((0, 0), (1, 0), (2, 0)))
    waveform = GaussWaveformCfg(length=0.1, sigma=0.02)
    prog = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=1),
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
    request = build_request(prog, "integrated", env.soc.binding, 0)
    first, second = [slot.envelopes[0] for slot in request.templates[0].events[:2]]
    assert first is not None and second is not None
    assert first is not second
    assert first != second
    if difference == "clock":
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(np.array(first.sample_us) / 2, second.sample_us)
    elif difference == "maxv":
        np.testing.assert_array_equal(first.sample_us, second.sample_us)
        np.testing.assert_array_equal(np.array(first.values) * 2, second.values)
    else:
        assert any(value.imag != 0 for value in second.values)


def test_flat_top_durations_keep_distinct_sample_axes():
    from zcu_tools.program.v2.simulation.timeline import TimelineInterpreter

    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0), (1, 0)))
    prog = ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=1),
        modules=[
            PulseCfg(
                ch=1,
                nqz=1,
                freq=80,
                gain=0.2,
                waveform=FlatTopWaveformCfg(
                    length=length,
                    raise_waveform=GaussWaveformCfg(length=0.1, sigma=0.02),
                ),
            ).build(f"flat{i}")
            for i, length in enumerate((0.15, 0.20))
        ],
    )
    # The scalar diagnostic seam retains whole flat-tops, making the playback
    # duration part of envelope identity observable without decomposition.
    first, second = TimelineInterpreter(prog).run_point({"reps": 0}).events
    assert first.envelope is not None and second.envelope is not None
    assert first.envelope.values == second.envelope.values
    assert first.envelope.sample_us != second.envelope.sample_us
    assert second.envelope.sample_us[-1] - first.envelope.sample_us[
        -1
    ] == pytest.approx(second.duration_us - first.duration_us)
