"""Semantic loop/table scheduling through real program acquisition."""

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.control import Branch, Repeat
from zcu_tools.program.v2.modules.delay import DelayAuto, Join
from zcu_tools.program.v2.modules.dmem import LoadValue, LoadWord, ScanWith
from zcu_tools.program.v2.modules.pulse import PulseCfg, TableLengthPulse
from zcu_tools.program.v2.modules.readout import (
    PulseReadout,
    PulseReadoutCfg,
    TablePulseReadout,
)
from zcu_tools.program.v2.modules.reset import NoneResetCfg, TwoPulseResetCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.utils import readout_freq_words

from tests.program.v2.simulation.support import raw_of, request_of


def drive(length=0.1, gain=0.2):
    return PulseCfg(
        ch=1, nqz=1, freq=80, gain=gain, waveform=ConstWaveformCfg(length=length)
    )


def receiver():
    return PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )


def acquire(modules, *, sweep=None, reps=1):
    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0), (1, 0)))
    prog = ModularProgramV2(
        env.soccfg, ProgramV2Cfg(reps=reps), modules=modules, sweep=sweep
    )
    try:
        prog.acquire(env.soc, progress=False)
    except ValueError as exc:
        pytest.fail(f"supported semantic control must acquire: {exc}")
    return request_of(env), prog, env


@pytest.mark.parametrize("compress", [False, True])
def test_loadvalue_repeat_branch_computedpulse_and_wait_follow_execution_order(
    compress,
):
    loader = LoadValue(
        "selection",
        [1, 0] * 16,
        idx_reg="shot",
        val_reg="selected",
        auto_compress=compress,
    )
    candidates = [drive(0.1, 0.2), drive(0.3, -0.4)]
    computed = ComputedPulse("gate", val_reg="selected", pulses=candidates)
    branch = Branch("choice", computed, drive(0.2).build("other"), compare_by="inner")
    loop = Repeat("inner", 2).add_content(branch)
    request, _, _ = acquire(
        [loader, loop, receiver().build("readout")], sweep=[("shot", 32)]
    )
    for i, timeline in enumerate(
        request.materialize_point(p) for p in range(int(np.prod(request.point_dims)))
    ):
        first, second, readout = timeline.events
        assert first.gain == pytest.approx(-0.4 if i % 2 == 0 else 0.2, abs=1e-4)
        assert first.duration_us == pytest.approx(0.3 if i % 2 == 0 else 0.1, abs=0.003)
        # ComputedPulse reserves the max candidate, even when selecting the short one.
        assert second.start_us == pytest.approx(0.3, abs=0.007)
        assert readout.start_us == pytest.approx(0.5, abs=0.012)
        assert ("inner", 0) in first.iterations
        assert ("inner", 1) in second.iterations
        assert dict(timeline.coordinates)["shot"] == i


def test_scanwith_runtime_repeat_count_and_delay_cycles_are_local_to_each_acquire():
    scan = ScanWith("scan", [0, 1, 2], val_reg="count").add_content(
        Repeat("repeat", "count", range_hint=(0, 2)).add_content(drive().build("drive"))
    )
    wait = LoadValue("wait", [43], idx_reg="reps", val_reg="cycles")
    modules = [
        wait,
        scan,
        DelayAuto("wait_loaded", t="cycles"),
        receiver().build("readout"),
    ]
    request, prog, env = acquire(modules)
    assert [e.source for e in request.materialize_point(0).events] == [
        "drive",
        "drive",
        "drive",
        "readout",
    ]
    assert request.materialize_point(0).events[-1].start_us == pytest.approx(
        0.4, abs=0.015
    )
    prog.acquire(env.soc, progress=False)
    assert request_of(env).materialize_point(0) == request.materialize_point(0)


def test_tablelength_uses_selected_duration_and_sweeps_use_quantized_controls():
    pulse = TableLengthPulse(
        "length", drive(), lengths=[0.1, 0.3], idx_reg="length_axis"
    )
    cfg = receiver()
    cfg.gain = QickParam(0.1, {"length_axis": 0.1})
    request, prog, _ = acquire(
        [pulse, cfg.build("readout")], sweep=[("length_axis", 2)], reps=2
    )
    assert request.loop_dims == (2, 2)
    for i, events in enumerate(
        request.materialize_point(p).events
        for p in range(int(np.prod(request.point_dims)))
    ):
        assert events[0].duration_us == pytest.approx([0.1, 0.3][i % 2], abs=0.003)
        assert events[1].start_us == pytest.approx([0.1, 0.3][i % 2], abs=0.006)
        assert events[1].gain == pytest.approx([0.1, 0.2][i % 2], abs=1e-4)
    assert raw_of(prog)[0].shape == (2, 2, 1, 2)


def test_loadword_uint32_frequency_and_signed_gain_are_decoded_only_by_readout():
    env = white_noise(std_adc=0)
    gen, adc = readout_freq_words(
        env.soccfg, [100, 200], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    assert adc[0] > 2**31  # final downconversion pattern, not a positive scalar
    modules = [
        LoadWord("freq", gen, "axis", "freq_word"),
        LoadWord("adc", adc, "axis", "adc_word"),
        LoadWord("gain", [3276, (-6552) % 2**32], "axis", "gain_word"),
        PulseReadout(
            "readout",
            receiver(),
            freq_val="freq_word",
            ro_freq_val="adc_word",
            gain_val="gain_word",
        ),
    ]
    request, _, _ = acquire(modules, sweep=[("axis", 2)])
    assert [
        e[0].frequency_mhz
        for e in (
            request.materialize_point(p).events
            for p in range(int(np.prod(request.point_dims)))
        )
    ] == pytest.approx([100, 200], abs=1e-4)
    assert [
        e[0].gain
        for e in (
            request.materialize_point(p).events
            for p in range(int(np.prod(request.point_dims)))
        )
    ] == pytest.approx([3276 / 32766, -6552 / 32766])
    assert [
        t.readouts[0].frequency_mhz[0]
        for t in (
            request.materialize_point(p)
            for p in range(int(np.prod(request.point_dims)))
        )
    ] == pytest.approx([100, 200], abs=1e-4)


def test_table_readout_initial_template_exec_after_and_outer_wrap_are_not_index_shortcuts():
    env = white_noise(std_adc=0)
    gen, adc = readout_freq_words(
        env.soccfg, [200, 300], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    # The compiled template starts at cfg.freq=100. The last hook restores table[0]=200.
    readout = TablePulseReadout(
        "readout", receiver(), idx_reg="axis", freq_words=gen, ro_freq_words=adc
    )
    request, prog, env = acquire([readout], sweep=[("axis", 2)], reps=1)
    assert [
        events[0].frequency_mhz
        for events in (
            request.materialize_point(p).events
            for p in range(int(np.prod(request.point_dims)))
        )
    ] == pytest.approx([100, 300], abs=1e-4)
    prog.acquire(env.soc, progress=False)
    assert [
        events[0].frequency_mhz
        for events in (
            request_of(env).materialize_point(p).events
            for p in range(int(np.prod(request.point_dims)))
        )
    ] == pytest.approx([100, 300], abs=1e-4)


def test_reset_wrappers_expand_only_actual_parallel_pulses():
    reset = TwoPulseResetCfg(
        pulse1_cfg=drive(0.1),
        pulse2_cfg=PulseCfg(
            ch=0, nqz=1, freq=60, gain=0.2, waveform=ConstWaveformCfg(length=0.2)
        ),
    ).build("reset")
    request, _, _ = acquire(
        [NoneResetCfg().build("none"), reset, receiver().build("readout")]
    )
    assert len(request.materialize_point(0).events) == 3
    assert [e.start_us for e in request.materialize_point(0).events[:2]] == [0, 0]
    assert request.materialize_point(0).events[-1].start_us == pytest.approx(
        0.2, abs=0.004
    )


def test_join_cross_branch_register_dependency_rejected_before_backend():
    modules = [
        Join(
            LoadValue("writer", [0], "reps", "selection"),
            ComputedPulse("reader", val_reg="selection", pulses=[drive(), drive(0.2)]),
        ),
        receiver().build("readout"),
    ]
    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0), (1, 0)))
    prog = ModularProgramV2(env.soccfg, ProgramV2Cfg(), modules=modules)
    with pytest.raises(ValueError, match="Join.*register.*selection"):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


def test_unknown_word_consumer_fails_with_module_coordinate_and_register():
    modules = [
        LoadWord("raw", [1], "reps", "cycles"),
        DelayAuto("invalid_wait", t="cycles"),
        receiver().build("readout"),
    ]
    env = white_noise(std_adc=0)
    prog = ModularProgramV2(env.soccfg, ProgramV2Cfg(), modules=modules)
    with pytest.raises(ValueError, match="DelayAuto.*invalid_wait.*reps.*cycles.*word"):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds
