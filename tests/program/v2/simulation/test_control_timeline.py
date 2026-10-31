"""Semantic loop/table scheduling through real program acquisition.

Remove the scalar diagnostic reset parameter when the Module-tree interpreter
is deleted. Keep the acquisition case on the public acquisition route.
"""

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.base import Module
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
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.utils import readout_freq_words

from tests.program.v2.simulation.support import raw_of, request_of
from tests.program.v2.simulation.time_observations import time_us, times_us


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
        env.soccfg,
        modules=modules,
        sweep=sweep,
        reps=reps,
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
        assert time_us(request.time_base, first.duration_ticks) == pytest.approx(
            0.3 if i % 2 == 0 else 0.1, abs=0.003
        )
        # ComputedPulse reserves the max candidate, even when selecting the short one.
        assert time_us(request.time_base, second.start_ticks) == pytest.approx(
            0.3, abs=0.007
        )
        assert time_us(request.time_base, readout.start_ticks) == pytest.approx(
            0.5, abs=0.012
        )
        assert ("inner", 0) in first.iterations
        assert ("inner", 1) in second.iterations
        assert dict(timeline.coordinates)["shot"] == i


def test_scanwith_runtime_repeat_count_and_delay_cycles_are_local_to_each_acquire():
    scan = ScanWith("scan", [0, 1, 2], val_reg="count").add_content(
        Repeat("repeat", "count", range_hint=(0, 2)).add_content(drive().build("drive"))
    )
    wait = LoadValue("wait", [43, 86], idx_reg="shot", val_reg="cycles")
    modules = [
        wait,
        scan,
        DelayAuto("wait_loaded", t="cycles"),
        receiver().build("readout"),
    ]
    request, prog, env = acquire(modules, sweep=[("shot", 2)])
    assert request.point_dims == (2,)
    for point, readout_start in enumerate([0.4, 0.5]):
        timeline = request.materialize_point(point)
        assert [e.source for e in timeline.events] == [
            "drive",
            "drive",
            "drive",
            "readout",
        ]
        assert time_us(
            request.time_base, timeline.events[-1].start_ticks
        ) == pytest.approx(readout_start, abs=0.015)
    prog.acquire(env.soc, progress=False)
    for point in range(2):
        assert request_of(env).materialize_point(point) == request.materialize_point(
            point
        )


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
        assert events[0].source == "length"
        assert time_us(request.time_base, events[0].duration_ticks) == pytest.approx(
            [0.1, 0.3][i % 2], abs=0.003
        )
        assert time_us(request.time_base, events[1].start_ticks) == pytest.approx(
            [0.1, 0.3][i % 2], abs=0.006
        )
        assert events[1].gain == pytest.approx([0.1, 0.2][i % 2], abs=1e-4)
    assert raw_of(prog)[0].shape == (2, 2, 1, 2)


@pytest.mark.parametrize("flat_top", [False, True])
@pytest.mark.parametrize("branches", [1, 2])
@pytest.mark.parametrize("preceding_pulse", [False, True])
def test_join_table_lengths_start_together_and_wait_for_each_points_longest_branch(
    flat_top,
    branches,
    preceding_pulse,
):
    from zcu_tools.program.v2.modules.waveform import (
        FlatTopWaveformCfg,
        GaussWaveformCfg,
    )

    env = white_noise(std_adc=0, n_gens=3, routes=((0, 0), (1, 0), (2, 0)))
    first = drive()
    first.pre_delay = 0.02
    first.post_delay = 0.03
    second = drive().model_copy(update={"ch": 2})
    if flat_top:
        first.waveform = FlatTopWaveformCfg(
            length=0.2, raise_waveform=GaussWaveformCfg(length=0.04, sigma=0.01)
        )
    pulses = [
        TableLengthPulse("first", first, lengths=[0.1, 0.3], idx_reg="length"),
        TableLengthPulse("second", second, lengths=[0.2, 0.1], idx_reg="length"),
    ]
    modules: list[Module] = [
        Join(None, [None], *[[None, pulse, None] for pulse in pulses[:branches]]),
        receiver().build("readout"),
    ]
    if preceding_pulse:
        modules.insert(0, drive(length=0.04).build("reset"))
    try:
        prog = ModularProgramV2(env.soccfg, modules=modules, sweep=[("length", 2)])
    except RuntimeError as exc:
        pytest.fail(f"parallel table-length pulses must compile: {exc}")
    prog.acquire(env.soc, progress=False)
    request = request_of(env)
    origin = 0.04 if preceding_pulse else 0.0
    waits = [0.15, 0.35] if branches == 1 else [0.2, 0.35]
    for point, expected_wait in enumerate(waits):
        events = request.materialize_point(point).events
        actual_sources = [event.source for event in events]
        expected_sources = (
            (["reset"] if preceding_pulse else [])
            + ["first"] * (3 if flat_top else 1)
            + (["second"] if branches == 2 else [])
            + ["readout"]
        )
        assert actual_sources == expected_sources
        first_event = next(e for e in events if e.source == "first")
        readout = next(e for e in events if e.gen_ch == 0)
        assert time_us(request.time_base, first_event.start_ticks) == pytest.approx(
            origin + 0.02, abs=0.003
        )
        if branches == 2:
            second_event = next(e for e in events if e.gen_ch == 2)
            assert time_us(
                request.time_base, second_event.start_ticks
            ) == pytest.approx(origin, abs=0.003)
        assert time_us(request.time_base, readout.start_ticks) == pytest.approx(
            origin + expected_wait, abs=0.007
        )
    env.devices.close_all_devices()


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


def test_table_readout_wrap_conflicting_with_initial_template_is_rejected():
    env = white_noise(std_adc=0)
    gen, adc = readout_freq_words(
        env.soccfg, [200, 300], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    # The compiled template starts at cfg.freq=100. The last hook restores table[0]=200.
    readout = TablePulseReadout(
        "readout", receiver(), idx_reg="axis", freq_words=gen, ro_freq_words=adc
    )
    prog = ModularProgramV2(env.soccfg, modules=[readout], sweep=[("axis", 2)], reps=1)
    with pytest.raises(
        ValueError, match=r"MetaInst.*preopt.*axis.*table wrap disagrees"
    ):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


@pytest.mark.parametrize(
    ("kind", "count"), [("single", 1), ("parallel", 2), ("bath", 3)]
)
def test_reset_request_events_keep_authored_wrapper_name(kind, count):
    cavity = drive()
    qubit = drive().model_copy(update={"ch": 0})
    if kind == "single":
        cfg = PulseResetCfg(pulse_cfg=cavity)
    elif kind == "parallel":
        cfg = TwoPulseResetCfg(pulse1_cfg=cavity, pulse2_cfg=qubit)
    else:
        cfg = BathResetCfg(
            cavity_tone_cfg=cavity, qubit_tone_cfg=qubit, pi2_cfg=drive()
        )
    request, _, _ = acquire([cfg.build("authored_reset"), receiver().build("readout")])
    assert [event.source for event in request.materialize_point(0).events] == [
        *(["authored_reset"] * count),
        "readout",
    ]


def test_reset_wrappers_expand_only_actual_parallel_pulses():
    reset = TwoPulseResetCfg(
        pulse1_cfg=drive(0.1),
        pulse2_cfg=PulseCfg(
            ch=0, nqz=1, freq=60, gain=0.2, waveform=ConstWaveformCfg(length=0.2)
        ),
    ).build("reset")
    request, _, _ = acquire([None, reset, receiver().build("readout")])
    assert len(request.materialize_point(0).events) == 3
    assert [
        time_us(request.time_base, e.start_ticks)
        for e in request.materialize_point(0).events[:2]
    ] == [0, 0]
    assert time_us(
        request.time_base, request.materialize_point(0).events[-1].start_ticks
    ) == pytest.approx(0.2, abs=0.004)


def test_two_pulse_reset_swept_lengths_use_machine_blocking_end():
    env = white_noise(std_adc=0, n_gens=3, routes=((0, 0), (1, 0), (2, 0)))
    pulses = [
        PulseCfg(
            ch=ch,
            nqz=1,
            freq=80,
            gain=0.2,
            pre_delay=0.01,
            post_delay=0.02,
            waveform=ConstWaveformCfg(length=QickParam(0.05, {"length": 0.1})),
        )
        for ch in (1, 2)
    ]
    reset = TwoPulseResetCfg(pulse1_cfg=pulses[0], pulse2_cfg=pulses[1])
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.2),
        pre_delay=0.01,
        post_delay=0.02,
        trig_offset=0.03,
        ro_length=0.1,
    )
    try:
        prog = ModularProgramV2(
            env.soccfg,
            modules=[reset.build("reset"), readout.build("readout")],
            sweep=[("length", 3)],
        )
        prog.acquire(env.soc, progress=False)
        request = request_of(env)
        timelines = [request.materialize_point(point) for point in range(3)]
        # Machine scheduling emits literal tProc timestamps 84, 93 and 181.
        # The two swept waveforms remain 19, 38 and 57 generator clocks long.
        for point, timeline in enumerate(timelines):
            assert [event.source for event in timeline.events] == [
                "reset",
                "reset",
                "readout",
            ]
            assert [
                time_us(request.time_base, event.duration_ticks)
                for event in timeline.events[:2]
            ] == pytest.approx([([19, 38, 57][point] / 384)] * 2)
            assert time_us(
                request.time_base, timeline.events[-1].start_ticks
            ) == pytest.approx(84 / 430.08)
            assert times_us(
                request.time_base, timeline.readouts[0].trigger_ticks
            ) == pytest.approx((93 / 430.08,))
            assert time_us(request.time_base, timeline.duration_ticks) == pytest.approx(
                181 / 430.08
            )
    finally:
        env.devices.close_all_devices()


def test_reps_counter_lookup_in_join_rejected_before_backend():
    modules = [
        Join(
            LoadValue("writer", [0], "reps", "selection"),
            ComputedPulse("reader", val_reg="selection", pulses=[drive(), drive(0.2)]),
        ),
        receiver().build("readout"),
    ]
    env = white_noise(std_adc=0, n_gens=2, routes=((0, 0), (1, 0)))
    prog = ModularProgramV2(env.soccfg, modules=modules)
    with pytest.raises(
        ValueError, match=r"MetaInst.*preopt.*(?:reps shell|cross-shot.*s12)"
    ):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds


def test_reps_counter_word_lookup_rejected_before_backend():
    modules = [
        LoadWord("raw", [1], "reps", "cycles"),
        DelayAuto("invalid_wait", t="cycles"),
        receiver().build("readout"),
    ]
    env = white_noise(std_adc=0)
    prog = ModularProgramV2(env.soccfg, modules=modules)
    with pytest.raises(
        ValueError, match=r"MetaInst.*preopt.*(?:reps shell|cross-shot.*s12)"
    ):
        prog.acquire(env.soc, progress=False)
    assert not env.soc.binding.rounds
