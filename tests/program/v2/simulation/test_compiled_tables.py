"""Table and reset effects read from public compiled programs, not Module trees."""

from fractions import Fraction
from math import ceil

import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules import Delay, LoadValue, LoadWord
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg, TableLengthPulse
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

from tests.program._profiles import GEN_MAXV, GEN_MHZ, TPROC_MHZ, load_soccfg
from tests.program._timing_expectations import (
    generator_duration,
    generator_tproc_grid,
    runtime_cycles,
    tproc_ceil,
)
from tests.program.v2.simulation._compiled_support import request
from tests.program.v2.simulation.time_observations import time_us, times_us


def drive(length=0.1, gain=0.2, *, ch=1, post: float = 0):
    return PulseCfg(
        ch=ch,
        nqz=1,
        freq=80,
        gain=gain,
        waveform=ConstWaveformCfg(length=length),
        post_delay=post,
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


def module_program(modules, *, sweep=(), reps=2, body=None, data_registers=16):
    env = white_noise(load_soccfg(n_gens=3), std_adc=0, routes=((0, 0),), seed=None)
    env.soccfg["tprocs"][0]["dreg_qty"] = data_registers
    builder = TimelineBuilder(env.soccfg, reps=reps, initial_delay=0, relax_delay=0.1)
    for name, count in sweep:
        builder.add_loop(name, count)
    if body is None:
        for module in modules:
            builder.emit(module)
    else:
        body(builder)
    builder.delay_by_timeline()
    return builder.build(), env


def test_shared_readout_cfg_keeps_each_semantic_source():
    cfg = receiver()
    # ADC and generator durations stay on their own quantized clock grids.
    prog, env = module_program(
        [cfg.build("herald"), Delay("adc_gap", 1 / TPROC_MHZ), cfg.build("terminal")]
    )
    observed = request(prog, env)
    point = observed.materialize_point(0)
    assert [event.source for event in point.events] == ["herald", "terminal"]
    assert [event.gen_ch for event in point.events] == [0, 0]
    assert len(times_us(observed.time_base, point.readouts[0].trigger_ticks)) == 2


def test_native_shared_pulse_cfg_keeps_each_semantic_source_and_playback():
    cfg = drive()
    prog, env = module_program(
        [Pulse("first", cfg), Pulse("second", cfg), receiver().build("readout")]
    )
    observed = request(prog, env)
    point = observed.materialize_point(0)
    assert [event.source for event in point.events] == ["first", "second", "readout"]
    assert [event.gen_ch for event in point.events] == [1, 1, 0]
    first, second, _ = point.events
    for event in (first, second):
        assert time_us(observed.time_base, event.duration_ticks) == float(
            generator_duration(prog.soccfg, 0.1, 1)
        )
        assert event.frequency_mhz == pytest.approx(80, abs=1e-5)
        assert event.gain == pytest.approx(0.2, abs=1e-4)
    assert [
        time_us(observed.time_base, first.start_ticks),
        time_us(observed.time_base, second.start_ticks),
    ] == [
        0,
        tproc_ceil(prog.soccfg, generator_duration(prog.soccfg, 0.1, 1)) / TPROC_MHZ,
    ]
    assert len(times_us(observed.time_base, point.readouts[0].trigger_ticks)) == 1


@pytest.mark.parametrize("compressed", [False, True])
def test_loadvalue_selects_compiled_waveform_from_declared_table(compressed):
    loader = LoadValue(
        "select", [1, 0] * 16, "scan", "selected", auto_compress=compressed
    )
    computed = ComputedPulse(
        "choice", val_reg="selected", pulses=[drive(0.1, 0.2), drive(0.3, -0.4)]
    )
    prog, env = module_program(
        [loader, computed, receiver().build("readout")],
        sweep=(("scan", 32),),
    )
    result = request(prog, env)
    assert result.point_dims == (32,)
    for index in range(32):
        event = result.materialize_point(index).events[0]
        assert event.gain == pytest.approx(-0.4 if index % 2 == 0 else 0.2, abs=1e-4)
        assert time_us(result.time_base, event.duration_ticks) == pytest.approx(
            float(generator_duration(prog.soccfg, 0.3 if index % 2 == 0 else 0.1, 1))
        )


def test_tablelength_places_next_emission_after_each_selected_quantized_length():
    table = TableLengthPulse("length", drive(), lengths=[0.1, 0.3], idx_reg="scan")
    prog, env = module_program(
        [table, receiver().build("readout")], sweep=(("scan", 2),)
    )
    result = request(prog, env)
    ends = [
        ceil(runtime_cycles(prog.soccfg, int(prog.us2cycles(length, gen_ch=1)), 1))
        for length in (0.1, 0.3)
    ]
    for index, (length, end) in enumerate(zip((0.1, 0.3), ends, strict=True)):
        events = result.materialize_point(index).events
        assert events[0].source == "length"
        assert time_us(result.time_base, events[0].duration_ticks) == pytest.approx(
            float(generator_duration(prog.soccfg, length, 1))
        )
        assert time_us(result.time_base, events[1].start_ticks) == pytest.approx(
            end / TPROC_MHZ
        )


def test_loadword_preserves_uint32_patterns_for_pulse_and_adc_consumers():
    gen, adc = readout_freq_words(
        load_soccfg(), [100, 200], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    modules = [
        LoadWord("frequency", gen, "scan", "freq_word"),
        LoadWord("adc", adc, "scan", "adc_word"),
        LoadWord("gain", [3276, (-6552) % 2**32], "scan", "gain_word"),
        PulseReadout(
            "readout",
            receiver(),
            freq_val="freq_word",
            ro_freq_val="adc_word",
            gain_val="gain_word",
        ),
    ]
    prog, env = module_program(modules, sweep=(("scan", 2),))
    result = request(prog, env)
    for index, (frequency, gain) in enumerate([(100, 3276), (200, -6552)]):
        point = result.materialize_point(index)
        assert point.events[0].source == "readout"
        assert point.events[0].frequency_mhz == pytest.approx(frequency, abs=1e-4)
        assert point.events[0].gain == pytest.approx(gain / GEN_MAXV)
        assert point.readouts[0].frequency_mhz == pytest.approx((frequency,), abs=1e-4)


def test_table_readout_exec_after_updates_and_wrap_survive_outer_reps():
    gen, adc = readout_freq_words(
        load_soccfg(), [100, 250, 200], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    table = TablePulseReadout(
        "readout", receiver(), idx_reg="scan", freq_words=gen, ro_freq_words=adc
    )
    prog, env = module_program([table], sweep=(("scan", 3),), reps=3)
    result = request(prog, env)
    assert [result.materialize_point(i).events[0].source for i in range(3)] == [
        "readout",
        "readout",
        "readout",
    ]
    assert [
        result.materialize_point(i).events[0].frequency_mhz for i in range(3)
    ] == pytest.approx([100, 250, 200], abs=1e-4)
    assert [
        result.materialize_point(i).readouts[0].frequency_mhz[0] for i in range(3)
    ] == pytest.approx([100, 250, 200], abs=1e-4)
    assert request(prog, env).materialize_point(0) == result.materialize_point(0)


def test_join_preserves_readout_owner_and_author_pulse_child():
    readout = receiver().build("readout")
    child = drive(ch=1).build("author_child")

    def body(p):
        p.fork("default", into="child")
        p.emit(readout)
        p.emit(child, timeline="child")
        p.join("default", "child", into="default")

    prog, env = module_program([readout, child], body=body)
    point = request(prog, env).materialize_point(0)
    assert [event.source for event in point.events] == ["readout", "author_child"]
    assert [event.gen_ch for event in point.events] == [0, 1]
    assert point.readouts[0].generator_channels == (0,)


def test_branch_dispatch_and_repeat_preserve_selected_sources():
    first = drive(gain=0.2).build("first")
    second = drive(gain=-0.4).build("second")
    readout = receiver().build("readout")

    def body(p):
        with p.branch(by="scan", align_timeline=True) as cases:
            with cases.case(0):
                p.emit(first)
            with cases.case(1), p.repeat("inner", 2):
                p.emit(second)
        p.emit(readout)

    prog, env = module_program(
        [first, second, readout], sweep=(("scan", 2),), body=body
    )
    result = request(prog, env)
    assert [e.source for e in result.materialize_point(0).events] == [
        "first",
        "readout",
    ]
    assert [e.source for e in result.materialize_point(1).events] == [
        "second",
        "second",
        "readout",
    ]
    assert [e.iterations for e in result.materialize_point(1).events[:2]] == [
        (("inner", 0),),
        (("inner", 1),),
    ]


@pytest.mark.parametrize("kind", ["single", "parallel", "bath"])
def test_native_reset_forms_expand_physical_pulses_and_paired_post_delays(kind):
    profile = load_soccfg(n_gens=3)
    cavity_end = tproc_ceil(
        profile, generator_duration(profile, 0.1, 1) + Fraction("0.02")
    )
    qubit_end = tproc_ceil(profile, generator_duration(profile, 0.3, 2))
    pi2_end = tproc_ceil(profile, generator_duration(profile, 0.05, 1))
    cavity = drive(0.1, ch=1, post=0.02)
    qubit = drive(0.3, ch=2)
    if kind == "single":
        reset = PulseResetCfg(pulse_cfg=cavity).build("reset")
        expected = [(1, 0)]
        readout_start = cavity_end
    elif kind == "parallel":
        reset = TwoPulseResetCfg(pulse1_cfg=cavity, pulse2_cfg=qubit).build("reset")
        expected = [(1, 0), (2, 0)]
        readout_start = max(cavity_end, qubit_end)
    else:
        reset = BathResetCfg(
            cavity_tone_cfg=cavity, qubit_tone_cfg=qubit, pi2_cfg=drive(0.05, ch=1)
        ).build("reset")
        expected = [(1, 0), (2, 0), (1, cavity_end)]
        readout_start = max(qubit_end, cavity_end + pi2_end)
    prog, env = module_program([reset, receiver().build("readout")])
    observed = request(prog, env)
    events = observed.materialize_point(0).events
    assert [e.gen_ch for e in events[:-1]] == [ch for ch, _ in expected]
    assert [
        time_us(observed.time_base, e.start_ticks) for e in events[:-1]
    ] == pytest.approx([t / TPROC_MHZ for _, t in expected])
    assert time_us(observed.time_base, events[-1].start_ticks) == pytest.approx(
        readout_start / TPROC_MHZ
    )


@pytest.mark.parametrize("nested", [False, True])
def test_native_module_branch_preserves_case_order_empty_paths_and_repeated_emissions(
    nested,
):
    short = drive(25 / GEN_MHZ).build("short")
    long = drive(50 / GEN_MHZ, ch=2).build("long")
    readout = receiver().build("readout")

    def choose(p):
        with p.branch(by="scan", align_timeline=True) as cases:
            with cases.case(0):
                p.emit(short)
                p.emit(long)
            with cases.case(2):
                p.emit(short)

    def body(p):
        for _ in range(2):
            if nested:
                with p.branch(by="scan", align_timeline=True) as cases:
                    with cases.case(0):
                        choose(p)
                    with cases.case(2):
                        p.emit(short)
            else:
                choose(p)
        p.emit(readout)

    prog, env = module_program([short, long, readout], sweep=(("scan", 3),), body=body)
    result = request(prog, env)
    short_end = tproc_ceil(
        prog.soccfg, generator_duration(prog.soccfg, 25 / GEN_MHZ, 1)
    )
    long_end = tproc_ceil(prog.soccfg, generator_duration(prog.soccfg, 50 / GEN_MHZ, 2))
    branch_end = short_end + long_end
    assert result.point_names == ("scan",)
    for index, (sources, times) in enumerate(
        [
            (
                ["short", "long", "short", "long", "readout"],
                [0, short_end, branch_end, branch_end + short_end, 2 * branch_end],
            ),
            # Both emissions account max(84,0,28)=84, including nested Branch.
            (["readout"], [2 * branch_end]),
            (["short", "short", "readout"], [0, branch_end, 2 * branch_end]),
        ]
    ):
        point = result.materialize_point(index)
        assert [e.source for e in point.events] == sources
        assert [
            time_us(result.time_base, e.start_ticks) for e in point.events
        ] == pytest.approx([t / TPROC_MHZ for t in times])
        assert times_us(
            result.time_base, point.readouts[0].trigger_ticks
        ) == pytest.approx((times[-1] / TPROC_MHZ,))
    prog.compile()
    rebuilt = request(prog, env)
    assert [rebuilt.materialize_point(i) for i in range(3)] == [
        result.materialize_point(i) for i in range(3)
    ]


def test_program_join_only_merges_runtime_ends():
    left = drive(38 / GEN_MHZ).build("left")
    right = drive(77 / GEN_MHZ, ch=2).build("right")
    origin = LoadValue("origin", [7, 20, 10], "scan", "origin_cycles")
    readout = receiver().build("readout")

    def body(p):
        p.emit(origin)
        p.fork("default", into="parallel")
        p.timeline_inc("origin_cycles", timeline="parallel")
        for occurrence in range(2):
            left_time, right_time = f"left_{occurrence}", f"right_{occurrence}"
            p.fork("parallel", into=[left_time, right_time])
            p.emit(left, timeline=left_time)
            p.emit(left, timeline=left_time)
            p.emit(right, timeline=right_time)
            p.join(left_time, right_time, into="parallel")
        p.join("parallel", into="default")
        p.emit(readout)

    prog, env = module_program(
        [origin, left, right, readout],
        sweep=(("scan", 3),),
        body=body,
        data_registers=32,
    )
    result = request(prog, env)
    left_end = tproc_ceil(prog.soccfg, generator_duration(prog.soccfg, 38 / GEN_MHZ, 1))
    right_end = tproc_ceil(
        prog.soccfg, generator_duration(prog.soccfg, 77 / GEN_MHZ, 2)
    )
    joined = max(2 * left_end, right_end)
    for index, start in enumerate([7, 20, 10]):
        point = result.materialize_point(index)
        assert [e.source for e in point.events] == ["left", "left", "right"] * 2 + [
            "readout"
        ]
        assert [
            time_us(result.time_base, e.start_ticks) for e in point.events
        ] == pytest.approx(
            [
                (start + t) / TPROC_MHZ
                for t in [0, left_end, 0, joined, joined + left_end, joined, 2 * joined]
            ]
        )
        assert times_us(
            result.time_base, point.readouts[0].trigger_ticks
        ) == pytest.approx(((start + 2 * joined) / TPROC_MHZ,))


def test_parallel_table_lengths_keep_crossing_maxima_across_repeated_runtime_emissions():
    # Keep exact integer-affine candidates on any caller-owned clock ratio.
    step = generator_tproc_grid(load_soccfg(3, 1), 1)
    left_lengths = [n * step / GEN_MHZ for n in (1, 2, 3)]
    right_lengths = list(reversed(left_lengths))
    left = TableLengthPulse(
        "left",
        drive(),
        lengths=left_lengths,
        idx_reg="scan",
    )
    right = TableLengthPulse(
        "right",
        drive(ch=2),
        lengths=right_lengths,
        idx_reg="scan",
    )
    origin = LoadValue("origin", [7, 20, 10], "scan", "origin_cycles")
    readout = receiver().build("readout")

    def body(p):
        p.emit(origin)
        p.timeline_inc("origin_cycles")
        for occurrence in range(2):
            other = f"right_{occurrence}"
            p.fork("default", into=other)
            p.emit(left)
            p.emit(right, timeline=other)
            p.join("default", other, into="default")
        p.emit(readout)

    prog, env = module_program(
        [origin, left, right, readout],
        sweep=(("scan", 3),),
        body=body,
        data_registers=32,
    )
    result = request(prog, env)
    # These table lengths are integer-affine on the tProc grid, so remain exact.
    maxima = [
        max(
            tproc_ceil(prog.soccfg, generator_duration(prog.soccfg, left_length, 1)),
            tproc_ceil(prog.soccfg, generator_duration(prog.soccfg, right_length, 2)),
        )
        for left_length, right_length in zip(left_lengths, right_lengths, strict=True)
    ]
    for index, (origin_cycle, maximum) in enumerate(
        zip((7, 20, 10), maxima, strict=True)
    ):
        point = result.materialize_point(index)
        assert [e.source for e in point.events] == ["left", "right"] * 2 + ["readout"]
        assert [
            time_us(result.time_base, e.start_ticks) for e in point.events
        ] == pytest.approx(
            [
                (origin_cycle + t) / TPROC_MHZ
                for t in [0, 0, maximum, maximum, 2 * maximum]
            ]
        )
        assert [
            time_us(result.time_base, e.duration_ticks) for e in point.events[:2]
        ] == pytest.approx(
            [
                float(generator_duration(prog.soccfg, left_lengths[index], 1)),
                float(generator_duration(prog.soccfg, right_lengths[index], 2)),
            ]
        )
        assert times_us(
            result.time_base, point.readouts[0].trigger_ticks
        ) == pytest.approx(((origin_cycle + 2 * maximum) / TPROC_MHZ,))
