"""Table and reset effects read from public compiled programs, not Module trees."""

import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import LoadValue, LoadWord, ScanWith
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.control import Branch, Repeat
from zcu_tools.program.v2.modules.delay import Join
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
from zcu_tools.program.v2.timeline import TimelineProgramV2
from zcu_tools.program.v2.utils import readout_freq_words

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


def module_program(
    modules, *, native=True, sweep=(), reps=2, body=None, data_registers=16
):
    env = white_noise(std_adc=0, n_gens=3, routes=((0, 0),))
    env.soccfg["tprocs"][0]["dreg_qty"] = data_registers
    if not native:
        return ModularProgramV2(
            env.soccfg,
            modules=modules,
            sweep=list(sweep),
            reps=reps,
            initial_delay=0,
            relax_delay=0.1,
        ), env

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            for name, count in sweep:
                self.add_loop(name, count)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            if body is None:
                for module in modules:
                    self.emit(module)
            else:
                body(self)

    return Program(env.soccfg, reps=reps, initial_delay=0, relax_delay=0.1), env


def test_modular_shared_readout_cfg_keeps_each_semantic_source():
    cfg = receiver()
    prog, env = module_program(
        [cfg.build("herald"), cfg.build("terminal")], native=False
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
        assert time_us(observed.time_base, event.duration_ticks) == 38 / 384
        assert event.frequency_mhz == pytest.approx(80, abs=1e-5)
        assert event.gain == pytest.approx(0.2, abs=1e-4)
    assert [time_us(observed.time_base, first.start_ticks), time_us(observed.time_base, second.start_ticks)] == [0, 43 / 430.08]
    assert len(times_us(observed.time_base, point.readouts[0].trigger_ticks)) == 1


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_loadvalue_selects_compiled_waveform_from_declared_table(native, compressed):
    loader = LoadValue(
        "select", [1, 0] * 16, "scan", "selected", auto_compress=compressed
    )
    computed = ComputedPulse(
        "choice", val_reg="selected", pulses=[drive(0.1, 0.2), drive(0.3, -0.4)]
    )
    prog, env = module_program(
        [loader, computed, receiver().build("readout")],
        native=native,
        sweep=(("scan", 32),),
    )
    result = request(prog, env)
    assert result.point_dims == (32,)
    for index in range(32):
        event = result.materialize_point(index).events[0]
        assert event.gain == pytest.approx(-0.4 if index % 2 == 0 else 0.2, abs=1e-4)
        assert time_us(result.time_base, event.duration_ticks) == pytest.approx((115 if index % 2 == 0 else 38) / 384)


@pytest.mark.parametrize("native", [False, True])
def test_tablelength_places_next_emission_after_each_selected_quantized_length(native):
    table = TableLengthPulse("length", drive(), lengths=[0.1, 0.3], idx_reg="scan")
    prog, env = module_program(
        [table, receiver().build("readout")], native=native, sweep=(("scan", 2),)
    )
    result = request(prog, env)
    # Legacy lowering keeps its existing duration-table timestamp policy.
    ends = [43, 129] if native else [44, 130]
    for index, (duration, end) in enumerate(zip([38, 115], ends, strict=True)):
        events = result.materialize_point(index).events
        assert events[0].source == "length"
        assert time_us(result.time_base, events[0].duration_ticks) == pytest.approx(duration / 384)
        assert time_us(result.time_base, events[1].start_ticks) == pytest.approx(end / 430.08)


@pytest.mark.parametrize("native", [False, True])
def test_loadword_preserves_uint32_patterns_for_pulse_and_adc_consumers(native):
    gen, adc = readout_freq_words(
        make_mock_soccfg(), [100, 200], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
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
    prog, env = module_program(modules, native=native, sweep=(("scan", 2),))
    result = request(prog, env)
    for index, (frequency, gain) in enumerate([(100, 3276), (200, -6552)]):
        point = result.materialize_point(index)
        assert point.events[0].source == "readout"
        assert point.events[0].frequency_mhz == pytest.approx(frequency, abs=1e-4)
        assert point.events[0].gain == pytest.approx(gain / 32766)
        assert point.readouts[0].frequency_mhz == pytest.approx((frequency,), abs=1e-4)


@pytest.mark.parametrize("native", [False, True])
def test_table_readout_exec_after_updates_and_wrap_survive_outer_reps(native):
    gen, adc = readout_freq_words(
        make_mock_soccfg(), [100, 250, 200], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    table = TablePulseReadout(
        "readout", receiver(), idx_reg="scan", freq_words=gen, ro_freq_words=adc
    )
    prog, env = module_program([table], native=native, sweep=(("scan", 3),), reps=3)
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


@pytest.mark.parametrize("native", [False, True])
def test_scanwith_uses_local_counter_to_select_each_candidate(native):
    choice = ComputedPulse(
        "choice",
        val_reg="selected",
        pulses=[drive(gain=0.1), drive(gain=0.2), drive(gain=0.3)],
    )
    scan = ScanWith("local", [2, 0, 1], val_reg="selected").add_content(choice)
    prog, env = module_program([scan, receiver().build("readout")], native=native)
    result = request(prog, env)
    events = result.materialize_point(0).events
    assert [e.gain for e in events[:3]] == pytest.approx([0.3, 0.1, 0.2], abs=1e-4)
    assert result.point_dims == ()
    assert [e.iterations[0][1] for e in events[:3]] == [0, 1, 2]


@pytest.mark.parametrize("native", [False, True])
def test_join_preserves_readout_owner_and_author_pulse_child(native):
    joined = Join(receiver().build("readout"), drive(ch=1).build("author_child"))
    prog, env = module_program([joined], native=native)
    point = request(prog, env).materialize_point(0)
    assert [event.source for event in point.events] == ["readout", "author_child"]
    assert [event.gen_ch for event in point.events] == [0, 1]
    assert point.readouts[0].generator_channels == (0,)


def test_legacy_table_join_compiled_request_keeps_each_authored_source():
    joined = Join(
        TableLengthPulse("left", drive(), lengths=[0.1, 0.2], idx_reg="scan"),
        TableLengthPulse("right", drive(ch=2), lengths=[0.2, 0.1], idx_reg="scan"),
    )
    prog, env = module_program(
        [joined, receiver().build("readout")], native=False, sweep=(("scan", 2),)
    )
    result = request(prog, env)
    for index in range(2):
        assert [event.source for event in result.materialize_point(index).events] == [
            "left",
            "right",
            "readout",
        ]


def test_legacy_branch_dispatch_table_and_repeat_preserve_selected_sources():
    choice = Branch(
        "scan",
        drive(gain=0.2).build("first"),
        Repeat("inner", 2).add_content(drive(gain=-0.4).build("second")),
    )
    prog, env = module_program(
        [choice, receiver().build("readout")], native=False, sweep=(("scan", 2),)
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


def test_legacy_branch_uses_loaded_selector_not_acquisition_lane_order():
    loader = LoadValue("selector", [2, 0, 1], "scan", "selected")
    choice = Branch(
        "choice",
        drive(gain=0.1).build("first"),
        drive(gain=0.2).build("second"),
        drive(gain=0.3).build("third"),
        compare_by="selected",
    )
    prog, env = module_program(
        [loader, choice, receiver().build("readout")],
        native=False,
        sweep=(("scan", 3),),
    )
    result = request(prog, env)
    assert [result.materialize_point(i).events[0].source for i in range(3)] == [
        "third",
        "first",
        "second",
    ]
    assert [
        result.materialize_point(i).events[0].gain for i in range(3)
    ] == pytest.approx([0.3, 0.1, 0.2], abs=1e-4)


@pytest.mark.parametrize(
    ("kind", "count"), [("single", 1), ("parallel", 2), ("bath", 3)]
)
def test_legacy_reset_compiled_events_keep_wrapper_owner(kind, count):
    cavity, qubit = drive(ch=1), drive(ch=2)
    if kind == "single":
        reset = PulseResetCfg(pulse_cfg=cavity)
    elif kind == "parallel":
        reset = TwoPulseResetCfg(pulse1_cfg=cavity, pulse2_cfg=qubit)
    else:
        reset = BathResetCfg(
            cavity_tone_cfg=cavity, qubit_tone_cfg=qubit, pi2_cfg=drive(ch=1)
        )
    prog, env = module_program(
        [reset.build("authored_reset"), receiver().build("readout")], native=False
    )
    events = request(prog, env).materialize_point(0).events
    assert [event.source for event in events] == ["authored_reset"] * count + [
        "readout"
    ]


@pytest.mark.parametrize("kind", ["single", "parallel", "bath"])
def test_native_reset_forms_expand_physical_pulses_and_paired_post_delays(kind):
    cavity = drive(0.1, ch=1, post=0.02)
    qubit = drive(0.3, ch=2)
    if kind == "single":
        reset = PulseResetCfg(pulse_cfg=cavity).build("reset")
        expected = [(1, 0)]
        readout_start = 52
    elif kind == "parallel":
        reset = TwoPulseResetCfg(pulse1_cfg=cavity, pulse2_cfg=qubit).build("reset")
        expected = [(1, 0), (2, 0)]
        readout_start = 129
    else:
        reset = BathResetCfg(
            cavity_tone_cfg=cavity, qubit_tone_cfg=qubit, pi2_cfg=drive(0.05, ch=1)
        ).build("reset")
        expected = [(1, 0), (2, 0), (1, 52)]
        readout_start = 129
    prog, env = module_program([reset, receiver().build("readout")])
    observed = request(prog, env)
    events = observed.materialize_point(0).events
    assert [e.gen_ch for e in events[:-1]] == [ch for ch, _ in expected]
    assert [time_us(observed.time_base, e.start_ticks) for e in events[:-1]] == pytest.approx(
        [t / 430.08 for _, t in expected]
    )
    assert time_us(observed.time_base, events[-1].start_ticks) == pytest.approx(readout_start / 430.08)


@pytest.mark.parametrize("nested", [False, True])
def test_native_module_branch_preserves_case_order_empty_paths_and_repeated_emissions(
    nested,
):
    short = drive(25 / 384).build("short")
    long = drive(50 / 384, ch=2).build("long")
    choice = Branch("choice", [short, None, long], None, short, compare_by="scan")
    if nested:
        choice = Branch("outer", choice, None, short, compare_by="scan")
    prog, env = module_program(
        [choice, choice, receiver().build("readout")], sweep=(("scan", 3),)
    )
    result = request(prog, env)
    assert result.point_names == ("scan",)
    for index, (sources, times) in enumerate(
        [
            (["short", "long", "short", "long", "readout"], [0, 28, 84, 112, 168]),
            (["readout"], [0]),
            (["short", "short", "readout"], [0, 28, 56]),
        ]
    ):
        point = result.materialize_point(index)
        assert [e.source for e in point.events] == sources
        assert [time_us(result.time_base, e.start_ticks) for e in point.events] == pytest.approx(
            [t / 430.08 for t in times]
        )
        assert times_us(result.time_base, point.readouts[0].trigger_ticks) == pytest.approx((times[-1] / 430.08,))
    prog.compile()
    rebuilt = request(prog, env)
    assert [rebuilt.materialize_point(i) for i in range(3)] == [
        result.materialize_point(i) for i in range(3)
    ]


@pytest.mark.parametrize("bookkeeping", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_parallel_modules_emit_children_but_program_join_only_merges_runtime_ends(
    bookkeeping, nested
):
    left = drive(38 / 384).build("left")
    right = drive(77 / 384, ch=2).build("right")
    joined = Join([left, left], right)
    if nested:
        joined = Join(None, joined)
    origin = LoadValue("origin", [7, 20, 10], "scan", "origin_cycles")
    readout = receiver().build("readout")

    def body(p):
        p.emit(origin)
        p.fork("default", into="parallel")
        p.delay_timeline("origin_cycles", timeline="parallel")
        for occurrence in range(2):
            if bookkeeping:
                # Only the emits output waves. Fork/join manipulate timeline ends.
                left_time, right_time = f"left_{occurrence}", f"right_{occurrence}"
                p.fork("parallel", into=[left_time, right_time])
                p.emit(left, timeline=left_time)
                p.emit(left, timeline=left_time)
                p.emit(right, timeline=right_time)
                p.join(left_time, right_time, into="parallel")
            else:
                p.emit(joined, timeline="parallel")
        p.join("parallel", into="default")
        p.emit(readout)

    prog, env = module_program(
        [origin, joined, readout], sweep=(("scan", 3),), body=body, data_registers=32
    )
    result = request(prog, env)
    for index, start in enumerate([7, 20, 10]):
        point = result.materialize_point(index)
        assert [e.source for e in point.events] == ["left", "left", "right"] * 2 + [
            "readout"
        ]
        assert [time_us(result.time_base, e.start_ticks) for e in point.events] == pytest.approx(
            [(start + t) / 430.08 for t in [0, 43, 0, 87, 130, 87, 174]]
        )
        assert times_us(result.time_base, point.readouts[0].trigger_ticks) == pytest.approx(((start + 174) / 430.08,))


@pytest.mark.parametrize("nested", [False, True])
def test_parallel_exchange_preserves_child_sources_and_complete_scan_sequences(nested):
    first_cfg = drive(25 / 384)
    first_cfg.pre_delay = 7 / 430.08
    first = first_cfg.build("first")
    scan = ScanWith("local", [7, 9], "value").add_content(
        drive(25 / 384).build("scan_pulse")
    )
    other_cfg = drive(50 / 384, ch=2)
    other_cfg.pre_delay = 17 / 430.08
    other = other_cfg.build("other")
    advancing = Join([first, scan]) if nested else [first, scan]
    joined = Join(advancing, other)
    readout = receiver().build("readout")

    def body(p):
        p.emit(joined, tag="first_group")
        p.delay_timeline(11 / 430.08)
        p.emit(joined, tag="second_group")
        p.emit(readout)

    prog, env = module_program([joined, readout], body=body, data_registers=32)
    observed = request(prog, env)
    point = observed.materialize_point(0)
    # The advancing author-first sequence emits last, without moving its origin.
    assert [e.source for e in point.events] == [
        "other",
        "first",
        "scan_pulse",
        "scan_pulse",
    ] * 2 + ["readout"]
    assert [time_us(observed.time_base, e.start_ticks) for e in point.events] == pytest.approx(
        [t / 430.08 for t in [17, 7, 35, 63, 119, 109, 137, 165, 193]]
    )
    assert [point.events[i].iterations[-1][1] for i in [2, 3, 6, 7]] == [0, 1, 0, 1]
    assert times_us(observed.time_base, point.readouts[0].trigger_ticks) == pytest.approx((193 / 430.08,))


def test_parallel_table_lengths_keep_crossing_maxima_across_repeated_runtime_emissions():
    left = TableLengthPulse(
        "left", drive(), lengths=[25 / 384, 50 / 384, 75 / 384], idx_reg="scan"
    )
    right = TableLengthPulse(
        "right", drive(ch=2), lengths=[75 / 384, 50 / 384, 25 / 384], idx_reg="scan"
    )
    joined = Join(left, right)
    origin = LoadValue("origin", [7, 20, 10], "scan", "origin_cycles")
    readout = receiver().build("readout")

    def body(p):
        p.emit(origin)
        p.delay_timeline("origin_cycles")
        p.emit(joined)
        p.emit(joined)
        p.emit(readout)

    prog, env = module_program(
        [origin, joined, readout], sweep=(("scan", 3),), body=body, data_registers=32
    )
    result = request(prog, env)
    for index, (origin_cycle, maximum) in enumerate([(7, 84), (20, 56), (10, 84)]):
        point = result.materialize_point(index)
        assert [e.source for e in point.events] == ["left", "right"] * 2 + ["readout"]
        assert [time_us(result.time_base, e.start_ticks) for e in point.events] == pytest.approx(
            [(origin_cycle + t) / 430.08 for t in [0, 0, maximum, maximum, 2 * maximum]]
        )
        assert [time_us(result.time_base, e.duration_ticks) for e in point.events[:2]] == pytest.approx(
            [(25 + 25 * index) / 384, (75 - 25 * index) / 384]
        )
        assert times_us(result.time_base, point.readouts[0].trigger_ticks) == pytest.approx(
            ((origin_cycle + 2 * maximum) / 430.08,)
        )


@pytest.mark.parametrize("nested", [False, True])
def test_parallel_scanwith_inside_repeat_keeps_selected_words_coordinates_and_runtime_debits(
    nested,
):
    selected = ComputedPulse(
        "choice",
        val_reg="selected",
        pulses=[drive(25 / 384, 0.1), drive(25 / 384, 0.3)],
    )
    scan = ScanWith("local", [1, 0], "selected").add_content(selected)
    other = drive(50 / 384, ch=2).build("other")
    # Keep the reference-advancing sequence last; selected playback's resource
    # effects are not admitted for exchange by the producer.
    joined = Join(other, Join(scan) if nested else scan)
    origin = LoadValue("origin", [7, 11, 19], "scan", "origin_cycles")
    readout = receiver().build("readout")

    def body(p):
        p.emit(origin)
        p.fork("default", into="external")
        p.delay_timeline(200 / 430.08, timeline="external")
        p.delay_timeline("origin_cycles")
        with p.repeat("outer", 2):
            p.emit(joined)
        p.join("external", into="default")
        p.emit(readout)

    prog, env = module_program(
        [origin, joined, readout], sweep=(("scan", 3),), body=body, data_registers=32
    )
    result = request(prog, env)
    assert result.point_dims == (3,)
    for index, start in enumerate([7, 11, 19]):
        events = result.materialize_point(index).events
        assert [e.source for e in events] == ["other", "choice", "choice"] * 2 + [
            "readout"
        ]
        assert [time_us(result.time_base, e.start_ticks) for e in events] == pytest.approx(
            [(start + t) / 430.08 for t in [0, 0, 28, 56, 56, 84]] + [200 / 430.08]
        )
        assert [e.gain for e in events[:-1]] == pytest.approx(
            [0.2, 0.3, 0.1] * 2, abs=1e-4
        )
        assert [e.iterations[0] for e in events[:-1]] == [("outer", 0)] * 3 + [
            ("outer", 1)
        ] * 3
        assert [events[i].iterations[-1][1] for i in [1, 2, 4, 5]] == [0, 1, 0, 1]
        assert events[-1].iterations == ()
