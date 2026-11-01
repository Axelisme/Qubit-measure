"""Module emissions are measured at the public compiler/analysis seam."""

from collections.abc import Callable, Sequence
from fractions import Fraction

import pytest
from qick.asm_v2 import Macro, QickParam, QickSweep1D, WriteReg
from zcu_tools.program.v2.modules import Module, Pulse, PulseCfg
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, SoftDelay
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg, CosineWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import GEN_MHZ, TPROC_MHZ, load_soccfg
from tests.program._timing_expectations import generator_duration, native_sweep, readout_duration, tproc_ceil

from tests.program.v2.timing_support import shots as _shots
from tests.program.v2.timing_support import waves as _waves

PROFILE = load_soccfg(3, 1)
PULSE_END = tproc_ceil(PROFILE, generator_duration(PROFILE, 0.1))
READ_PRE = tproc_ceil(PROFILE, 0.05)
READ_TRIGGER = tproc_ceil(PROFILE, 0.1)
READ_END = READ_PRE + tproc_ceil(PROFILE, generator_duration(PROFILE, 0.3, 1) + Fraction('0.02'))
READ_SHELL = round(READ_TRIGGER + readout_duration(PROFILE, 0.4) * TPROC_MHZ)


def _pulse(name="drive", *, ch=0, length: float | QickParam = 0.1, **kwargs):
    return Pulse(
        name,
        PulseCfg(
            ch=ch,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=length),
            **kwargs,
        ),
    )


def _program(
    modules: Sequence[Module],
    body: Callable[[TimelineProgramV2], None] | None = None,
    initialize: Callable[[TimelineProgramV2], None] | None = None,
):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            if initialize is not None:
                initialize(self)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            if body is None:
                for module in modules:
                    self.emit(module)
            else:
                body(self)

    return Program(load_soccfg(3, 1), reps=2, initial_delay=0, relax_delay=0)


def _envelope_shapes(prog: TimelineProgramV2) -> list[tuple[int, ...]]:
    envelopes = prog.envelopes[0]["envs"]
    if not isinstance(envelopes, dict):
        raise TypeError("QICK envelope declarations must be a mapping")
    return [envelope["data"].shape for envelope in envelopes.values()]


def test_same_named_distinct_pulses_keep_distinct_envelopes():
    def pulses(names):
        return [
            Pulse(
                name,
                PulseCfg(
                    ch=0,
                    nqz=1,
                    freq=1000,
                    gain=0.5,
                    waveform=CosineWaveformCfg(length=length),
                ),
            )
            for name, length in zip(names, (0.1, 0.2), strict=True)
        ]

    actual = _program(pulses(("same", "same")))
    fresh = _program(pulses(("first", "second")))
    assert actual.prog_list == fresh.prog_list
    assert len(_envelope_shapes(actual)) == 2
    assert _envelope_shapes(actual) == _envelope_shapes(fresh)


def test_shared_pulse_definitions_deduplicate_in_each_program():
    shared = [_pulse("first"), _pulse("second")]
    first = _program(shared)
    second = _program(shared)
    fresh = _program([_pulse("first"), _pulse("second")])
    assert second.prog_list == fresh.prog_list == first.prog_list
    first.compile()
    assert first.prog_list == fresh.prog_list


@pytest.mark.parametrize("timeline", [None, "drive"])
@pytest.mark.parametrize(
    "delays", [(0, 0), (0.01, 0.02)],
)
def test_reused_scalar_pulse_has_independent_tags_and_common_end(
    timeline, delays
):
    pre_delay, post_delay = delays
    pre = tproc_ceil(PROFILE, pre_delay)
    end = pre + tproc_ceil(PROFILE, generator_duration(PROFILE, 0.1) + Fraction(str(post_delay)))
    events = [pre, end + pre]
    duration = 2 * end
    pulse = _pulse(pre_delay=pre_delay, post_delay=post_delay)

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        prog.emit(pulse, timeline=timeline, tag="first")
        prog.emit(pulse, timeline=timeline, tag="second")

    prog = _program([pulse], body)
    # Each emission quantizes pre once and adds post to physical playback before ceil.
    assert [_waves(s) for s in _shots(prog)] == [events] * 2
    assert [s.body_end_cycle - s.start_cycle for s in _shots(prog)] == [duration] * 2
    assert prog.get_time_param("first", "t") * TPROC_MHZ == pytest.approx(events[0])
    assert prog.get_time_param("second", "t") * TPROC_MHZ == pytest.approx(events[1])
    expected = _shots(prog)
    prog.compile()
    assert _shots(prog) == expected


@pytest.mark.parametrize("swept", [False, True])
def test_joined_emissions_preserve_parallel_origins_and_event_end(swept):
    a = _pulse("a")
    # 25 generator clocks per step is exactly28 tProc cycles.
    b = _pulse(
        "b",
        ch=1,
        length=QickSweep1D("scan", 77 / GEN_MHZ, 127 / GEN_MHZ) if swept else 0.2,
    )
    tail = _pulse("tail", ch=2)

    def initialize(prog):
        if swept:
            prog.add_loop("scan", 3)

    def body(prog):
        prog.fork("default", into=["a", "b"])
        prog.emit(a, timeline="a")
        prog.emit(b, timeline="b")
        prog.join("a", "b", into="default")
        prog.emit(tail)

    shots = _shots(_program([a, b, tail], body, initialize))
    lengths = native_sweep(PROFILE, 77 / GEN_MHZ, 127 / GEN_MHZ, 3, gen_ch=1) if swept else [int(PROFILE.us2cycles(0.2, gen_ch=1))]
    branch_ends = [max(PULSE_END, tproc_ceil(PROFILE, generator_duration(PROFILE, n / GEN_MHZ, 1))) for n in lengths]
    expected = [[0, 0, end] for end in branch_ends]
    assert [_waves(s) for s in shots] == expected * 2
    ends = [end + PULSE_END for end in branch_ends]
    assert [s.body_end_cycle - s.start_cycle for s in shots] == ends * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == ends * 2


def test_repeat_emissions_keep_independent_tags_and_actual_end_spacing():
    pulse = _pulse()

    def body(prog):
        prog.emit(pulse, tag="before")
        prog.delay_by_timeline()
        with prog.repeat("r", 2):
            prog.emit(pulse, tag="inside")
            prog.delay_by_timeline()
        prog.emit(pulse, tag="after")

    prog = _program([pulse], body)
    shots = _shots(prog)
    assert [_waves(s) for s in shots] == [[i * PULSE_END for i in range(4)]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [4 * PULSE_END] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [4 * PULSE_END] * 2
    assert prog.get_time_param("inside", "t") * TPROC_MHZ == pytest.approx(0)


@pytest.mark.parametrize("delay_by_timeline", [False, True])
def test_default_emit_only_moves_end_until_sync(delay_by_timeline):
    pulse = _pulse()

    def body(prog):
        prog.emit(pulse)
        if delay_by_timeline:
            prog.delay_by_timeline()
        prog.pulse(0, pulse.pulse_name(prog))

    shots = _shots(_program([pulse], body))
    assert [_waves(s) for s in shots] == [[0, PULSE_END if delay_by_timeline else 0]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [PULSE_END] * 2


def test_shared_runtime_readout_keeps_each_programs_templates():
    shared = _readout(freq_val="frequency", gain_val="gain", ro_freq_val="adc_word")
    others = []

    def build(readout, *, padded=False, interleave=False):
        def initialize(prog):
            if padded:
                prog.add_reg("padding")
                prog.declare(_pulse("padding", ch=2, length=0.2))
            for register in ("frequency", "gain", "adc_word"):
                prog.add_reg(register)

        def body(prog):
            if interleave:
                others.append(build(shared, padded=True))
            prog.write_reg("frequency", 1000)
            prog.write_reg("gain", 100)
            prog.write_reg("adc_word", 2000)
            prog.emit(readout)

        return _program([readout], body, initialize)

    first = build(shared, interleave=True)
    fresh = build(
        _readout(freq_val="frequency", gain_val="gain", ro_freq_val="adc_word")
    )
    other_fresh = build(
        _readout(freq_val="frequency", gain_val="gain", ro_freq_val="adc_word"),
        padded=True,
    )
    assert first.prog_list == fresh.prog_list
    assert _shots(first) == _shots(fresh)
    assert others[0].prog_list == other_fresh.prog_list
    first.compile()
    assert first.prog_list == fresh.prog_list


def _readout(**kwargs):
    return PulseReadout(
        "read",
        PulseReadoutCfg(
            gen_ch=1,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.3),
            pre_delay=0.05,
            post_delay=0.02,
            ro_ch=0,
            ro_length=0.4,
            trig_offset=0.1,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("delay_type", [Delay, SoftDelay])
@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("timeline", [None, "drive"])
def test_t1_including_readout_uses_event_end_at_each_scan_point(
    delay_type, swept, timeline
):
    delay = QickSweep1D("wait", 0.2, 0.8) if swept else 0.2
    modules = [_pulse(), delay_type("wait", delay), _readout()]

    def initialize(prog):
        if swept:
            prog.add_loop("wait", 4)

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        for module in modules:
            prog.emit(module, timeline=timeline)

    shots = _shots(_program(modules, body, initialize))
    waits = native_sweep(PROFILE, 0.2, 0.8, 4) if swept else [tproc_ceil(PROFILE, 0.2)]
    offsets = [PULSE_END + wait for wait in waits]
    assert [_waves(s) for s in shots] == [
        [0, offset, offset + READ_PRE] for offset in offsets
    ] * 2
    assert [
        [(e.cycle - s.start_cycle, e.value) for e in s.events if e.kind == "data"]
        for s in shots
    ] == [[(offset + READ_TRIGGER, 1), (offset + READ_TRIGGER + 10, 0)] for offset in offsets] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [
        offset + READ_END for offset in offsets
    ] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [
        offset + READ_SHELL for offset in offsets
    ] * 2


@pytest.mark.parametrize("delay_type", [Delay, SoftDelay])
def test_delay_emit_does_not_move_reference_and_does_not_receil(delay_type):
    delay = delay_type("wait", 0.504)
    pulse = _pulse()

    def body(prog):
        prog.emit(delay)
        prog.pulse(0, pulse.pulse_name(prog))
        prog.delay_by_timeline()
        prog.pulse(0, pulse.pulse_name(prog))

    shots = _shots(_program([delay, pulse], body))
    end = tproc_ceil(PROFILE, 0.504)
    assert [_waves(s) for s in shots] == [[0, end]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [end] * 2


def test_wrong_pulse_time_is_detected_by_independent_timing_expectation():
    pulse = _pulse()

    def body(prog):
        prog.emit(pulse)
        prog.pulse(0, pulse.pulse_name(prog), t=1)

    shot = _shots(_program([pulse], body))[0]
    with pytest.raises(AssertionError):
        assert _waves(shot) == [0, 0]


@pytest.mark.parametrize(
    "kind", ["pulse", "two_pulse", "bath"],
)
@pytest.mark.parametrize("timeline", [None, "reset"])
def test_reset_composite_keeps_internal_overlap(kind, timeline):
    pre = tproc_ceil(PROFILE, 0.02)
    cavity_end = pre + tproc_ceil(PROFILE, generator_duration(PROFILE, 0.2) + Fraction('0.01'))
    qubit_end = pre + tproc_ceil(PROFILE, generator_duration(PROFILE, 0.1, 1) + Fraction('0.01'))
    pi2_end = pre + tproc_ceil(PROFILE, generator_duration(PROFILE, 0.05, 1) + Fraction('0.01'))
    reset_waves = [pre] if kind == 'pulse' else [pre, pre]
    length = cavity_end if kind == 'pulse' else max(cavity_end, qubit_end)
    if kind == 'bath':
        reset_waves += [cavity_end + pre]
        length = max(qubit_end, cavity_end + pi2_end)
    def cfg(ch, duration):
        return PulseCfg(
            ch=ch,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=duration),
            pre_delay=0.02,
            post_delay=0.01,
        )

    first, second = cfg(0, 0.2), cfg(1, 0.1)
    if kind == "pulse":
        reset = PulseResetCfg(pulse_cfg=first).build("reset")
    elif kind == "two_pulse":
        reset = TwoPulseResetCfg(pulse1_cfg=first, pulse2_cfg=second).build("reset")
    else:
        reset = BathResetCfg(
            cavity_tone_cfg=first, qubit_tone_cfg=second, pi2_cfg=cfg(1, 0.05)
        ).build("reset")
    readout = _readout()

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        prog.timeline_inc(10 / TPROC_MHZ, timeline=timeline or "default")
        prog.emit(reset, timeline=timeline)
        prog.emit(readout, timeline=timeline)

    shots = _shots(_program([reset, readout], body))
    # Cavity post finishes before pi/2 starts; the qubit overlaps both.
    assert [_waves(s) for s in shots] == [
        [10 + t for t in reset_waves] + [length + 10, length + 10 + READ_PRE]
    ] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [length + 10 + READ_END] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [length + 10 + READ_SHELL] * 2
    assert [
        [e.cycle - s.start_cycle for e in s.events if e.kind == "data"] for s in shots
    ] == [[length + 10 + READ_TRIGGER, length + 20 + READ_TRIGGER]] * 2


@pytest.mark.parametrize("kind", ["two_pulse", "bath"])
@pytest.mark.parametrize(
    "operation",
    ["delay", "delay_auto", "delay_by_timeline", "runtime_sync", "register", "repeat"],
)
def test_reset_rejects_reference_advancing_pulse_lowering_at_build(kind, operation):
    class AdvancingPulse(Pulse):
        def lower(
            self, prog: TimelineProgramV2, *, timeline: str | None, tag: str | None
        ) -> None:
            if operation == "delay":
                prog.delay(0.1)
            elif operation == "delay_auto":
                prog.delay_auto(0.1)
            elif operation == "delay_by_timeline":
                prog.timeline_inc(0.1, timeline=timeline)
                prog.delay_by_timeline(timeline=timeline)
            elif operation == "runtime_sync":
                prog.write_reg("duration", 10)
                prog.timeline_inc("duration", timeline=timeline)
                prog.delay_by_timeline(timeline=timeline)
            elif operation == "register":
                prog.write_reg("duration", 10)
                prog.delay_reg_auto("duration")
            elif operation == "repeat":
                prog.delay_by_timeline(timeline=timeline)
                with prog.repeat("inner", 2, timeline=timeline):
                    prog.timeline_inc(0.1, timeline=timeline)
                    prog.delay_by_timeline(timeline=timeline)
            else:
                raise ValueError(f"unknown reference operation: {operation}")

    def cfg(ch):
        return PulseCfg(
            ch=ch,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.1),
        )

    first, second = cfg(0), cfg(1)
    child = AdvancingPulse("advancing", first)
    if kind == "two_pulse":
        reset = TwoPulseResetCfg(pulse1_cfg=first, pulse2_cfg=second).build("reset")
        reset.reset_pulse1 = child
    else:
        reset = BathResetCfg(
            cavity_tone_cfg=first, qubit_tone_cfg=second, pi2_cfg=first
        ).build("reset")
        reset.res_pulse = child

    def initialize(prog):
        prog.add_reg("duration")

    with pytest.raises(
        ValueError, match="cannot advance reference; use timeline_inc instead"
    ):
        _program([reset], initialize=initialize)


class _PrepareRegisters(Macro):
    """Public QICK register declarations for existing consumers."""

    def preprocess(self, prog):
        for name in ("gain_word", "freq_word", "ro_freq_word", "wait_cycles"):
            prog.add_reg(name)

    def translate(self, prog):
        for name, value in (
            ("gain_word", 8192),
            ("freq_word", 12345678),
            ("ro_freq_word", 23456789),
            ("wait_cycles", 100),
        ):
            WriteReg(dst=name, src=value).translate(prog)


def _prepare_registers(prog):
    prog.append_macro(_PrepareRegisters())


@pytest.mark.parametrize("timeline", ["default", "future"])
@pytest.mark.parametrize("wait", [0.2, QickSweep1D("scan", 0.2, 0.4), "wait_cycles"])
def test_delay_auto_owner_rejects_every_explicit_timeline_at_entry(timeline, wait):
    auto = DelayAuto("auto", wait)

    def body(prog):
        with pytest.raises(ValueError, match="delay_auto.*timeline_inc"):
            prog.emit(auto, timeline=timeline)

    _program([auto], body, _prepare_registers)


@pytest.mark.parametrize(
    "wait", [0.2, QickSweep1D("scan", 0.2, 0.4)]
)
def test_delay_auto_owner_is_native_reference_advance_not_timeline_duration(
    wait
):
    waits = native_sweep(PROFILE, 0.2, 0.4, 3) if isinstance(wait, QickParam) else [int(PROFILE.us2cycles(wait))]
    expected = [round(generator_duration(PROFILE, 0.1) * TPROC_MHZ) + value for value in waits]
    pulse = _pulse()
    auto = DelayAuto("auto", wait)

    def initialize(prog):
        prog.declare_gen(2, nqz=1)
        prog.add_pulse(
            2, "native", style="const", freq=1000, phase=0, gain=0.5, length=0.1
        )
        if isinstance(wait, QickParam):
            prog.add_loop("scan", 3)

    def body(prog):
        prog.emit(pulse)
        prog.emit(auto)
        prog.pulse(2, "native")
        prog.emit(pulse)

    shots = _shots(_program([pulse, auto], body, initialize))
    assert [_waves(s) for s in shots] == [[0, t, t] for t in expected] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [
        t + PULSE_END for t in expected
    ] * 2


@pytest.mark.parametrize("consumer", ["native", "owner"])
def test_known_register_delay_auto_uses_static_proof_then_register_cycles(consumer):
    auto = DelayAuto("auto", "wait_cycles")
    pulse = _pulse(length=0.2)

    def body(prog):
        prog.fork("default", into="other")
        prog.timeline_inc(40 / TPROC_MHZ)
        prog.timeline_inc(60 / TPROC_MHZ, timeline="other")
        prog.pulse(0, pulse.pulse_name(prog))
        if consumer == "native":
            prog.delay_reg_auto("wait_cycles")
        else:
            prog.emit(auto)
        prog.pulse(0, pulse.pulse_name(prog), t=1 / TPROC_MHZ, timeline="default")
        prog.pulse(1, "native", timeline="other")

    def initialize(prog):
        _prepare_registers(prog)
        prog.declare_gen(1, nqz=1)
        prog.add_pulse(
            1, "native", style="const", freq=1000, phase=0, gain=0.5, length=0.1
        )

    shots = _shots(_program([auto, pulse], body, initialize))
    origin = round(generator_duration(PROFILE, 0.2) * TPROC_MHZ) + 100
    end = origin + 1 + tproc_ceil(PROFILE, generator_duration(PROFILE, 0.2))
    assert [_waves(s) for s in shots] == [[0, origin + 1, origin]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [end] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [end] * 2


@pytest.mark.parametrize("timeline", ["default", "other"])
@pytest.mark.parametrize(
    "end", [90 / TPROC_MHZ, QickSweep1D("scan", 60 / TPROC_MHZ, 90 / TPROC_MHZ)]
)
def test_runtime_guard_rejects_unknown_custom_time_producer(timeline, end):
    pulse = _pulse(length=0.2)

    def initialize(prog):
        _prepare_registers(prog)
        prog.add_loop("scan", 2)

    def body(prog):
        if timeline != "default":
            prog.fork("default", into=timeline)
        prog.timeline_inc(end, timeline=timeline)
        prog.pulse(0, pulse.pulse_name(prog))
        prog.delay_reg_auto("wait_cycles")

    with pytest.raises(ValueError, match="wait_cycles.*unknown.*producer"):
        _program([pulse], body, initialize)


def test_runtime_consumer_without_channels_advances_known_register_only():
    shots = _shots(
        _program([DelayAuto("auto", "wait_cycles")], initialize=_prepare_registers)
    )
    assert [(s.start_cycle, s.body_end_cycle, s.end_cycle) for s in shots] == [
        (0, 100, 100),
        (100, 200, 200),
    ]


@pytest.mark.parametrize("timeline", [None, "read"])
@pytest.mark.parametrize(
    "words",
    [
        {"gain_val": "gain_word"},
        {"freq_val": "freq_word"},
        {"ro_freq_val": "ro_freq_word"},
        {
            "gain_val": "gain_word",
            "freq_val": "freq_word",
            "ro_freq_val": "ro_freq_word",
        },
    ],
)
def test_runtime_words_bind_requested_registers_to_output_fields(words, timeline):
    readout = _readout(**words)

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        prog.emit(readout, timeline=timeline)

    prog = _program([readout], body, _prepare_registers)
    # These are final hardware fields, not timing analysis or declaration identity.
    expected = {
        "gain_val": ("w3", "gain_word", 8192),
        "freq_val": ("w0", "freq_word", 12345678),
        "ro_freq_val": ("w0", "ro_freq_word", 23456789),
    }
    for consumer in words:
        field, name, value = expected[consumer]
        port = (
            prog.soccfg["readouts"][0]["tproc_ctrl"]
            if consumer == "ro_freq_val"
            else prog.soccfg["gens"][1]["tproc_ch"]
        )
        address = "r" + str(prog.reg_dict[name].addr)
        assert any(
            i.get("CMD") == "REG_WR"
            and i.get("DST") == address
            and i.get("LIT") == f"#{value}"
            for i in prog.prog_list
        )
        patches = [
            index
            for index, inst in enumerate(prog.prog_list)
            if inst.get("CMD") == "REG_WR"
            and inst.get("DST") == field
            and inst.get("OP") == address
        ]
        assert patches
        for index in patches:
            output = next(
                i for i in prog.prog_list[index + 1 :] if i.get("CMD") == "WPORT_WR"
            )
            assert (output["DST"], output["SRC"]) == (str(port), "r_wave")
    assert [_waves(s) for s in _shots(prog)] == [[0, READ_PRE]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in _shots(prog)] == [READ_END] * 2


@pytest.mark.parametrize("timeline", ["default", "other"])
def test_runtime_guard_accepts_swept_end_equal_to_static_at_last_point(timeline):
    pulse = _pulse(length=0.2)

    def initialize(prog):
        _prepare_registers(prog)
        prog.add_loop("scan", 2)

    def body(prog):
        if timeline != "default":
            prog.fork("default", into=timeline)
        prog.timeline_inc(
            QickSweep1D("scan", 60 / TPROC_MHZ, round(generator_duration(PROFILE, 0.2) * TPROC_MHZ) / TPROC_MHZ), timeline=timeline
        )
        prog.pulse(0, pulse.pulse_name(prog))
        prog.delay_reg_auto("wait_cycles")
        prog.pulse(
            0, pulse.pulse_name(prog), t=1 / TPROC_MHZ, timeline=timeline, tag="after"
        )

    prog = _program([pulse], body, initialize)
    assert prog.get_time_param("after", "t") * TPROC_MHZ == pytest.approx(1)
    origin = round(generator_duration(PROFILE, 0.2) * TPROC_MHZ) + 100
    assert [_waves(s) for s in _shots(prog)] == [[0, origin + 1]] * 4


def test_swept_sync_leaves_zero_end_for_subsequent_runtime_guard():
    auto = DelayAuto("auto", "wait_cycles")

    def initialize(prog):
        _prepare_registers(prog)
        prog.add_loop("scan", 4)

    def body(prog):
        prog.timeline_inc(QickSweep1D("scan", 0.2, 0.8))
        prog.delay_by_timeline()
        prog.emit(auto)

    shots = _shots(_program([auto], body, initialize))
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [t + 100 for t in native_sweep(PROFILE, 0.2, 0.8, 4)] * 2


def test_emit_cannot_overlap_same_channel_via_another_fork():
    pulse = _pulse()

    def body(prog):
        prog.fork("default", into="other")
        prog.emit(pulse)
        prog.emit(pulse, timeline="other")

    with pytest.raises(ValueError, match="other.*channel 0.*overlap"):
        _program([pulse], body)


@pytest.mark.parametrize("timeline", ["missing", ""])
def test_emit_rejects_undeclared_timeline(timeline):
    pulse = _pulse()
    with pytest.raises(ValueError, match="unknown timeline.*fork"):
        _program([pulse], lambda p: p.emit(pulse, timeline=timeline))


def test_emitting_an_undeclared_module_is_rejected():
    with pytest.raises(ValueError, match="declared"):
        _program([], lambda prog: prog.emit(_pulse()))
