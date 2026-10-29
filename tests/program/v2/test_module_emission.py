"""Module emissions are measured at the public compiler/analysis seam."""

from collections.abc import Callable, Sequence

import pytest
from qick.asm_v2 import Macro, QickParam, QickSweep1D, WriteReg
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import Module, Pulse, PulseCfg
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, SoftDelay
from zcu_tools.program.v2.modules.readout import PulseReadout, PulseReadoutCfg
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2


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

    return Program(make_mock_soccfg(3, 1), reps=2, initial_delay=0, relax_delay=0)


def _shots(prog):
    return ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list, prog.labels, prog.timing_layout, dmem=prog.compile_datamem()
        )
    ).shots


def _waves(shot):
    return [e.cycle - shot.start_cycle for e in shot.events if e.kind == "wave"]


@pytest.mark.parametrize("timeline", [None, "drive"])
@pytest.mark.parametrize(
    "delays, events, duration",
    [
        ((0, 0), [0, 44], 88),
        ((0.01, 0.02), [4, 62], 116),
    ],
)
def test_reused_scalar_pulse_has_independent_tags_and_common_end(
    timeline, delays, events, duration
):
    pre_delay, post_delay = delays
    pulse = _pulse(pre_delay=pre_delay, post_delay=post_delay)

    def body(prog):
        if timeline is not None:
            prog.fork("default", into=timeline)
        prog.emit(pulse, timeline=timeline, tag="first")
        prog.emit(pulse, timeline=timeline, tag="second")

    prog = _program([pulse], body)
    # pre_delay ceil=5, pulse ceil(ceil(.1*384)/384*430.08)=44,
    # post_delay ceil=9: rule-a duration 58; playback pre_delay rounds to4.
    assert [_waves(s) for s in _shots(prog)] == [events] * 2
    assert [s.body_end_cycle - s.start_cycle for s in _shots(prog)] == [duration] * 2
    assert prog.get_time_param("first", "t") * 430.08 == pytest.approx(events[0])
    assert prog.get_time_param("second", "t") * 430.08 == pytest.approx(events[1])
    expected = _shots(prog)
    prog.compile()
    assert _shots(prog) == expected


@pytest.mark.parametrize("swept", [False, True])
def test_joined_emissions_preserve_parallel_origins_and_rule_a_end(swept):
    a = _pulse("a")
    b = _pulse("b", ch=1, length=QickSweep1D("scan", 0.2, 0.4) if swept else 0.2)
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
    expected = [[0, 0, 87], [0, 0, 130], [0, 0, 173]] if swept else [[0, 0, 87]]
    assert [_waves(s) for s in shots] == expected * 2
    ends = [131, 174, 217] if swept else [131]
    assert [s.body_end_cycle - s.start_cycle for s in shots] == ends * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == ends * 2


def test_repeat_emissions_keep_independent_tags_and_exact_rule_a_spacing():
    pulse = _pulse()

    def body(prog):
        prog.emit(pulse, tag="before")
        with prog.repeat("r", 2):
            prog.emit(pulse, tag="inside")
        prog.emit(pulse, tag="after")

    prog = _program([pulse], body)
    shots = _shots(prog)
    assert [_waves(s) for s in shots] == [[0, 44, 88, 132]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [176] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [176] * 2
    assert prog.get_time_param("inside", "t") * 430.08 == pytest.approx(0)


@pytest.mark.parametrize("sync", [False, True])
def test_default_emit_only_moves_end_until_sync(sync):
    pulse = _pulse()

    def body(prog):
        prog.emit(pulse)
        if sync:
            prog.sync()
        prog.pulse(0, pulse.pulse_id)

    shots = _shots(_program([pulse], body))
    assert [_waves(s) for s in shots] == [[0, 44 if sync else 0]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [44] * 2


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
def test_t1_including_readout_uses_rule_a_at_each_scan_point(
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
    # Drive rule-a=44; wait start=87, span=259 truncated to86 per step.
    # Readout config at 131+86*k, trigger at +43, playback at +22.
    # ADC ceil(.4*307.2)=123, trigger ceil(.1*430.08)=44:
    # readout rule-a ceil(44+123/307.2*430.08)=217.
    offsets = [131, 217, 303, 389] if swept else [131]
    assert [_waves(s) for s in shots] == [
        [0, offset, offset + 22] for offset in offsets
    ] * 2
    assert [
        [(e.cycle - s.start_cycle, e.value) for e in s.events if e.kind == "data"]
        for s in shots
    ] == [[(offset + 43, 1), (offset + 53, 0)] for offset in offsets] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [
        offset + 217 for offset in offsets
    ] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [
        offset + 217 for offset in offsets
    ] * 2


@pytest.mark.parametrize("delay_type", [Delay, SoftDelay])
def test_delay_emit_does_not_move_reference_and_does_not_receil(delay_type):
    delay = delay_type("wait", 0.504)
    pulse = _pulse()

    def body(prog):
        prog.emit(delay)
        prog.pulse(0, pulse.pulse_id)
        prog.sync()
        prog.pulse(0, pulse.pulse_id)

    shots = _shots(_program([delay, pulse], body))
    # ceil(.504*430.08)=217, not218.
    assert [_waves(s) for s in shots] == [[0, 217]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [217] * 2


def test_wrong_pulse_time_is_detected_by_independent_timing_expectation():
    pulse = _pulse()

    def body(prog):
        prog.emit(pulse)
        prog.pulse(0, pulse.pulse_id, t=1)

    shot = _shots(_program([pulse], body))[0]
    with pytest.raises(AssertionError):
        assert _waves(shot) == [0, 0]


@pytest.mark.parametrize(
    "kind, reset_waves, length",
    [
        ("pulse", [9], 101),
        ("two_pulse", [9, 9], 101),
        ("bath", [9, 9, 110], 138),
    ],
)
@pytest.mark.parametrize("timeline", [None, "reset"])
def test_reset_composite_keeps_internal_overlap(kind, reset_waves, length, timeline):
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
        prog.delay_timeline(10 / 430.08, timeline=timeline or "default")
        prog.emit(reset, timeline=timeline)
        prog.emit(readout, timeline=timeline)

    shots = _shots(_program([reset, readout], body))
    # ceil pre/post=9/5; .2-us pulse ->77 generator ticks ->86.24 tProc,
    # .05-us pi/2 ->20 generator ticks ->22.4 tProc: durations101 and37.
    assert [_waves(s) for s in shots] == [
        [10 + t for t in reset_waves] + [length + 10, length + 32]
    ] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [length + 227] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [length + 227] * 2
    assert [
        [e.cycle - s.start_cycle for e in s.events if e.kind == "data"] for s in shots
    ] == [[length + 53, length + 63]] * 2


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
        with pytest.raises(ValueError, match="delay_auto.*delay_timeline"):
            prog.emit(auto, timeline=timeline)

    _program([auto], body, _prepare_registers)


@pytest.mark.parametrize(
    "wait, expected", [(0.2, [129]), (QickSweep1D("scan", 0.2, 0.4), [129, 172, 215])]
)
def test_delay_auto_owner_is_native_reference_advance_not_timeline_duration(
    wait, expected
):
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
        t + 44 for t in expected
    ] * 2


@pytest.mark.parametrize("consumer", ["native", "owner"])
def test_known_register_delay_auto_uses_static_proof_then_register_cycles(consumer):
    auto = DelayAuto("auto", "wait_cycles")
    pulse = _pulse(length=0.2)

    def body(prog):
        prog.fork("default", into="other")
        prog.delay_timeline(40 / 430.08)
        prog.delay_timeline(60 / 430.08, timeline="other")
        prog.pulse(0, pulse.pulse_id)
        if consumer == "native":
            prog.delay_reg_auto("wait_cycles")
        else:
            prog.emit(auto)
        prog.pulse(0, pulse.pulse_id, t=1 / 430.08, timeline="default")
        prog.pulse(1, "native", timeline="other")

    def initialize(prog):
        _prepare_registers(prog)
        prog.declare_gen(1, nqz=1)
        prog.add_pulse(
            1, "native", style="const", freq=1000, phase=0, gain=0.5, length=0.1
        )

    shots = _shots(_program([auto, pulse], body, initialize))
    # Native .2 rounds to77 generator ticks ->86.24 tProc -> static86,
    # proving ends40 and60 <=86. Known runtime register adds100.
    assert [_waves(s) for s in shots] == [[0, 187, 186]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [274] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [274] * 2


@pytest.mark.parametrize("timeline", ["default", "other"])
@pytest.mark.parametrize(
    "end", [90 / 430.08, QickSweep1D("scan", 60 / 430.08, 90 / 430.08)]
)
def test_runtime_guard_rejects_end_exceeding_static_at_any_point(timeline, end):
    pulse = _pulse(length=0.2)

    def initialize(prog):
        _prepare_registers(prog)
        prog.add_loop("scan", 2)

    def body(prog):
        if timeline != "default":
            prog.fork("default", into=timeline)
        prog.delay_timeline(end, timeline=timeline)
        prog.pulse(0, pulse.pulse_id)
        prog.delay_reg_auto("wait_cycles")

    with pytest.raises(ValueError, match=timeline + ".*sync"):
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
    assert [_waves(s) for s in _shots(prog)] == [[0, 22]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in _shots(prog)] == [217] * 2


@pytest.mark.parametrize("timeline", ["default", "other"])
def test_runtime_guard_accepts_swept_end_equal_to_static_at_last_point(timeline):
    pulse = _pulse(length=0.2)

    def initialize(prog):
        _prepare_registers(prog)
        prog.add_loop("scan", 2)

    def body(prog):
        if timeline != "default":
            prog.fork("default", into=timeline)
        prog.delay_timeline(
            QickSweep1D("scan", 60 / 430.08, 86 / 430.08), timeline=timeline
        )
        prog.pulse(0, pulse.pulse_id)
        prog.delay_reg_auto("wait_cycles")
        prog.pulse(0, pulse.pulse_id, t=1 / 430.08, timeline=timeline, tag="after")

    prog = _program([pulse], body, initialize)
    assert prog.get_time_param("after", "t") * 430.08 == pytest.approx(1)
    assert [_waves(s) for s in _shots(prog)] == [[0, 187]] * 4


def test_swept_sync_leaves_zero_end_for_subsequent_runtime_guard():
    auto = DelayAuto("auto", "wait_cycles")

    def initialize(prog):
        _prepare_registers(prog)
        prog.add_loop("scan", 4)

    def body(prog):
        prog.delay_timeline(QickSweep1D("scan", 0.2, 0.8))
        prog.sync()
        prog.emit(auto)

    shots = _shots(_program([auto], body, initialize))
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [187, 273, 359, 445] * 2


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
