"""Public emission contracts and migration parity; retire parity pins at legacy removal."""

from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import pytest
from qick.asm_v2 import AveragerProgramV2, Macro, QickSweep1D, WriteReg
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
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


def _pulse(name="drive", *, ch=0, length=0.1, **kwargs):
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


def _legacy(modules, *, initialize=None, **kwargs):
    class Program(ModularProgramV2):
        def _initialize(self, cfg):
            if initialize is not None:
                initialize(self)
            super()._initialize(cfg)

    return Program(
        make_mock_soccfg(3, 1),
        modules,
        reps=2,
        initial_delay=0,
        relax_delay=0,
        **kwargs,
    )


def _assert_equal(actual: AveragerProgramV2, expected: AveragerProgramV2):
    assert actual.prog_list == expected.prog_list
    assert actual.labels == expected.labels
    for memory in ("pmem", "wmem", "dmem"):
        np.testing.assert_array_equal(actual.binprog[memory], expected.binprog[memory])
    assert actual.asm() == expected.asm()


def test_two_consecutive_scalar_pulses_are_bitwise_equal():
    _assert_equal(
        _program([_pulse("a"), _pulse("b")]), _legacy([_pulse("a"), _pulse("b")])
    )


@pytest.mark.parametrize("delay_type", [Delay, SoftDelay])
@pytest.mark.parametrize("swept", [False, True])
def test_scalar_and_swept_delays_are_bitwise_equal(delay_type, swept):
    delay = QickSweep1D("wait", 0.2, 0.8) if swept else 0.2

    def modules():
        return [_pulse(), delay_type("wait", delay), _pulse("end", ch=1)]

    def initialize(prog):
        if swept:
            prog.add_loop("wait", 4)

    _assert_equal(
        _program(modules(), initialize=initialize),
        _legacy(modules(), sweep=[("wait", 4)] if swept else None),
    )


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


def test_scalar_t1_with_readout_is_bitwise_equal():
    def modules():
        return [_pulse(), Delay("wait", 0.2), _readout()]

    _assert_equal(_program(modules()), _legacy(modules()))


@pytest.mark.parametrize("kind", ["pulse", "two_pulse", "bath"])
def test_each_reset_is_bitwise_equal(kind):
    def pulse_cfg(ch, length):
        return PulseCfg(
            ch=ch,
            nqz=1,
            freq=1000,
            gain=0.5,
            waveform=ConstWaveformCfg(length=length),
            pre_delay=0.02,
            post_delay=0.01,
        )

    def modules():
        first = pulse_cfg(0, 0.2)
        second = pulse_cfg(1, 0.1)
        if kind == "pulse":
            cfg = PulseResetCfg(pulse_cfg=first)
        elif kind == "two_pulse":
            cfg = TwoPulseResetCfg(pulse1_cfg=first, pulse2_cfg=second)
        else:
            cfg = BathResetCfg(
                cavity_tone_cfg=first, qubit_tone_cfg=second, pi2_cfg=pulse_cfg(1, 0.05)
            )
        return [cfg.build("reset"), _readout()]

    _assert_equal(_program(modules()), _legacy(modules()))


class _PrepareRegisters(Macro):
    """Prepare consumer words with public QICK APIs, not out-of-scope producers."""

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
def test_readout_runtime_word_consumers_are_bitwise_equal(words, timeline):
    readout = _readout(**words)

    def body(prog):
        prog.emit(readout, timeline=timeline)
        if timeline is not None:
            prog.sync(timeline=timeline)

    _assert_equal(
        _program([readout], body, _prepare_registers),
        _legacy([_readout(**words)], initialize=_prepare_registers),
    )


@pytest.mark.parametrize("wait", [0.2, "wait_cycles"])
def test_delay_auto_scalar_and_register_consumers_are_bitwise_equal(wait):
    def modules():
        return [_pulse(), DelayAuto("auto", t=wait), _pulse("end", ch=1)]

    _assert_equal(
        _program(modules(), initialize=_prepare_registers),
        _legacy(modules(), initialize=_prepare_registers),
    )


def _absolute_playback_ticks(prog):
    reference = 0
    times = []
    for inst in prog.prog_list:
        if inst.get("CMD") == "TIME" and inst.get("C_OP") == "inc_ref":
            reference += int(inst["LIT"][1:])
        elif inst["CMD"] == "WPORT_WR":
            times.append(reference + int(inst["TIME"][1:]))
    return times


def test_repeated_emissions_have_independent_time_tags_and_rebase_named_cursor():
    drive = _pulse(pre_delay=0.01, post_delay=0.02)
    default_drive = _pulse("default", ch=1, pre_delay=0.01, post_delay=0.02)

    def initialize(prog):
        prog.declare_gen(2, nqz=1)
        prog.add_pulse(
            2, "native", style="const", freq=1000, phase=0, gain=0.5, length=0.05
        )

    def body(prog):
        prog.delay(0.2, timeline="drive")
        prog.emit(drive, timeline="drive", tag="first")
        prog.delay(0.1)
        prog.emit(default_drive, tag="default")
        prog.pulse(2, "native", t=0, tag="after")
        prog.emit(drive, timeline="drive", tag="second")

    prog = _program([drive, default_drive], body, initialize)
    clock = 430.08
    # Named start 87+round(.01*f)=91; owner duration 5+44+9=58 ticks.
    # Native delay moves ref to43; default owner moves it to101.
    # Named cursor rebases from145 to44, so its next pulse is at101+44+4=149.
    assert _absolute_playback_ticks(prog) == [91, 47, 101, 149]
    assert prog.get_time_param("first", "t") * clock == pytest.approx(91)
    assert prog.get_time_param("default", "t") * clock == pytest.approx(4)
    assert prog.get_time_param("after", "t") == 0
    assert prog.get_time_param("second", "t") * clock == pytest.approx(48)
    before = (prog.prog_list.copy(), prog.asm())
    prog.compile()
    assert (prog.prog_list, prog.asm()) == before
    assert _absolute_playback_ticks(prog) == [91, 47, 101, 149]


def test_emitting_an_undeclared_module_is_rejected():
    with pytest.raises(ValueError, match="declared"):
        _program([], lambda prog: prog.emit(_pulse()))


@pytest.mark.parametrize("native", [False, True])
def test_runtime_register_on_named_timeline_is_rejected_at_entry(native):
    wait = DelayAuto("wait", "wait_cycles")

    def body(prog):
        action = (
            partial(prog.delay_auto, "wait_cycles", timeline="future")
            if native
            else partial(prog.emit, wait, timeline="future")
        )
        with pytest.raises(ValueError, match="runtime.*default timeline"):
            action()

    _program([wait], body, _prepare_registers)


@pytest.mark.parametrize(
    "offset",
    [
        0.3,
        QickSweep1D("scan", 0.1, 0.4),
    ],
)
def test_runtime_reference_advance_rejects_unproven_named_offset(offset):
    wait = DelayAuto("wait", "wait_cycles")

    def initialize(prog):
        _prepare_registers(prog)
        prog.add_loop("scan", 3)
        prog.declare_gen(0, nqz=1)
        prog.add_pulse(
            0, "native", style="const", freq=1000, phase=0, gain=0.5, length=0.2
        )

    def body(prog):
        prog.delay(offset, timeline="future")
        prog.pulse(0, "native")
        prog.emit(wait)

    with pytest.raises(ValueError, match="future.*sync.*delay_auto"):
        _program([wait], body, initialize)


@pytest.mark.parametrize("native", [False, True])
def test_runtime_advance_zeros_proven_cursor_but_only_statically_rebases_ends(native):
    wait = DelayAuto("wait", "wait_cycles", gens=False, ros=True)

    def initialize(prog):
        _prepare_registers(prog)
        prog.declare_gen(0, nqz=1)
        prog.add_pulse(
            0, "long", style="const", freq=1000, phase=0, gain=0.5, length=0.6
        )
        prog.declare_readout(0, length=0.2)

    def body(prog):
        prog.pulse(0, "long", timeline="future")
        prog.delay(0.1, timeline="future")
        prog.trigger([0])
        if native:
            prog.delay_auto("wait_cycles", gens=False, ros=True)
        else:
            prog.emit(wait)
        prog.delay(0, timeline="future", tag="cursor")
        prog.delay_auto(0, gens=True, ros=False, timeline="future", tag="end")

    prog = _program([wait], body, initialize)
    # ADC window rounds to61 ADC cycles, then nearest85 tProc ticks.
    # offset44 <= S85, so cursor is zero for every nonnegative register word.
    # Generator end230 gen cycles =257.6 tProc ticks; only S85 is deducted.
    assert prog.get_time_param("cursor", "t") == 0
    assert prog.get_time_param("end", "t") * 430.08 == pytest.approx(173)
    assert any(i.get("C_OP") == "inc_ref" and "R1" in i for i in prog.prog_list)


@pytest.mark.parametrize("delay_type", [Delay, SoftDelay])
def test_named_delay_module_does_not_round_aligned_duration_twice(delay_type):
    wait = delay_type("wait", 0.504)
    prog = _program([wait], lambda prog: prog.emit(wait, timeline="future", tag="wait"))
    assert prog.get_time_param("wait", "t") * 430.08 == pytest.approx(217)


def test_runtime_register_consumer_without_channels_is_bitwise_equal():
    _assert_equal(
        _program([DelayAuto("wait", "wait_cycles")], initialize=_prepare_registers),
        _legacy([DelayAuto("wait", "wait_cycles")], initialize=_prepare_registers),
    )


def test_module_comparison_detects_a_changed_pulse():
    with pytest.raises(AssertionError):
        _assert_equal(_program([_pulse(length=0.2)]), _legacy([_pulse()]))
