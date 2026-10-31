"""Native QICK operations retain reference/channel selection around B1 ends."""

import pytest
from qick.asm_v2 import QickSweep1D
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from .timing_support import shots as _shots


def _program(body, initialize=None):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            for ch in (0, 1):
                self.declare_gen(ch, nqz=1)
                self.add_pulse(
                    ch,
                    f"p{ch}",
                    style="const",
                    freq=1000,
                    phase=0,
                    gain=0.5,
                    length=0.1,
                )
            self.add_pulse(
                0, "pulse", style="const", freq=1000, phase=0, gain=0.5, length=0.1
            )
            self.declare_readout(0, length=0.4)
            self.add_readoutconfig(0, "ro", freq=100)
            if initialize is not None:
                initialize(self)

        def _body(self, cfg):
            body(self)

    return Program(make_mock_soccfg(2, 1), reps=1, initial_delay=0, relax_delay=0)


def _shot(prog):
    return ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, prog.timing_layout)
    ).shots[0]


@pytest.mark.parametrize(
    "gens,ros,advance", [(True, False, 43), (False, True, 172), (True, True, 172)]
)
def test_native_auto_keeps_channel_selection_and_decrements_all_ends(
    gens, ros, advance
):
    def body(prog):
        prog.fork("default", into="other")
        prog.delay_timeline(200 / 430.08)
        prog.delay_timeline(220 / 430.08, timeline="other")
        prog.pulse(0, "pulse")
        prog.trigger([0])
        prog.delay_auto(gens=gens, ros=ros)
        prog.pulse(1, "p1", tag="native")
        prog.pulse(0, "pulse", timeline="default", tag="default")
        prog.pulse(1, "p1", timeline="other", tag="other")

    prog = _program(body)
    assert prog.get_time_param("native", "t") == 0
    assert prog.get_time_param("default", "t") * 430.08 == pytest.approx(200 - advance)
    assert prog.get_time_param("other", "t") * 430.08 == pytest.approx(220 - advance)
    assert [e.cycle for e in _shot(prog).events if e.kind == "wave"] == [
        0,
        advance,
        200,
        220,
    ]


@pytest.mark.parametrize("length", [None, 0.8])
@pytest.mark.parametrize("offset", [0, 5])
@pytest.mark.parametrize("timeline", ["default", "read"])
def test_timeline_config_advances_to_its_instant_not_adc_window_or_length(
    length, offset, timeline
):
    def initialize(prog):
        kwargs = {} if length is None else {"length": length}
        prog.add_readoutconfig(0, "config", freq=100, **kwargs)

    def body(prog):
        if timeline != "default":
            prog.fork("default", into=timeline)
        prog.trigger([0], t=1)
        prog.delay_timeline(10 / 430.08, timeline=timeline)
        prog.send_readoutconfig(0, "config", t=offset / 430.08, timeline=timeline)
        prog.pulse(0, "pulse", timeline=timeline)

    shot = _shot(_program(body, initialize))
    # Instant config: E10+t0 ->10; E10+t5 ->15. The .1-us waveform
    # then adds ceil(38/384*430.08)=43. Other ADC timestamps affect only shell.
    assert [e.cycle for e in shot.events if e.kind == "wave"] == [10 + offset] * 2
    assert shot.body_end_cycle == 53 + offset
    assert [(e.cycle, e.value) for e in shot.events if e.kind == "data"] == [
        (430, 1),
        (440, 0),
    ]
    assert shot.end_cycle == 602


@pytest.mark.parametrize("delay, expected", [(20, 43), (60, 60)])
def test_timeline_pulse_auto_chooses_later_channel_or_end(delay, expected):
    def body(prog):
        prog.pulse(0, "pulse")
        prog.delay_timeline(delay / 430.08)
        prog.pulse(0, "pulse", t="auto", timeline="default")

    shot = _shot(_program(body))
    assert [e.cycle for e in shot.events if e.kind == "wave"] == [0, expected]


def test_timeline_trigger_advances_through_adc_window_and_retains_native_wait():
    def body(prog):
        prog.delay_timeline(10 / 430.08)
        prog.trigger([0], timeline="default")
        prog.wait_auto(tag="wait")
        prog.pulse(0, "pulse", timeline="default")

    prog = _program(body)
    # Native ADC window is123/307.2 us =172.2 tProc cycles.
    # Timeline closure ceilings 10+172.2 to183; native wait rounds to182.
    assert prog.get_time_param("wait", "t") * 430.08 == pytest.approx(182)
    shot = _shot(prog)
    assert [(e.cycle, e.value) for e in shot.events if e.kind == "data"] == [
        (10, 1),
        (20, 0),
    ]
    assert [e.cycle for e in shot.events if e.kind == "wave"] == [183]
    assert shot.body_end_cycle == 226


def test_fractional_event_start_and_trigger_width_ceil_once():
    def body(prog):
        prog.send_readoutconfig(
            0, "ro", t=0.2 / 430.08, timeline="default", tag="config"
        )
        prog.trigger(
            [0], t=0.2 / 430.08, width=0.2 / 430.08, timeline="default", tag="edge"
        )
        prog.sync()
        prog.pulse(0, "pulse", timeline="default", tag="after")

    prog = _program(body)
    assert prog.get_time_param("config", "t") * 430.08 == pytest.approx(1)
    assert prog.get_time_param("edge", "t") * 430.08 == pytest.approx(2)
    assert prog.get_time_param("edge", "width") * 430.08 == pytest.approx(1)
    assert [e.cycle for e in _shot(prog).events if e.kind == "wave"] == [1, 175]


@pytest.mark.parametrize("operation", ["pulse", "config", "trigger", "delay"])
@pytest.mark.parametrize("ticks, rounded", [(5e-11, 1), (1.00000000005, 2), (43, 43)])
def test_scalar_ceil_preserves_positive_tails_and_exact_grid(operation, ticks, rounded):
    def body(prog):
        value = ticks / 430.08
        if operation == "pulse":
            prog.pulse(0, "pulse", t=value, timeline="default", tag="event")
        elif operation == "config":
            prog.send_readoutconfig(0, "ro", t=value, timeline="default", tag="event")
        elif operation == "trigger":
            prog.trigger([0], t=value, width=value, timeline="default", tag="event")
        else:
            prog.delay_timeline(value)
            prog.delay_timeline(value, tag="event")
        prog.sync()
        prog.pulse(1, "p1", timeline="default", tag="after")

    prog = _program(body)
    start = 2 * rounded if operation == "delay" else rounded
    assert prog.get_time_param("event", "t") * 430.08 == pytest.approx(start)
    if operation == "trigger":
        assert prog.get_time_param("event", "width") * 430.08 == pytest.approx(rounded)
    # The mock waveform lasts 42.56 ticks; its ADC window lasts 172.2 ticks.
    end = start + {"pulse": 43, "config": 0, "trigger": 173, "delay": 0}[operation]
    assert [e.cycle for e in _shot(prog).events if e.kind == "wave"][-1] == end
    assert prog.get_time_param("after", "t") == 0


def test_author_start_and_trigger_width_sweeps_keep_native_affine_values():
    def body(prog):
        value = QickSweep1D("scan", 0.2 / 430.08, 3.8 / 430.08)
        prog.trigger([0], t=value, width=value, timeline="default", tag="edge")
        prog.sync()
        prog.pulse(0, "pulse", timeline="default", tag="after")

    prog = _program(body, lambda p: p.add_loop("scan", 4))
    assert prog.get_time_param("edge", "t", as_array=True) * 430.08 == pytest.approx(
        [0, 1, 2, 3]
    )
    assert prog.get_time_param(
        "edge", "width", as_array=True
    ) * 430.08 == pytest.approx([0, 1, 2, 3])
    assert [
        [e.cycle - s.start_cycle for e in s.events if e.kind == "wave"]
        for s in _shots(prog)
    ] == [[173], [174], [175], [176]]
    assert prog.compile_datamem() is None


def test_timeline_trigger_rejects_implicit_output_time_at_entry():
    def body(prog):
        with pytest.raises(ValueError, match="t=None"):
            prog.trigger([0], t=None, timeline="default")

    _program(body)


def test_resync_compiles_and_statically_decrements_every_timeline():
    def body(prog):
        prog.fork("default", into="other")
        prog.delay_timeline(40 / 430.08)
        prog.delay_timeline(70 / 430.08, timeline="other")
        prog.resync(10 / 430.08)
        prog.pulse(0, "pulse", timeline="default", tag="default")
        prog.pulse(1, "p1", timeline="other", tag="other")

    prog = _program(body)
    assert prog.get_time_param("default", "t") * 430.08 == pytest.approx(30)
    assert prog.get_time_param("other", "t") * 430.08 == pytest.approx(60)
    with pytest.raises(ValueError, match="s11.*s_usr_time"):
        _shot(prog)


@pytest.mark.parametrize(
    "start,stop,expected",
    [(12, 14, [12, 14]), (6, 8, [10, 10]), (10, 14, [10, 14]), (6, 10, [10, 10])],
)
def test_swept_floor_proves_sign_including_zero(start, stop, expected):
    def initialize(prog):
        prog.add_loop("scan", 2)

    def body(prog):
        prog.delay_timeline(QickSweep1D("scan", start / 430.08, stop / 430.08))
        prog.delay(10 / 430.08)
        prog.pulse(0, "pulse", timeline="default")

    prog = _program(body, initialize)
    shots = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, prog.timing_layout)
    ).shots
    assert [
        [e.cycle - s.start_cycle for e in s.events if e.kind == "wave"] for s in shots
    ] == [[x] for x in expected]


def test_crossing_swept_floor_preserves_each_point():
    def initialize(prog):
        prog.add_loop("scan", 2)

    def body(prog):
        prog.fork("default", into="drive")
        prog.delay_timeline(
            QickSweep1D("scan", 9 / 430.08, 11 / 430.08), timeline="drive"
        )
        prog.delay(10 / 430.08)

    result = _shots(_program(body, initialize))
    assert [s.body_end_cycle - s.start_cycle for s in result] == [10, 11]


def test_crossing_swept_auto_maximum_places_at_each_points_later_end():
    def initialize(prog):
        prog.add_loop("scan", 2)

    def body(prog):
        prog.fork("default", into="drive")
        prog.pulse(0, "pulse", t=QickSweep1D("scan", 0, 100 / 430.08))
        prog.delay_timeline(QickSweep1D("scan", 100 / 430.08, 0), timeline="drive")
        prog.pulse(0, "pulse", t="auto", timeline="drive")

    result = _shots(_program(body, initialize))
    assert [
        [
            event.cycle - shot.start_cycle
            for event in shot.events
            if event.kind == "wave"
        ]
        for shot in result
    ] == [[0, 100], [100, 143]]
    assert [shot.body_end_cycle - shot.start_cycle for shot in result] == [143, 186]
    assert [shot.end_cycle - shot.start_cycle for shot in result] == [143, 186]


def test_crossing_body_maximum_closes_at_each_points_latest_end():
    def initialize(prog):
        prog.add_loop("scan", 2)

    def body(prog):
        prog.fork("default", into="other")
        prog.delay_timeline(QickSweep1D("scan", 10 / 430.08, 30 / 430.08))
        prog.delay_timeline(
            QickSweep1D("scan", 30 / 430.08, 10 / 430.08), timeline="other"
        )

    result = _shots(_program(body, initialize))
    assert [s.body_end_cycle - s.start_cycle for s in result] == [30, 30]
