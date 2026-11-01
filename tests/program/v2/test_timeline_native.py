"""Native QICK operations retain reference/channel selection around B1 ends."""

import pytest
from qick.asm_v2 import QickSweep1D, Trigger
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import RO_MHZ, TPROC_MHZ, load_soccfg

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

    return Program(load_soccfg(2, 1), reps=1, initial_delay=0, relax_delay=0)


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
        prog.timeline_inc(200 / TPROC_MHZ)
        prog.timeline_inc(220 / TPROC_MHZ, timeline="other")
        prog.pulse(0, "pulse")
        prog.trigger([0])
        prog.delay_auto(gens=gens, ros=ros)
        prog.pulse(1, "p1", tag="native")
        prog.pulse(0, "pulse", timeline="default", tag="default")
        prog.pulse(1, "p1", timeline="other", tag="other")

    prog = _program(body)
    assert prog.get_time_param("native", "t") == 0
    assert prog.get_time_param("default", "t") * TPROC_MHZ == pytest.approx(
        200 - advance
    )
    assert prog.get_time_param("other", "t") * TPROC_MHZ == pytest.approx(220 - advance)
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
        prog.timeline_inc(10 / TPROC_MHZ, timeline=timeline)
        prog.send_readoutconfig(0, "config", t=offset / TPROC_MHZ, timeline=timeline)
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
        prog.timeline_inc(delay / TPROC_MHZ)
        prog.pulse(0, "pulse", t="auto", timeline="default")

    shot = _shot(_program(body))
    assert [e.cycle for e in shot.events if e.kind == "wave"] == [0, expected]


def test_timeline_trigger_advances_only_width_and_retains_adc_wait():
    def body(prog):
        prog.timeline_inc(10 / TPROC_MHZ)
        prog.trigger([0], timeline="default")
        prog.wait_auto(tag="wait")
        prog.pulse(0, "pulse", timeline="default")

    prog = _program(body)
    # Native ADC window is123/307.2 us =172.2 tProc cycles.
    # The named end is20 (trigger width only); native ADC wait still rounds to182.
    assert prog.get_time_param("wait", "t") * TPROC_MHZ == pytest.approx(182)
    shot = _shot(prog)
    assert [(e.cycle, e.value) for e in shot.events if e.kind == "data"] == [
        (10, 1),
        (20, 0),
    ]
    assert [e.cycle for e in shot.events if e.kind == "wave"] == [20]
    assert shot.body_end_cycle == 63
    assert shot.end_cycle == 182


def test_fractional_event_start_and_trigger_width_ceil_once():
    def body(prog):
        prog.send_readoutconfig(
            0, "ro", t=0.2 / TPROC_MHZ, timeline="default", tag="config"
        )
        prog.trigger(
            [0],
            t=0.2 / TPROC_MHZ,
            width=0.2 / TPROC_MHZ,
            timeline="default",
            tag="edge",
        )
        prog.delay_by_timeline()
        prog.pulse(0, "pulse", timeline="default", tag="after")

    prog = _program(body)
    assert prog.get_time_param("config", "t") * TPROC_MHZ == pytest.approx(1)
    assert prog.get_time_param("edge", "t") * TPROC_MHZ == pytest.approx(2)
    assert prog.get_time_param("edge", "width") * TPROC_MHZ == pytest.approx(1)
    assert [e.cycle for e in _shot(prog).events if e.kind == "wave"] == [1, 3]


@pytest.mark.parametrize("operation", ["pulse", "config", "trigger", "delay"])
@pytest.mark.parametrize(
    "ticks, rounded",
    [
        (119, 119),
        (0.2, 1),
        (1.0000000002, 2),
        # Integer cycles round-trip through us with >1e-10-cycle float noise.
        (1000006, 1000006),
        (1000010, 1000010),
        # This real fractional tail exceeds the relative tolerance.
        (1000000.00001, 1000001),
    ],
)
def test_scalar_ceil_preserves_grid_roundtrip_and_tails_outside_noise(
    operation, ticks, rounded
):
    def body(prog):
        value = ticks / TPROC_MHZ
        if operation == "pulse":
            prog.pulse(0, "pulse", t=value, timeline="default", tag="event")
        elif operation == "config":
            prog.send_readoutconfig(0, "ro", t=value, timeline="default", tag="event")
        elif operation == "trigger":
            prog.trigger([0], t=value, width=value, timeline="default", tag="event")
        else:
            prog.timeline_inc(value)
            prog.timeline_inc(value, tag="event")
        prog.delay_by_timeline()
        prog.pulse(1, "p1", timeline="default", tag="after")

    prog = _program(body)
    start = 2 * rounded if operation == "delay" else rounded
    assert prog.get_time_param("event", "t") * TPROC_MHZ == pytest.approx(
        start, rel=0, abs=1e-8
    )
    if operation == "trigger":
        assert prog.get_time_param("event", "width") * TPROC_MHZ == pytest.approx(
            rounded, rel=0, abs=1e-8
        )
    # The mock waveform lasts 42.56 ticks; its ADC window lasts 172.2 ticks.
    end = (
        start
        + {
            "pulse": 43,
            "config": 0,
            "trigger": rounded,
            "delay": 0,
        }[operation]
    )
    assert [e.cycle for e in _shot(prog).events if e.kind == "wave"][-1] == end
    assert prog.get_time_param("after", "t") == 0


def test_author_start_and_trigger_width_sweeps_keep_native_affine_values():
    def body(prog):
        value = QickSweep1D("scan", 0.2 / TPROC_MHZ, 3.8 / TPROC_MHZ)
        prog.trigger([0], t=value, width=value, timeline="default", tag="edge")
        prog.delay_by_timeline()
        prog.pulse(0, "pulse", timeline="default", tag="after")

    # Width sweep leaves0..3 cycles of ADC slack; this synthetic fixture allows it.
    def initialize(prog):
        prog.add_loop("scan", 4)
        prog.relax_delay = 3 / TPROC_MHZ

    prog = _program(body, initialize)
    assert prog.get_time_param("edge", "t", as_array=True) * TPROC_MHZ == pytest.approx(
        [0, 1, 2, 3]
    )
    assert prog.get_time_param(
        "edge", "width", as_array=True
    ) * TPROC_MHZ == pytest.approx([0, 1, 2, 3])
    assert [
        [e.cycle - s.start_cycle for e in s.events if e.kind == "wave"]
        for s in _shots(prog)
    ] == [[0], [2], [4], [6]]
    assert prog.compile_datamem() is None


@pytest.mark.parametrize("placement", ["static", "sweep", "runtime"])
def test_same_adc_overlap_fails_at_build_with_channel_and_cycles(placement):
    def initialize(prog):
        prog.declare_readout(0, length=100 / RO_MHZ)
        prog.add_loop("scan", 2)

    def body(prog):
        prog.fork("default", into="next_read")
        prog.trigger([0], timeline="default")
        if placement == "sweep":
            prog.timeline_inc(
                QickSweep1D("scan", 100 / TPROC_MHZ, 180 / TPROC_MHZ),
                timeline="next_read",
            )
        else:
            prog.timeline_inc(100 / TPROC_MHZ, timeline="next_read")
            if placement == "runtime":
                prog.timeline_inc("scan", timeline="next_read")
        prog.trigger([0], timeline="next_read")

    with pytest.raises(
        ValueError, match=r"next_read.*channel 0.*ADC overlap.*40 cycles"
    ):
        _program(body, initialize)


@pytest.mark.parametrize(
    "control",
    [
        "repeat",
        "repeat_sync",
        "nested_repeat",
        "branch_then_trigger",
        "trigger_then_branch",
    ],
)
def test_adc_overlap_is_rejected_across_control_flow(control):
    def initialize(prog):
        prog.declare_readout(0, length=100 / RO_MHZ)
        prog.add_loop("scan", 2)

    def trigger(prog, timeline: str | None = "default"):
        prog.trigger([0], timeline=timeline)
        prog.timeline_inc(90 / TPROC_MHZ, timeline=timeline)

    def body(prog):
        if control in {"repeat", "repeat_sync"}:
            with prog.repeat("reads", 2):
                trigger(prog)
                if control == "repeat_sync":
                    prog.delay_by_timeline()
        elif control == "nested_repeat":
            with prog.repeat("outer", 2), prog.repeat("inner", 1):
                trigger(prog)
        elif control == "branch_then_trigger":
            with prog.branch(by="scan", align_timeline=False) as cases, cases.case(0):
                trigger(prog, None)
            prog.trigger([0], timeline="default")
        else:
            trigger(prog)
            with prog.branch(by="scan", align_timeline=False) as cases, cases.case(0):
                prog.trigger([0])

    with pytest.raises(ValueError, match=r"channel 0.*ADC overlap.*40 cycles"):
        _program(body, initialize)


@pytest.mark.parametrize("previous", [None, "default"])
def test_native_trigger_keeps_native_time_and_overlap_behavior(previous):
    def body(prog):
        prog.trigger([0], timeline=previous)
        prog.trigger([0], t=0, tag="raw")
        prog.write_reg("s_out_time", 20)
        prog.trigger([0], t=None)

    prog = _program(body)
    assert prog.get_time_param("raw", "t") == 0
    assert [(e.cycle, e.value) for e in _shot(prog).events if e.kind == "data"][:4] == [
        (0, 1),
        (10, 0),
        (0, 1),
        (10, 0),
    ]


@pytest.mark.parametrize("swept", [False, True])
def test_quantized_native_adc_tail_allows_touching_named_trigger(swept):
    def initialize(prog):
        prog.declare_readout(0, length=100 / RO_MHZ)
        prog.add_loop("scan", 2)

    def body(prog):
        start = (
            QickSweep1D("scan", 0.2 / TPROC_MHZ, 1.2 / TPROC_MHZ)
            if swept
            else 0.2 / TPROC_MHZ
        )
        prog.trigger([0], t=start, width=10 / TPROC_MHZ)
        next_start = (
            QickSweep1D("scan", 140 / TPROC_MHZ, 141 / TPROC_MHZ)
            if swept
            else 140 / TPROC_MHZ
        )
        prog.timeline_inc(next_start)
        prog.trigger([0], timeline="default")

    prog = _program(body, initialize)
    assert [
        [
            e.cycle - shot.start_cycle
            for e in shot.events
            if e.kind == "data" and e.value
        ]
        for shot in _shots(prog)
    ] == ([[0, 140], [1, 141]] if swept else [[0, 140]] * 2)


def test_native_fractional_trigger_retains_qick_outputs_and_waits_actual_adc():
    def initialize(prog):
        prog.declare_readout(0, length=100 / RO_MHZ)

    def native(prog):
        prog.trigger([0], t=0.2 / TPROC_MHZ, width=10 / TPROC_MHZ)

    def qick(prog):
        prog.append_macro(
            Trigger(
                ros=[0],
                tts=None,
                pins=None,
                t=0.2 / TPROC_MHZ,
                width=10 / TPROC_MHZ,
                ddr4=False,
                mr=False,
                tag=None,
            )
        )

    prog = _program(native, initialize)
    reference = _program(qick, initialize)
    assert [i for i in prog.prog_list if i["CMD"] == "DPORT_WR"] == [
        i for i in reference.prog_list if i["CMD"] == "DPORT_WR"
    ]
    (wait,) = [i for i in prog.prog_list if i["CMD"] == "WAIT"]
    assert wait["TIME"] == "@140"


@pytest.mark.parametrize("control", ["flat", "branch", "repeat"])
def test_unlocatable_native_adc_window_rejects_later_named_trigger(control):
    def initialize(prog):
        prog.add_loop("scan", 2)
        prog.relax_delay = 200 / TPROC_MHZ

    def body(prog):
        prog.write_reg("s_out_time", 0)
        prog.trigger([0], t=None)
        if control == "branch":
            with prog.branch(by="scan", align_timeline=False) as cases, cases.case(0):
                prog.timeline_inc(200 / TPROC_MHZ)
                prog.trigger([0])
        elif control == "repeat":
            with prog.repeat("reads", 2):
                prog.timeline_inc(200 / TPROC_MHZ)
                prog.trigger([0])
        else:
            prog.timeline_inc(200 / TPROC_MHZ)
            prog.trigger([0], timeline="default")

    with pytest.raises(
        ValueError,
        match=r"channel 0.*cannot locate previous native trigger.*cycles unknown",
    ):
        _program(body, initialize)


def test_known_native_adc_window_replaces_unlocatable_predecessor():
    def initialize(prog):
        prog.declare_readout(0, length=100 / RO_MHZ)

    def body(prog):
        prog.write_reg("s_out_time", 0)
        prog.trigger([0], t=None)
        prog.trigger([0], t=0)
        prog.timeline_inc(140 / TPROC_MHZ)
        prog.trigger([0], timeline="default")

    prog = _program(body, initialize)
    assert [e.cycle for e in _shot(prog).events if e.kind == "data" and e.value] == [
        0,
        0,
        140,
    ]


def test_native_adc_tail_is_checked_before_a_named_trigger():
    def body(prog):
        prog.trigger([0], t=0)
        prog.timeline_inc(100 / TPROC_MHZ)
        prog.trigger([0], timeline="default")

    with pytest.raises(ValueError, match=r"default.*channel 0.*ADC overlap.*73 cycles"):
        _program(body)


@pytest.mark.parametrize(
    "placement", ["static", "sweep", "runtime", "repeat", "repeat_sync"]
)
def test_adc_windows_allow_touching_edges_and_shot_still_waits(placement):
    def initialize(prog):
        prog.declare_readout(0, length=100 / RO_MHZ)
        prog.add_loop("scan", 2)

    def one_read(prog):
        prog.trigger([0], timeline="default")
        prog.timeline_inc(130 / TPROC_MHZ)

    def body(prog):
        if placement in {"repeat", "repeat_sync"}:
            with prog.repeat("reads", 2):
                one_read(prog)
                if placement == "repeat_sync":
                    prog.delay_by_timeline()
        else:
            one_read(prog)
            if placement == "sweep":
                prog.timeline_inc(QickSweep1D("scan", 0, 40 / TPROC_MHZ))
            elif placement == "runtime":
                prog.timeline_inc("scan")
            prog.trigger([0], timeline="default")

    prog = _program(body, initialize)
    result = _shots(prog)
    expected = {"static": [0, 140], "sweep": [0, 180], "runtime": [0, 141]}
    starts = [
        [
            e.cycle - shot.start_cycle
            for e in shot.events
            if e.kind == "data" and e.value
        ]
        for shot in result
    ]
    assert starts == [[0, 140], expected.get(placement, [0, 140])]
    (wait,) = [inst for inst in prog.prog_list if inst["CMD"] == "WAIT"]
    wait_cycles = int(str(wait["TIME"]).removeprefix("@"))
    assert all(
        shot.body_end_cycle - shot.start_cycle + wait_cycles >= times[-1] + 140
        for shot, times in zip(result, starts, strict=True)
    )


def test_timeline_trigger_rejects_implicit_output_time_at_entry():
    def body(prog):
        with pytest.raises(ValueError, match="t=None"):
            prog.trigger([0], t=None, timeline="default")

    _program(body)


def test_resync_compiles_and_statically_decrements_every_timeline():
    def body(prog):
        prog.fork("default", into="other")
        prog.timeline_inc(40 / TPROC_MHZ)
        prog.timeline_inc(70 / TPROC_MHZ, timeline="other")
        prog.resync(10 / TPROC_MHZ)
        prog.pulse(0, "pulse", timeline="default", tag="default")
        prog.pulse(1, "p1", timeline="other", tag="other")

    prog = _program(body)
    assert prog.get_time_param("default", "t") * TPROC_MHZ == pytest.approx(30)
    assert prog.get_time_param("other", "t") * TPROC_MHZ == pytest.approx(60)
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
        prog.timeline_inc(QickSweep1D("scan", start / TPROC_MHZ, stop / TPROC_MHZ))
        prog.delay(10 / TPROC_MHZ)
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
        prog.timeline_inc(
            QickSweep1D("scan", 9 / TPROC_MHZ, 11 / TPROC_MHZ), timeline="drive"
        )
        prog.delay(10 / TPROC_MHZ)

    result = _shots(_program(body, initialize))
    assert [s.body_end_cycle - s.start_cycle for s in result] == [10, 11]


def test_crossing_swept_auto_maximum_places_at_each_points_later_end():
    def initialize(prog):
        prog.add_loop("scan", 2)

    def body(prog):
        prog.fork("default", into="drive")
        prog.pulse(0, "pulse", t=QickSweep1D("scan", 0, 100 / TPROC_MHZ))
        prog.timeline_inc(QickSweep1D("scan", 100 / TPROC_MHZ, 0), timeline="drive")
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
        prog.timeline_inc(QickSweep1D("scan", 10 / TPROC_MHZ, 30 / TPROC_MHZ))
        prog.timeline_inc(
            QickSweep1D("scan", 30 / TPROC_MHZ, 10 / TPROC_MHZ), timeline="other"
        )

    result = _shots(_program(body, initialize))
    assert [s.body_end_cycle - s.start_cycle for s in result] == [30, 30]
