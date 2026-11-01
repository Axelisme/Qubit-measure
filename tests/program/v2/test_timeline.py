"""B1 scheduling through the public compiler and linked timing analysis."""

from collections.abc import Callable

import pytest
from qick.asm_v2 import QickSweep1D
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program._profiles import GEN_MHZ, TPROC_MHZ, load_soccfg
from tests.program.v2.timing_support import shots as _shots
from tests.program.v2.timing_support import waves as _waves


def _program(
    body: Callable[[TimelineProgramV2], None],
    initialize: Callable[[TimelineProgramV2], None] | None = None,
    *,
    reps: int = 1,
    relax_delay: float = 0,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            for ch in range(2):
                self.declare_gen(ch, nqz=1)
                self.add_pulse(
                    ch,
                    name=f"p{ch}",
                    style="const",
                    freq=1000,
                    phase=0,
                    gain=0.5,
                    length=0.1,
                )
            if initialize is not None:
                initialize(self)

        def _body(self, cfg):
            body(self)

    return Program(
        load_soccfg(2, 1), reps=reps, initial_delay=0, relax_delay=relax_delay
    )


@pytest.mark.parametrize("timeline", ["default", "drive"])
def test_timeline_pulses_are_serial_but_native_pulse_stays_at_reference(timeline):
    def body(prog):
        if timeline != "default":
            prog.fork("default", into=timeline)
        prog.timeline_inc(10 / TPROC_MHZ, timeline=timeline)
        prog.pulse(0, "p0", timeline=timeline)
        prog.pulse(1, "p1", timeline=timeline)
        prog.pulse(0, "p0", t=0)

    (shot,) = _shots(_program(body))
    # Each .1 us waveform is 38 generator cycles = 42.56 tProc cycles,
    # so the next timeline start is ceil(10 + 42.56) = 53.
    assert _waves(shot) == [10, 53, 0]
    assert shot.body_end_cycle == 96
    assert shot.end_cycle == 96


@pytest.mark.parametrize("subcycle, scalar_end", [(0.2, 2), (5e-11, 0), (2e-10, 2)])
def test_each_delay_ceil_and_author_sweep_uses_native_affine_steps(
    subcycle, scalar_end
):
    def body(prog):
        prog.timeline_inc(subcycle / TPROC_MHZ)
        prog.timeline_inc(subcycle / TPROC_MHZ)
        prog.timeline_inc(
            QickSweep1D("scan", 0.2 / TPROC_MHZ, 3.8 / TPROC_MHZ), tag="end"
        )
        prog.pulse(0, "p0", timeline="default", tag="pulse")

    prog = _program(body, lambda p: p.add_loop("scan", 4))
    # Scalar tails inside 1e-10 ticks are noise; larger tails ceil each time.
    # Native sweep truncates its 1.2-tick step to one, giving 0,1,2,3.
    expected = [scalar_end, scalar_end + 1, scalar_end + 2, scalar_end + 3]
    assert [_waves(s) for s in _shots(prog)] == [[tick] for tick in expected]
    assert prog.get_time_param("end", "t", as_array=True) * TPROC_MHZ == pytest.approx(
        expected
    )
    assert prog.get_time_param(
        "pulse", "t", as_array=True
    ) * TPROC_MHZ == pytest.approx(expected)
    assert prog.compile_datamem() is None


def test_sync_rebases_default_and_other_timelines_and_shell_follows_body():
    def body(prog):
        prog.fork("default", into="other")
        prog.timeline_inc(40 / TPROC_MHZ)
        prog.timeline_inc(70 / TPROC_MHZ, timeline="other")
        prog.delay_by_timeline(timeline="default")
        prog.pulse(0, "p0")
        prog.pulse(1, "p1", timeline="other")

    shots = _shots(_program(body, reps=2, relax_delay=10 / TPROC_MHZ))
    assert [_waves(s) for s in shots] == [[40, 70], [40, 70]]
    assert [(s.start_cycle, s.body_end_cycle, s.end_cycle) for s in shots] == [
        (0, 113, 123),
        (123, 236, 246),
    ]


@pytest.mark.parametrize("advance, expected", [(20, 40), (40, 40), (60, 60)])
def test_native_delay_subtracts_each_timeline_with_zero_floor(advance, expected):
    def body(prog):
        prog.timeline_inc(40 / TPROC_MHZ)
        prog.delay(advance / TPROC_MHZ)
        prog.pulse(0, "p0", timeline="default")

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [expected]


def test_fork_copies_current_source_end_for_one_or_multiple_names_each_compile():
    def body(prog):
        prog.timeline_inc(10 / TPROC_MHZ)
        prog.fork("default", into=["a", "b"])
        prog.timeline_inc(5 / TPROC_MHZ, timeline="a")
        prog.fork("a", into="c")
        prog.timeline_inc(20 / TPROC_MHZ)
        prog.pulse(0, "p0", timeline="b")
        prog.pulse(1, "p1", timeline="c")

    prog = _program(body)
    before = _shots(prog)
    assert _waves(before[0]) == [10, 15]
    assert before[0].body_end_cycle == 58
    prog.compile()
    assert _shots(prog) == before


@pytest.mark.parametrize(
    "body",
    [
        lambda p: p.fork("unknown", into="x"),
        lambda p: p.fork("default", into="default"),
        lambda p: p.fork("default", into=["x", "x"]),
    ],
)
def test_fork_rejects_unknown_source_and_duplicate_targets(body):
    with pytest.raises(ValueError, match="timeline"):
        _program(body)


@pytest.mark.parametrize(
    "operation", ["pulse", "trigger", "delay", "delay_by_timeline", "config"]
)
def test_timeline_operations_reject_undeclared_names(operation):
    def body(prog):
        if operation == "pulse":
            prog.pulse(0, "p0", timeline="missing")
        elif operation == "trigger":
            prog.trigger(timeline="missing")
        elif operation == "delay":
            prog.timeline_inc(0.1, timeline="missing")
        elif operation == "config":
            prog.send_readoutconfig(0, "ro", timeline="missing")
        else:
            prog.delay_by_timeline(timeline="missing")

    def initialize(prog):
        prog.declare_readout(0, length=0.4)
        prog.add_readoutconfig(0, "ro", freq=100)

    with pytest.raises(ValueError, match="missing.*fork"):
        _program(body, initialize)


def test_join_merges_only_selected_ends_without_advancing_reference():
    def body(prog):
        prog.timeline_inc(10 / TPROC_MHZ)
        prog.fork("default", into=["a", "b", "omitted"])
        prog.timeline_inc(20 / TPROC_MHZ, timeline="a")
        prog.timeline_inc(30 / TPROC_MHZ, timeline="b")
        prog.timeline_inc(90 / TPROC_MHZ, timeline="omitted")
        prog.join("a", "b", into="default")
        prog.pulse(0, "p0", timeline="default")
        prog.pulse(1, "p1", timeline="a")
        prog.join("a", into="b")  # The source a remains available.

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [40, 30]
    assert shot.body_end_cycle == 100
    assert shot.end_cycle == 100


def test_join_retains_later_target_and_does_not_consume_sources():
    def body(prog):
        prog.fork("default", into=["source", "target"])
        prog.timeline_inc(10 / TPROC_MHZ, timeline="source")
        prog.timeline_inc(50 / TPROC_MHZ, timeline="target")
        prog.join("source", into="target")
        prog.pulse(0, "p0", timeline="target")
        prog.pulse(1, "p1", timeline="source")

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [50, 10]
    assert (shot.body_end_cycle, shot.end_cycle) == (93, 93)


@pytest.mark.parametrize(
    "source,target", [("missing", "default"), ("default", "missing")]
)
def test_join_rejects_undeclared_names(source, target):
    with pytest.raises(ValueError, match="missing.*fork"):
        _program(lambda p: p.join(source, into=target))


def test_join_proves_swept_order_by_difference_and_resets_each_shot():
    def initialize(prog):
        prog.add_loop("scan", 3)

    def body(prog):
        prog.fork("default", into=["a", "b"])
        prog.timeline_inc(
            QickSweep1D("scan", 10 / TPROC_MHZ, 30 / TPROC_MHZ), timeline="a"
        )
        prog.timeline_inc(
            QickSweep1D("scan", 20 / TPROC_MHZ, 40 / TPROC_MHZ), timeline="b"
        )
        prog.join("a", "b", into="default")
        prog.pulse(0, "p0", timeline="default")
        prog.pulse(1, "p1", timeline="a")

    prog = _program(body, initialize, reps=2, relax_delay=5 / TPROC_MHZ)
    shots = _shots(prog)
    assert [_waves(s) for s in shots] == [[20, 10], [30, 20], [40, 30]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in shots] == [63, 73, 83] * 2
    assert [s.end_cycle - s.start_cycle for s in shots] == [68, 78, 88] * 2
    prog.compile()
    assert _shots(prog) == shots


def test_join_uses_pointwise_crossing_swept_ends():
    def initialize(prog):
        prog.add_loop("scan", 3)

    def body(prog):
        prog.fork("default", into="a")
        prog.timeline_inc(QickSweep1D("scan", 10 / TPROC_MHZ, 30 / TPROC_MHZ))
        prog.timeline_inc(20 / TPROC_MHZ, timeline="a")
        prog.join("a", into="default")

    result = _shots(_program(body, initialize))
    assert [s.body_end_cycle - s.start_cycle for s in result] == [20, 20, 30]


def test_join_without_fork_rejoining_still_closes_all_body_ends():
    def body(prog):
        prog.fork("default", into="unjoined")
        prog.timeline_inc(90 / TPROC_MHZ, timeline="unjoined")
        prog.pulse(0, "p0", timeline="default")

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [0]
    assert (shot.body_end_cycle, shot.end_cycle) == (90, 90)


def test_fork_is_only_available_in_body():
    def initialize(prog):
        with pytest.raises(ValueError, match="_body"):
            prog.fork("default", into="x")

    _program(lambda p: None, initialize)


def test_cross_timeline_pulse_on_same_channel_cannot_bypass_overlap_guard():
    def body(prog):
        prog.fork("default", into="other")
        prog.pulse(0, "p0", timeline="default")
        prog.pulse(0, "p0", timeline="other")

    with pytest.raises(ValueError, match="other.*channel 0.*overlap"):
        _program(body)


def test_same_channel_pulses_may_touch_exactly_and_cross_channel_pulses_overlap():
    def initialize(prog):
        prog.add_pulse(
            0, "exact", style="const", freq=1000, phase=0, gain=0.5, length=25 / GEN_MHZ
        )  # 25 generator ticks =28 tProc ticks.

    def body(prog):
        prog.fork("default", into=["other", "parallel"])
        prog.pulse(0, "exact", timeline="default")
        prog.timeline_inc(28 / TPROC_MHZ, timeline="other")
        prog.pulse(0, "exact", timeline="other")
        prog.pulse(1, "p1", timeline="parallel")

    (shot,) = _shots(_program(body, initialize))
    assert _waves(shot) == [0, 28, 0]
    assert shot.body_end_cycle == 56


def test_native_same_channel_overlap_keeps_qick_behavior():
    def body(prog):
        prog.pulse(0, "p0", t=0)
        prog.pulse(0, "p0", t=0)

    (shot,) = _shots(_program(body))
    assert _waves(shot) == [0, 0]


def test_recompile_and_sweep_shots_reset_timeline_ends():
    def initialize(prog):
        prog.add_loop("scan", 3)

    def body(prog):
        prog.timeline_inc(QickSweep1D("scan", 10 / TPROC_MHZ, 30 / TPROC_MHZ))
        prog.pulse(0, "p0", timeline="default")

    prog = _program(body, initialize, reps=2)
    first = _shots(prog)
    assert [_waves(s) for s in first] == [[10], [20], [30]] * 2
    assert [s.body_end_cycle - s.start_cycle for s in first] == [53, 63, 73] * 2
    prog.compile()
    assert _shots(prog) == first
