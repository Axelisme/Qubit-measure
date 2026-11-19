"""Forward timeline transfers through program compilation and linked timing."""

import re
from collections.abc import Callable

import pytest
from qick.asm_v2 import AsmV2, QickSweep1D
from zcu_tools.program.v2 import TimelineBuilder, TimelineProgramV2
from zcu_tools.program.v2._timeline_selected import PlaybackChoice, SelectedPlayback
from zcu_tools.program.v2.modules import ConstWaveformCfg, LoadValue, Pulse

from tests.program._profiles import TPROC_MHZ, load_soccfg
from tests.program._timing_expectations import (
    generator_duration,
    readout_duration,
    tproc_ceil,
)
from tests.program.v2.timing_support import recompile, shots, waves


@pytest.mark.parametrize("synchronize", [False, True])
@pytest.mark.parametrize("pmem_size", [2048, 4096])
def test_forward_jumps_preserve_per_point_events_and_body_end(synchronize, pmem_size):
    profile = load_soccfg()
    profile["tprocs"][0]["pmem_size"] = pmem_size
    pulse = Pulse.Cfg(
        ch=0, nqz=1, freq=100, gain=0.5, waveform=ConstWaveformCfg(length=0.1)
    ).build("drive")

    def body(prog: TimelineBuilder | TimelineProgramV2) -> None:
        with prog.timeline("default"):
            prog.cond_jump("short", "depth", "Z", "-", 0)
            prog.timeline_inc(20 / TPROC_MHZ)
            prog.emit(pulse)
            if synchronize:
                prog.delay_by_timeline()
            prog.jump("depth")  # Same author name as the native sweep-loop label.
        prog.label("orphan")  # Still unreachable; only "short" resumes the path.
        prog.label("short")
        prog.emit(pulse)
        if synchronize:
            prog.delay_by_timeline()
        prog.label("depth")
        prog.emit(pulse)

    class Native(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("depth", 2)
            self.declare(pulse)

        def _body(self, cfg):
            body(self)

    builder = TimelineBuilder(profile, initial_delay=0, relax_delay=0)
    builder.add_loop("depth", 2)
    body(builder)
    built = builder.build()
    native = Native(profile, initial_delay=0, relax_delay=0)
    recompile(native)

    length = tproc_ceil(profile, generator_duration(profile, 0.1))
    short_end = length * 2 if synchronize else length * 2 + 20
    expected = [
        ([0, short_end - length], short_end),
        ([20, 20 + length], 20 + length * 2),
    ]
    for prog in (built, native):
        assert [
            (waves(s), s.body_end_cycle - s.start_cycle) for s in shots(prog)
        ] == expected


def _compile(
    body: Callable[[TimelineProgramV2], None],
    initialize: Callable[[TimelineProgramV2], None] | None = None,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("depth", 2)
            self.add_reg("duration")
            self.declare_gen(2, nqz=1)
            self.add_pulse(
                2, "p", style="const", freq=100, phase=0, gain=0.5, length=0.1
            )
            self.declare_readout(0, length=0.1)
            if initialize is not None:
                initialize(self)

        def _body(self, cfg):
            body(self)

    return Program(load_soccfg(n_gens=3), initial_delay=0, relax_delay=0)


@pytest.mark.parametrize(
    ("actions", "error"),
    [
        (
            lambda p: (p.label("again"), p.jump("again")),
            "timeline jump to 'again' is backward; general jumps must target a later label (use Repeat or a sweep for iteration)",
        ),
        (
            lambda p: (p.label("x"), p.label("x")),
            "timeline label 'x' is defined twice in _body",
        ),
        (
            lambda p: (p.cond_jump("x", "depth", "Z"), p.jump("a")),
            "timeline jump target 'x' is never defined in _body",
        ),
        (
            lambda p: (
                p.jump("done"),
                p.label("orphan"),
                p.asm_inst({"CMD": "NOP"}),
                p.label("done"),
            ),
            "unreachable timeline action after unconditional jump to 'done'; add a reachable label before this action",
        ),
        (lambda p: (p.label(""),), "timeline label '' requires a nonempty string"),
    ],
)
def test_invalid_author_transfers_fail_at_build(actions, error):
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        _compile(actions)


@pytest.mark.parametrize("region", ["Repeat", "Branch"])
def test_builder_rejects_region_local_transfers_at_build(region):
    builder = TimelineBuilder(load_soccfg())
    builder.add_loop("depth", 2)
    if region == "Repeat":
        with builder.repeat("i", 0):
            builder.label("x")
        prefix = "timeline label 'x' is inside Repeat 'i' body"
    else:
        with builder.branch(by="depth", align_timeline=True) as branch, branch.case(1):
            builder.jump("x")
        prefix = "timeline jump to 'x' is inside Branch 'depth' case 1"
    error = f"{prefix}; label, jump and cond_jump are allowed only at the top level of _body"
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        builder.build()


def test_native_shell_subroutine_and_raw_body_jump_keep_their_labels():
    def initialize(prog):
        prog.jump("_timeline_jump_0")
        prog.label(label="_timeline_jump_0")
        subroutine = AsmV2()
        subroutine.jump("ret_site")
        subroutine.nop()
        subroutine.label("ret_site")
        prog.add_subroutine("work", subroutine)

    def body(prog):
        prog.call("work")
        prog.asm_inst({"CMD": "JUMP", "LABEL": "raw_target"})
        prog.nop()
        prog.label("raw_target")
        prog.jump("work")  # The author target is not the native subroutine.
        prog.label("work")
        prog.nop()

    _compile(body, initialize)


@pytest.mark.parametrize("test", ["F", "NF"])
def test_peripheral_flag_tests_are_rejected_at_build(test):
    def body(prog):
        prog.cond_jump("join", "depth", test)
        prog.label("join")

    error = f"timeline cond_jump to 'join': peripheral-flag test {test!r} is unsupported in _body"
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        _compile(body)


def test_crossing_sweep_ends_merge_at_the_per_point_maximum():
    def body(prog):
        prog.cond_jump("ascending", "depth", "Z")
        prog.timeline_inc(QickSweep1D("depth", 30, 10) / TPROC_MHZ)
        prog.jump("join")
        prog.label("ascending")
        prog.timeline_inc(QickSweep1D("depth", 20, 40) / TPROC_MHZ)
        prog.label("join")
        prog.pulse(2, "p", timeline="default")

    # Each point executes the path with the smaller end, but adopts the
    # other edge's larger end when scheduling the shared pulse.
    assert [waves(s) for s in shots(_compile(body))] == [[30], [40]]


def test_fork_on_one_edge_does_not_supply_a_timeline_at_the_join():
    def body(prog):
        prog.cond_jump("join", "depth", "Z")
        prog.fork("default", into="optional")
        prog.label("join")
        prog.timeline_inc(0.1, timeline="optional")

    with pytest.raises(
        ValueError, match=r"^optional: unknown timeline; declare it with fork in _body$"
    ):
        _compile(body)


@pytest.mark.parametrize("kind", ["timeline 'default'", "generator 2", "ADC 0"])
def test_runtime_ends_require_the_same_value_on_every_edge(kind):
    def body(prog):
        prog.write_reg("duration", 40)
        prog.cond_jump("runtime_join" if kind != "ADC 0" else "other", "depth", "Z")
        if kind == "timeline 'default'":
            prog.timeline_inc("duration")
        else:
            prog.fork("default", into="offset")
            prog.timeline_inc("duration", timeline="offset")
            if kind == "generator 2":
                prog.pulse(2, "p", timeline="offset")
            else:
                prog.trigger([0], timeline="offset")
                prog.jump("runtime_join")
                prog.label("other")
                prog.trigger([0])
        prog.label("runtime_join")

    edge = "jump" if kind == "ADC 0" else "cond_jump"
    error = (
        f"label 'runtime_join': {kind} has different runtime ends on incoming edges "
        f"({edge} #1, fall-through); sync the paths or make their ends equal before the jump"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        _compile(body)


def test_same_runtime_timeline_end_survives_the_join():
    def body(prog):
        prog.write_reg("duration", 40)
        prog.timeline_inc("duration")
        prog.cond_jump("join", "depth", "Z")
        prog.nop()
        prog.label("join")
        prog.pulse(2, "p", timeline="default")

    assert [waves(s) for s in shots(_compile(body))] == [[40], [40]]


def test_native_auto_uses_worst_incoming_adc_end():
    def body(prog):
        prog.cond_jump("short", "depth", "Z")
        prog.trigger([0], t=60 / TPROC_MHZ)
        prog.jump("join")
        prog.label("short")
        prog.trigger([0])
        prog.label("join")
        prog.delay_auto()
        prog.pulse(2, "p")

    prog = _compile(body)
    expected = 60 + round(readout_duration(prog.soccfg, 0.1) * TPROC_MHZ)
    assert [waves(s) for s in shots(prog)] == [[expected], [expected]]


def test_unknown_adc_end_on_a_taken_edge_blocks_a_later_timeline_trigger():
    def body(prog):
        prog.cond_jump("known", "depth", "Z")
        prog.trigger([0], t=None)
        prog.jump("join")
        prog.label("known")
        prog.trigger([0])
        prog.label("join")
        prog.trigger([0], t=0.3, timeline="default")

    error = "default: channel 0 ADC overlap; cannot locate previous native trigger (t=None); overlap cycles unknown"
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        _compile(body)


def test_incoming_read_counts_must_match():
    def body(prog):
        prog.cond_jump("join", "depth", "Z")
        prog.trigger([0])
        prog.label("join")

    error = "label 'join': ADC 0 reads differ across incoming edges: cond_jump #1 -> 0, fall-through -> 1"
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        _compile(body)


def test_trigs_counts_equal_reads_on_each_path_not_all_source_triggers():
    def body(prog):
        prog.cond_jump("short", "depth", "Z")
        prog.trigger([0], t=0.2)
        prog.jump("join")
        prog.label("short")
        prog.trigger([0])
        prog.label("join")
        prog.trigger([0], t=0.4, timeline="default")

    prog = _compile(body)
    assert prog.ro_chs[0]["trigs"] == 2
    for shot in shots(prog):
        assert (
            sum(e.kind in ("data", "trigger") and bool(e.value) for e in shot.events)
            == 2
        )


@pytest.mark.parametrize(("initial", "consume"), [(None, True), (5, True), (5, False)])
def test_producer_time_must_be_defined_and_equal_on_every_edge(initial, consume):
    def body(prog):
        if initial is not None:
            prog.write_reg("duration", initial)
        prog.cond_jump("join", "depth", "Z")
        prog.write_reg("duration", 9)
        prog.label("join")
        if consume:
            prog.timeline_inc("duration")

    if consume:
        with pytest.raises(
            ValueError,
            match=r"^duration: unknown or uninitialized time producer bounds$",
        ):
            _compile(body)
    else:
        assert [s.body_end_cycle - s.start_cycle for s in shots(_compile(body))] == [
            0,
            0,
        ]


def test_label_merges_selection_proofs_for_playback():
    loader = LoadValue("select", [0, 0], "depth", "address")

    def body(prog):
        prog.cond_jump("second", "depth", "Z")
        loader.lower_selected(prog)
        prog.jump("join")
        prog.label("second")
        relation = loader.lower_selected(prog)
        prog.label("join")
        prog.pulse_by_reg(
            2,
            "address",
            timeline="default",
            selection=SelectedPlayback(
                relation.selector,
                "address",
                (PlaybackChoice("p"), PlaybackChoice("p")),
                relation,
            ),
        )

    assert [waves(s) for s in shots(_compile(body, lambda p: p.declare(loader)))] == [
        [0],
        [0],
    ]
