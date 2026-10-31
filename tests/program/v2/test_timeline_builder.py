# tolerated(file-size): Builder recording and replay share native compile fixtures.
"""Builder recording and native-program compilation share a public contract.

The handwritten parity oracle remains until TimelineProgramV2 author hooks are
retired; changes to the common compiler must preserve parity, not old ASM bytes.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from qick import QickConfig
from qick.asm_v2 import QickParam, QickSweep1D
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.program import v2
from zcu_tools.program.v2.mocksoc import make_mux_mock_soccfg
from zcu_tools.program.v2.modules.readout import (
    MuxReadoutCfg,
    PulseReadout,
    PulseReadoutCfg,
)
from zcu_tools.program.v2.modules.reset import TwoPulseResetCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program.v2.timing_support import recompile, shots, waves


@pytest.mark.parametrize("frontend", ["builder", "native"])
@pytest.mark.parametrize("operation", ["timeline_inc", "delay_by_timeline"])
def test_named_time_operations_keep_end_and_reference_separate(
    mock_soccfg, frontend, operation
):
    pulse = _pulse("drive")
    gap = v2.Delay("gap", 100 / 430.08)

    def body(program: v2.TimelineBuilder | v2.TimelineProgramV2) -> None:
        program.emit(pulse)
        program.fork("default", into="work")
        if operation == "timeline_inc":
            program.timeline_inc(100 / 430.08, timeline="work")
            program.emit(pulse, timeline="work", tag="after")
        else:
            program.emit(gap, timeline="work")
            program.delay_by_timeline(timeline="work")
            program.emit(pulse, tag="after")

    prog = _compile_scoped(mock_soccfg, frontend, body, (pulse, gap))
    (shot,) = shots(prog)
    assert waves(shot) == [0, 143]
    assert (shot.body_end_cycle, shot.end_cycle) == (186, 186)
    expected_offset = 143 if operation == "timeline_inc" else 0
    assert prog.get_time_param("after", "t") * 430.08 == pytest.approx(expected_offset)


@pytest.mark.parametrize("sync_at_end", [False, True])
def test_builder_repeat_keeps_author_reference_advances(mock_soccfg, sync_at_end):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    drive = _pulse("drive")
    with builder.repeat("r", 2):
        builder.emit(drive)
        builder.timeline_inc(5 / 430.08)
        if sync_at_end:
            builder.delay_by_timeline()
    builder.emit(drive)
    prog = builder.build()
    (shot,) = shots(prog)
    assert waves(shot) == [0, 48, 96]
    assert (shot.body_end_cycle, shot.end_cycle) == (139, 139)
    if sync_at_end:
        assert len(prog.reg_dict) == 2
    else:
        assert len(prog.reg_dict) > 2


@pytest.mark.parametrize("align", [False, True])
def test_builder_replays_variable_repeat_inside_case(mock_soccfg, align):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0, reps=2)
    builder.add_loop("scan", 2)
    drive = _pulse("drive")
    with (
        builder.branch(by="scan", align_timeline=align) as cases,
        cases.case(0),
        builder.repeat("r", 2),
    ):
        builder.emit(drive)
        builder.timeline_inc(5 / 430.08)
    builder.emit(drive)
    prog = builder.build()
    assert [waves(s) for s in shots(prog)] == [[0, 48, 96], [96 if align else 0]] * 2
    prog.compile()
    assert [waves(s) for s in shots(prog)] == [[0, 48, 96], [96 if align else 0]] * 2


def test_builder_places_on_external_timeline_after_runtime_repeat_and_branch(
    mock_soccfg,
):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0, reps=2)
    builder.add_loop("scan", 3)
    drive = _pulse("drive")
    builder.fork("default", into="external")
    builder.timeline_inc(100 / 430.08, timeline="external")
    builder.emit(drive)
    builder.delay_by_timeline()
    with builder.repeat("r", "scan", range_hint=(0, 2)):
        builder.timeline_inc(15 / 430.08)
        builder.delay_by_timeline()
    with builder.branch(by="scan", align_timeline=False, timeline="external") as branch:
        with branch.case(0):
            builder.timeline_inc(5 / 430.08)
        with branch.case(2):
            builder.timeline_inc(15 / 430.08)
    builder.emit(drive, timeline="external")

    prog = builder.build()
    assert [waves(shot) for shot in shots(prog)] == [[0, 105], [0, 100], [0, 115]] * 2
    assert [shot.body_end_cycle - shot.start_cycle for shot in shots(prog)] == [
        148,
        143,
        158,
    ] * 2


def test_builder_forwards_author_register_time_hint(mock_soccfg):
    from zcu_tools.program.v2.modules import LoadValue

    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    builder.add_loop("scan", 3)
    builder.emit(LoadValue("time", [10, 20, 30], "scan", "duration"))
    builder.timeline_inc("duration", range_hint=(10, 30))
    builder.emit(_pulse("drive"))
    assert [waves(s) for s in shots(builder.build())] == [[10], [20], [30]]


def _readout(name="read", *, gain: float | QickParam = 0.5):
    return PulseReadout(
        name,
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            ro_ch=0,
            freq=100,
            gain=gain,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=0.2,
            post_delay=0.05,
        ),
    )


@pytest.mark.parametrize("align_timeline", [True, False])
def test_sparse_indexed_cases_match_native_and_leave_missing_points_empty(
    mock_soccfg, align_timeline
):
    def body(program: v2.TimelineBuilder | v2.TimelineProgramV2, pulse: v2.Pulse):
        with program.branch(by="scan", align_timeline=align_timeline) as branch:
            # Declaration order does not change the selector's zero-based index.
            with branch.case(3):
                program.emit(pulse)
                program.emit(pulse)
            with branch.case(1):
                program.emit(pulse)
        with program.branch(by="scan", align_timeline=align_timeline):
            pass  # A branch may leave every point empty.

    pulse = _pulse("drive")
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    builder.add_loop("scan", 4)
    body(builder, pulse)
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 4)
            self.drive = _pulse("drive")
            self.declare(self.drive)

        def _body(self, cfg):
            body(self, self.drive)

    expected = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    assert actual.prog_list == expected.prog_list
    np.testing.assert_array_equal(actual.binprog["pmem"], expected.binprog["pmem"])
    assert [waves(shot) for shot in shots(actual)] == [[], [0], [], [0, 43]]
    assert [shot.body_end_cycle - shot.start_cycle for shot in shots(actual)] == (
        [86, 86, 86, 86] if align_timeline else [0, 43, 0, 86]
    )
    actual.compile()
    np.testing.assert_array_equal(actual.binprog["pmem"], expected.binprog["pmem"])


@pytest.mark.parametrize("frontend", ["builder", "native"])
@pytest.mark.parametrize(
    ("index", "message"), [(0, "duplicate"), (-1, "index"), (2, "index")]
)
def test_indexed_branch_rejects_duplicate_and_out_of_range_points(
    mock_soccfg, frontend, index, message
):
    def body(program: v2.TimelineBuilder | v2.TimelineProgramV2):
        with program.branch(by="scan", align_timeline=True) as branch:
            with branch.case(0):
                program.timeline_inc(0.1)
            with branch.case(index):
                program.timeline_inc(0.2)

    with pytest.raises(ValueError, match=message):
        _compile_scoped(mock_soccfg, frontend, body, loops=(("scan", 2),))


@pytest.mark.parametrize("frontend", ["builder", "native"])
@pytest.mark.parametrize("align_timeline", [True, False])
def test_branch_rejects_emission_outside_cases_at_build(
    mock_soccfg, frontend, align_timeline
):
    pulse = _pulse("outside")

    def body(program: v2.TimelineBuilder | v2.TimelineProgramV2):
        with program.branch(by="scan", align_timeline=align_timeline):
            program.emit(pulse)

    with pytest.raises(ValueError, match="before or after.*with"):
        _compile_scoped(
            mock_soccfg, frontend, body, modules=(pulse,), loops=(("scan", 2),)
        )


@pytest.mark.parametrize("frontend", ["builder", "native"])
@pytest.mark.parametrize("align_timeline", [True, False])
@pytest.mark.parametrize("operation", ["delay", "delay_auto", "delay_by_timeline"])
def test_branch_rejects_reference_advances_at_build(
    mock_soccfg, frontend, align_timeline, operation
):
    def body(program: v2.TimelineBuilder | v2.TimelineProgramV2):
        with (
            program.branch(by="scan", align_timeline=align_timeline) as branch,
            branch.case(0),
        ):
            if operation == "delay":
                program.delay(0.1)
            elif operation == "delay_auto":
                program.delay_auto(0.1)
            elif operation == "delay_by_timeline":
                program.delay_by_timeline()

    with pytest.raises(ValueError, match="timeline_inc"):
        _compile_scoped(mock_soccfg, frontend, body, loops=(("scan", 2),))


@pytest.mark.parametrize("heralded", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_sparse_preparations_after_reset_and_herald_compile_on_forked_timelines(
    heralded, reset
):
    soccfg = make_mux_mock_soccfg(3, 2, gen_ch=0, ro_chs=(0, 1))
    builder = v2.TimelineBuilder(soccfg, initial_delay=0, relax_delay=1)
    builder.add_loop("prepared_f1", 2)  # 0 = g, 1 = e (outer)
    builder.add_loop("prepared_f2", 2)  # 0 = g, 1 = e (inner)
    x180_f1, x180_f2 = _pulse("x180_f1", 1), _pulse("x180_f2", 2)
    readout = MuxReadoutCfg(
        gen_ch=0,
        nqz=1,
        mixer_freq=1000,
        length=0.1,
        ro_length=0.2,
        post_delay=0.05,
        tones=[
            MuxReadoutTone(ro_ch=0, freq=1000, gain=0.1),
            MuxReadoutTone(ro_ch=1, freq=1080, gain=0.1),
        ],
    ).build("readout")
    if reset:
        builder.emit(
            TwoPulseResetCfg(
                pulse1_cfg=v2.PulseCfg(
                    ch=1,
                    nqz=1,
                    freq=100,
                    gain=0.5,
                    waveform=ConstWaveformCfg(length=0.1),
                ),
                pulse2_cfg=v2.PulseCfg(
                    ch=2,
                    nqz=1,
                    freq=100,
                    gain=0.5,
                    waveform=ConstWaveformCfg(length=0.1),
                ),
            ).build("reset")
        )
    if heralded:
        builder.emit(readout, tag="herald")
    # wait for herald/reset to finish before the branch-selected preparation
    builder.delay_by_timeline()
    builder.fork("default", into=["f1", "f2"])
    with (
        builder.branch(by="prepared_f1", align_timeline=True, timeline="f1") as cases,
        cases.case(1),
    ):
        builder.emit(x180_f1)
    with (
        builder.branch(by="prepared_f2", align_timeline=True, timeline="f2") as cases,
        cases.case(1),
    ):
        builder.emit(x180_f2)
    builder.join("f1", "f2", into="default")
    builder.emit(readout, tag="terminal")
    prog = builder.build()
    print(f"heralded={heralded} reset={reset} data_registers={len(prog.reg_dict)}")
    drive_ports = {soccfg["gens"][ch]["tproc_ch"] for ch in (1, 2)}
    scheduled = shots(prog)
    drives = [
        [
            event
            for event in shot.events
            if event.kind == "wave" and event.port in drive_ports
        ]
        for shot in scheduled
    ]
    assert [len(events) for events in drives] == [
        2 * int(reset),
        2 * int(reset) + 1,
        2 * int(reset) + 1,
        2 * int(reset) + 2,
    ]
    print(f"ee_cycles={[event.cycle for event in drives[-1][-2:]]}")
    # Both selected preparations must start together, after the optional prefix.
    assert drives[-1][-2].cycle == drives[-1][-1].cycle


def test_builder_sweep_reused_readout_and_timing_match_native(mock_soccfg: QickConfig):
    builder = v2.TimelineBuilder(
        mock_soccfg, reps=2, initial_delay=0.1, relax_delay=0.3
    )
    builder.add_loop("outer", 2)
    gain = builder.add_sweep(
        "gain", v2.SweepCfg(start=0.2, stop=0.4, step=0.1, expts=3)
    )
    readout = _readout(gain=gain)
    builder.emit(readout, tag="first")
    builder.timeline_inc(0.05)
    builder.delay_by_timeline()
    builder.delay(0.1, tag="gap")
    builder.emit(readout, tag="second")
    builder.delay_auto(0.1, gens=False, ros=True, tag="wait")
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("outer", 2)
            self.add_loop("gain", 3)
            self.readout = _readout(gain=QickSweep1D("gain", 0.2, 0.4))
            self.declare(self.readout)

        def _body(self, cfg):
            self.emit(self.readout, tag="first")
            self.timeline_inc(0.05)
            self.delay_by_timeline()
            self.delay(0.1, tag="gap")
            self.emit(self.readout, tag="second")
            self.delay_auto(0.1, gens=False, ros=True, tag="wait")

    expected = Native(mock_soccfg, reps=2, initial_delay=0.1, relax_delay=0.3)
    assert actual.prog_list == expected.prog_list
    assert actual.sweep_loops == expected.sweep_loops
    for tag in ("first", "second", "gap", "wait"):
        assert actual.get_time_param(tag, "t") == expected.get_time_param(tag, "t")
    actual.compile()
    assert actual.prog_list == expected.prog_list


def test_builder_nested_fixed_repeat_matches_native_and_reuses_declarations(
    mock_soccfg,
):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    readout = _readout()
    builder.emit(readout, tag="before")
    builder.delay_by_timeline()
    with builder.repeat("outer", 2):
        builder.emit(readout, tag="outer_read")
        builder.delay_by_timeline()
        with builder.repeat("inner", 3):
            builder.timeline_inc(0.1)
            builder.emit(readout, tag="inner_read")
            builder.delay_by_timeline()
        builder.delay_by_timeline()
    builder.emit(readout, tag="after")
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(readout)

        def _body(self, cfg):
            self.emit(readout, tag="before")
            self.delay_by_timeline()
            with self.repeat("outer", 2):
                self.emit(readout, tag="outer_read")
                self.delay_by_timeline()
                with self.repeat("inner", 3):
                    self.timeline_inc(0.1)
                    self.emit(readout, tag="inner_read")
                    self.delay_by_timeline()
                self.delay_by_timeline()
            self.emit(readout, tag="after")

    expected = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    assert actual.prog_list == expected.prog_list
    actual.compile()
    assert actual.prog_list == expected.prog_list


def test_builder_branch_timeline_delays_inside_repeat_match_native(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    builder.add_loop("scan", 2)
    readout = _readout()
    builder.delay_by_timeline()
    with builder.repeat("pairs", 2):
        with builder.branch(by="scan", align_timeline=True) as branch:
            with branch.case(0):
                builder.emit(readout)
                builder.timeline_inc(0.1)
            with branch.case(1):
                builder.timeline_inc(0.4)
                builder.emit(readout)
                builder.timeline_inc(0.1)
        builder.delay_by_timeline()
    builder.emit(readout)
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(readout)

        def _body(self, cfg):
            self.delay_by_timeline()
            with self.repeat("pairs", 2):
                with self.branch(by="scan", align_timeline=True) as branch:
                    with branch.case(0):
                        self.emit(readout)
                        self.timeline_inc(0.1)
                    with branch.case(1):
                        self.timeline_inc(0.4)
                        self.emit(readout)
                        self.timeline_inc(0.1)
                self.delay_by_timeline()
            self.emit(readout)

    expected = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    assert actual.prog_list == expected.prog_list
    actual.compile()
    assert actual.prog_list == expected.prog_list


def test_builder_fork_emit_join_matches_native_and_captures_target_names(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    readout, pulse = _readout(), _pulse("drive")
    targets = ["read", "drive"]
    builder.fork("default", into=targets)
    targets.clear()
    builder.emit(readout, timeline="read")
    builder.emit(pulse, timeline="drive")
    builder.join("read", "drive", into="default")
    builder.delay_by_timeline()
    builder.emit(readout)
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(readout)
            self.declare(pulse)

        def _body(self, cfg):
            self.fork("default", into=["read", "drive"])
            self.emit(readout, timeline="read")
            self.emit(pulse, timeline="drive")
            self.join("read", "drive", into="default")
            self.delay_by_timeline()
            self.emit(readout)

    expected = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    assert actual.prog_list == expected.prog_list
    actual.compile()
    assert actual.prog_list == expected.prog_list


@pytest.mark.parametrize("structure", ["repeat", "branch", "timeline"])
def test_builder_build_requires_closed_recording_contexts(mock_soccfg, structure):
    builder = v2.TimelineBuilder(mock_soccfg)
    if structure == "repeat":
        context = builder.repeat("body", 2)
    elif structure == "branch":
        context = builder.branch(by="scan", align_timeline=True)
    else:
        context = builder.timeline("default")
    if structure == "branch":
        builder.add_loop("scan", 1)
    with context as branch:
        with pytest.raises(ValueError, match="(open|close|context)"):
            builder.build()
        if branch is not None:
            with branch.case(0):
                builder.timeline_inc(0.1)
        else:
            builder.timeline_inc(0.1)
    # Failed admission must not freeze or discard the unfinished recording.
    assert builder.build().prog_list


def _invalid_structure(
    program: v2.TimelineBuilder | v2.TimelineProgramV2, scenario: str
) -> None:
    if scenario == "external-repeat":
        program.fork("default", into="external")
        program.delay_by_timeline()
        with program.repeat("body", 2):
            program.timeline_inc(0.1, timeline="external")
            program.delay_by_timeline()
    elif scenario == "expired-local":
        program.delay_by_timeline()
        with program.repeat("body", 2):
            program.fork("default", into="local")
            program.join("local", into="default")
            program.delay_by_timeline()
        program.delay_by_timeline(timeline="local")
    elif scenario == "negative-repeat":
        program.delay_by_timeline()
        with program.repeat("body", -1):
            pass
            program.delay_by_timeline()
    elif scenario == "fixed-range":
        program.delay_by_timeline()
        with program.repeat("body", 2, range_hint=(1, 2)):
            pass
            program.delay_by_timeline()
    elif scenario == "unknown-join":
        program.join("missing", into="default")
    elif scenario == "duplicate-fork":
        program.fork("default", into=["same", "same"])
    elif scenario == "selector":
        with (
            program.branch(by="measurement", align_timeline=True) as branch,
            branch.case(0),
        ):
            program.delay(0.1)
    elif scenario == "case-index":
        with program.branch(by="scan", align_timeline=True) as branch, branch.case(2):
            program.delay(0.1)
    else:
        raise ValueError(f"unknown scenario: {scenario}")


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        ("external-repeat", "external"),
        ("expired-local", "local: unknown timeline"),
        ("negative-repeat", "repeat n"),
        ("fixed-range", "range_hint"),
        ("unknown-join", "missing: unknown timeline"),
        ("duplicate-fork", "new, distinct"),
        ("selector", "selector"),
        ("case-index", "branch case index"),
    ],
)
def test_builder_structure_errors_propagate_program_messages_at_build(
    mock_soccfg, scenario, message
):
    builder = v2.TimelineBuilder(mock_soccfg)
    builder.add_loop("scan", 2)
    _invalid_structure(builder, scenario)

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)

        def _body(self, cfg):
            _invalid_structure(self, scenario)

    with pytest.raises(ValueError, match=message) as expected:
        Native(mock_soccfg)
    with pytest.raises(ValueError, match=message) as actual:
        builder.build()
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("count", [0, 2])
def test_builder_repeat_rebases_external_end_and_expires_local_names(
    mock_soccfg, count
):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    readout = _readout()
    builder.fork("default", into=["external", "parent"])
    builder.timeline_inc(2.0, timeline="external")
    builder.timeline_inc(0.1, timeline="parent")
    builder.delay_by_timeline(timeline="parent")
    with builder.repeat("body", count, timeline="parent"):
        builder.fork("parent", into="local")
        builder.emit(readout, timeline="local")
        builder.join("local", into="parent")
        builder.delay_by_timeline(timeline="parent")
    builder.fork("parent", into="local")
    builder.join("external", into="local")
    builder.emit(readout, timeline="local")
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(readout)

        def _body(self, cfg):
            self.fork("default", into=["external", "parent"])
            self.timeline_inc(2.0, timeline="external")
            self.timeline_inc(0.1, timeline="parent")
            self.delay_by_timeline(timeline="parent")
            with self.repeat("body", count, timeline="parent"):
                self.fork("parent", into="local")
                self.emit(readout, timeline="local")
                self.join("local", into="parent")
                self.delay_by_timeline(timeline="parent")
            self.fork("parent", into="local")
            self.join("external", into="local")
            self.emit(readout, timeline="local")

    assert (
        actual.prog_list
        == Native(mock_soccfg, initial_delay=0, relax_delay=0).prog_list
    )


def test_builder_register_repeat_forwards_bounds_and_preserves_remaining_end(
    mock_soccfg,
):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    builder.add_loop("scan", 2)
    loader = v2.LoadValue("count", [2, 3], idx_reg="scan", val_reg="n")
    readout = _readout()
    builder.emit(loader)
    builder.fork("default", into="external")
    builder.timeline_inc(2.0, timeline="external")
    builder.delay_by_timeline()
    with builder.repeat("body", "n", range_hint=(2, 3)):
        builder.emit(readout)
        builder.delay_by_timeline()
    builder.join("external", into="default")
    builder.emit(readout)
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(loader)
            self.declare(readout)

        def _body(self, cfg):
            self.emit(loader)
            self.fork("default", into="external")
            self.timeline_inc(2.0, timeline="external")
            self.delay_by_timeline()
            with self.repeat("body", "n", range_hint=(2, 3)):
                self.emit(readout)
                self.delay_by_timeline()
            self.join("external", into="default")
            self.emit(readout)

    expected = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    assert actual.prog_list == expected.prog_list
    np.testing.assert_array_equal(actual.compile_datamem(), expected.compile_datamem())


def test_builder_case_handle_rejects_overlap_and_expires_on_exit(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg)
    builder.add_loop("scan", 1)
    with builder.branch(by="scan", align_timeline=True) as branch, branch.case(0):
        with pytest.raises(ValueError, match="overlap"), branch.case(0):
            pass
        builder.timeline_inc(0.1)
    with pytest.raises(ValueError, match="open"), branch.case(0):
        pass
    assert builder.build().prog_list
    with pytest.raises(ValueError, match="frozen"), branch.case(0):
        pass


def test_native_add_sweep_declares_axis_and_returns_uniform_parameter(mock_soccfg):
    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            gain = self.add_sweep(
                "gain", v2.SweepCfg(start=0.2, stop=0.4, step=0.1, expts=3)
            )
            self.readout = _readout(gain=gain)
            self.declare(self.readout)

        def _body(self, cfg):
            self.emit(self.readout)

    class Explicit(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("gain", 3)
            self.readout = _readout(gain=QickSweep1D("gain", 0.2, 0.4))
            self.declare(self.readout)

        def _body(self, cfg):
            self.emit(self.readout)

    actual = Native(mock_soccfg)
    assert actual.prog_list == Explicit(mock_soccfg).prog_list
    assert [(axis.name, axis.count) for axis in actual.sweep_loops] == [
        ("reps", 1),
        ("gain", 3),
    ]


@pytest.mark.parametrize(
    "operation",
    [
        "build",
        "loop",
        "sweep",
        "emit",
        "pulse",
        "trigger",
        "delay",
        "auto",
        "end",
        "delay_by_timeline",
        "fork",
        "join",
        "repeat",
        "branch",
        "timeline",
    ],
)
def test_builder_freezes_after_build(mock_soccfg, operation):
    builder = v2.TimelineBuilder(mock_soccfg)
    builder.build()
    actions = {
        "build": builder.build,
        "loop": lambda: builder.add_loop("later", 2),
        "sweep": lambda: builder.add_sweep(
            "later", v2.SweepCfg(start=0, stop=1, step=1, expts=2)
        ),
        "emit": lambda: builder.emit(_readout()),
        "pulse": lambda: builder.pulse(0, "later"),
        "trigger": builder.trigger,
        "delay": lambda: builder.delay(0.1),
        "auto": builder.delay_auto,
        "end": lambda: builder.timeline_inc(0.1),
        "delay_by_timeline": builder.delay_by_timeline,
        "fork": lambda: builder.fork("default", into="later"),
        "join": lambda: builder.join("default", into="default"),
        "repeat": lambda: builder.repeat("later", 2),
        "branch": lambda: builder.branch(by="later", align_timeline=True),
        "timeline": lambda: builder.timeline("default"),
    }
    with pytest.raises(ValueError, match="(frozen|built)"):
        actions[operation]()


@pytest.mark.parametrize("first", ["loop", "sweep"])
@pytest.mark.parametrize("second", ["loop", "sweep"])
def test_duplicate_builder_axes_raise_at_recording(mock_soccfg, first, second):
    builder = v2.TimelineBuilder(mock_soccfg)
    spec = v2.SweepCfg(start=0, stop=1, step=1, expts=2)
    actions = {
        "loop": lambda: builder.add_loop("scan", 2),
        "sweep": lambda: builder.add_sweep("scan", spec),
    }
    actions[first]()
    with pytest.raises(ValueError, match="scan"):
        actions[second]()
    # Rejection leaves only the accepted axis.
    assert [(axis.name, axis.count) for axis in builder.build().sweep_loops] == [
        ("reps", 1),
        ("scan", 2),
    ]


def test_mux_objects_share_resources_despite_different_emission_timing():
    soccfg = make_mux_mock_soccfg(1, 1, gen_ch=0, ro_chs=(0,))
    cfg = MuxReadoutCfg(
        gen_ch=0,
        nqz=1,
        mixer_freq=1000,
        length=0.1,
        ro_length=0.2,
        tones=[MuxReadoutTone(ro_ch=0, freq=1000, gain=0.1)],
    )
    first = cfg.build("first")
    second = cfg.with_updates(pre_delay=0.01, post_delay=0.03, trig_offset=0.02).build(
        "second"
    )
    builder = v2.TimelineBuilder(soccfg, initial_delay=0, relax_delay=0)
    builder.emit(first)
    builder.emit(second)
    prog = builder.build()
    assert list(prog.ro_chs) == [0]
    assert prog.ro_chs[0]["trigs"] == 2
    assert len(waves(shots(prog)[0])) == 2


def test_distinct_readout_objects_share_identical_adc_declaration(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    builder.emit(_readout("first"))
    builder.emit(_readout("second"))
    prog = builder.build()
    assert list(prog.ro_chs) == [0]
    assert prog.ro_chs[0]["length"] == 61
    assert prog.ro_chs[0]["trigs"] == 2
    # Each read emits one ADC configuration and one generator waveform.
    assert len(waves(shots(prog)[0])) == 4
    prog.compile()
    assert prog.ro_chs[0]["trigs"] == 2


@pytest.mark.parametrize("resource", ["register", "register address"])
@pytest.mark.parametrize("frontend", ["builder", "native"])
def test_conflicting_module_objects_name_both_owners(mock_soccfg, resource, frontend):
    if resource == "register address":
        first = DeclarationModule("first", lambda p: p.add_reg("a", addr=10))
        second = DeclarationModule("second", lambda p: p.add_reg("b", addr=10))
    else:
        first = v2.LoadValue("first", [], idx_reg="index", val_reg="shared")
        second = v2.LoadValue("second", [], idx_reg="index", val_reg="shared")

    def compile_program():
        if frontend == "builder":
            builder = v2.TimelineBuilder(mock_soccfg)
            builder.emit(first)
            builder.emit(second)
            return builder.build()

        class Native(v2.TimelineProgramV2):
            def _initialize(self, cfg):
                self.declare(first)
                self.declare(second)

        return Native(mock_soccfg)

    with pytest.raises(ValueError, match="register") as error:
        compile_program()
    for module in (first, second):
        assert module.name in str(error.value)
        assert repr(module) in str(error.value)


def _pulse(name, ch=1, *, pulse_id: str | None = None):
    return v2.Pulse(
        name,
        v2.PulseCfg(
            ch=ch,
            nqz=1,
            freq=100,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.1),
        ),
        pulse_id=pulse_id,
    )


def test_parallel_swept_pulses_compile_without_materializing_dominated_ends(
    mock_soccfg,
):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    frequency = builder.add_sweep(
        "frequency", v2.SweepCfg(start=100, stop=102, step=1, expts=3)
    )
    duration = builder.add_sweep(
        "duration", v2.SweepCfg(start=71 / 384, stop=639 / 384, step=71 / 384, expts=9)
    )
    for name, length in (("prepare", 77 / 384), ("drive", duration)):
        builder.emit(
            v2.Join(
                *(
                    v2.Pulse(
                        f"{name}_{ch}",
                        v2.PulseCfg(
                            ch=ch,
                            nqz=1,
                            freq=100,
                            gain=0.5,
                            waveform=ConstWaveformCfg(length=0.1),
                            post_delay=4.3 / 430.08,
                        ),
                        length=length,
                        freq=frequency,
                    )
                    for ch in (0, 1)
                )
            )
        )
    builder.build()


@pytest.mark.parametrize("extra_post_cycles", [0, 1])
def test_parallel_swept_groups_fit_native_registers_and_keep_pointwise_timing(
    mock_soccfg,
    extra_post_cycles,
):
    builder = v2.TimelineBuilder(mock_soccfg, reps=2, initial_delay=0, relax_delay=0)
    frequency = builder.add_sweep(
        "frequency", v2.SweepCfg(start=100, stop=102, step=1, expts=3)
    )
    duration = builder.add_sweep(
        "duration", v2.SweepCfg(start=71 / 384, stop=639 / 384, step=71 / 384, expts=9)
    )
    for name, length in (
        ("prepare", 77 / 384),
        ("drive", duration),
        ("contrast", 77 / 384),
    ):
        builder.emit(
            v2.Join(
                *(
                    v2.Pulse(
                        f"{name}_{ch}",
                        v2.PulseCfg(
                            ch=ch,
                            nqz=1,
                            freq=100,
                            gain=0.5,
                            waveform=ConstWaveformCfg(length=0.1),
                            post_delay=(
                                4.3
                                + (
                                    extra_post_cycles
                                    if name == "drive" and ch == 1
                                    else 0
                                )
                            )
                            / 430.08,
                        ),
                        length=length,
                        freq=frequency,
                    )
                    for ch in (0, 1)
                )
            )
        )
    prog = builder.build()
    # Each aggregate rounds the physical tail plus post once. These nine
    # drive ends retain their fractional generator/tProc clock relationship.
    drive_ends = [
        end + extra_post_cycles for end in [84, 164, 243, 323, 402, 482, 561, 641, 720]
    ]
    expected_waves = [[0, 0, 91, 91, 91 + end, 91 + end] for end in drive_ends]
    expected_ends = [182 + end for end in drive_ends]
    for _ in range(2):
        scheduled = shots(prog)
        assert [waves(shot) for shot in scheduled] == expected_waves * 6
        assert [shot.body_end_cycle - shot.start_cycle for shot in scheduled] == (
            expected_ends * 6
        )
        # Invalidate QICK's binary cache to exercise a fresh compile, not a no-op.
        recompile(prog)


def test_builder_join_and_native_outputs_match_native(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    readout = _readout()
    pulse = v2.Pulse(
        "drive",
        v2.PulseCfg(
            ch=1,
            nqz=1,
            freq=100,
            gain=0.5,
            waveform=ConstWaveformCfg(length=0.1),
        ),
        pulse_id="native_drive",
    )
    joined = v2.Join(readout, pulse)
    builder.emit(joined, tag="parallel")
    builder.delay_by_timeline()
    builder.pulse(1, "native_drive", t=0.1, timeline="default", tag="pulse")
    builder.trigger(
        ros=[0],
        tts=[],
        pins=[],
        t=0.02,
        width=0.05,
        ddr4=False,
        mr=False,
        tag="trigger",
        timeline="default",
    )
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(joined)

        def _body(self, cfg):
            self.emit(joined, tag="parallel")
            self.delay_by_timeline()
            self.pulse(1, "native_drive", t=0.1, timeline="default", tag="pulse")
            self.trigger(
                ros=[0],
                tts=[],
                pins=[],
                t=0.02,
                width=0.05,
                ddr4=False,
                mr=False,
                tag="trigger",
                timeline="default",
            )

    expected = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    assert actual.prog_list == expected.prog_list
    for tag in ("pulse", "trigger"):
        assert actual.get_time_param(tag, "t") == expected.get_time_param(tag, "t")


def test_builder_preserves_borrowed_selector_and_shared_pulse_templates(mock_soccfg):
    cfg = v2.PulseCfg(
        ch=1,
        nqz=1,
        freq=100,
        gain=0.5,
        waveform=ConstWaveformCfg(length=0.1),
    )

    def modules():
        loader = v2.LoadValue("selector", [0, 1], idx_reg="scan", val_reg="selected")
        computed = v2.ComputedPulse(
            "chosen",
            val_reg="selected",
            pulses=[cfg, cfg.model_copy(update={"gain": 0.25})],
        )
        # Distinct objects intentionally share generator and template definitions.
        return [loader, computed, v2.Pulse("first", cfg), v2.Pulse("second", cfg)]

    shared = modules()

    def build(items, *, padded=False):
        builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
        builder.add_loop("scan", 2)
        if padded:
            builder.emit(v2.LoadValue("padding", [91, 83, 79], "scan", "padding"))
            builder.emit(v2.Pulse("padding", cfg.model_copy(update={"gain": 0.1})))
        for module in items:
            builder.emit(module)
        return builder.build()

    first = build(shared)
    actual = build(shared, padded=True)
    fresh = build(modules(), padded=True)
    assert actual.prog_list == fresh.prog_list
    np.testing.assert_array_equal(actual.compile_datamem(), fresh.compile_datamem())
    padded_timing = shots(actual)
    recompile(first)
    assert shots(actual) == padded_timing
    recompile(actual)
    assert shots(actual) == padded_timing
    assert first.prog_list == build(modules()).prog_list

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.modules = modules()
            for module in self.modules:
                self.declare(module)

        def _body(self, cfg):
            for module in self.modules:
                self.emit(module)

    expected = Native(mock_soccfg, initial_delay=0, relax_delay=0)
    assert first.prog_list == expected.prog_list
    np.testing.assert_array_equal(first.compile_datamem(), expected.compile_datamem())


@pytest.mark.parametrize("method", ["delay_auto", "trigger"])
def test_builder_rejects_positional_boolean_controls(mock_soccfg, method):
    builder = v2.TimelineBuilder(mock_soccfg)
    # Dynamic caller intentionally exercises invalid call shapes.
    operation: Callable[..., None]
    if method == "delay_auto":
        operation = builder.delay_auto
        args = (0, False)
    else:
        operation = builder.trigger
        args = (None, None, None, 0, None, True)
    with pytest.raises(TypeError, match="positional"):
        operation(*args)


class DeclarationModule(v2.Module):
    """Synthetic owner declaring resources without emitting output."""

    def __init__(self, name, declare: Callable[[v2.TimelineProgramV2], object]):
        self.name = name
        self.declaration = declare

    def init(self, prog):
        raise NotImplementedError("native-only fixture")

    def run(self, prog, t=0.0):
        raise NotImplementedError("native-only fixture")

    def declare(self, prog):
        self.declaration(prog)

    def lower(self, prog, *, timeline, tag):
        pass


def test_explicit_scratch_register_reuse_remains_valid(mock_soccfg):
    owner = DeclarationModule("owner", lambda p: p.add_reg("scratch"))
    borrower = DeclarationModule(
        "borrower", lambda p: p.add_reg("scratch", allow_reuse=True)
    )
    builder = v2.TimelineBuilder(mock_soccfg)
    builder.emit(owner)
    builder.emit(borrower)
    actual = builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(owner)
            self.declare(borrower)

    assert actual.prog_list == Native(mock_soccfg).prog_list


def test_failed_build_also_freezes_builder(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg)
    builder.emit(DeclarationModule("first", lambda p: p.add_reg("shared")))
    builder.emit(DeclarationModule("second", lambda p: p.add_reg("shared")))
    with pytest.raises(ValueError, match="register"):
        builder.build()
    with pytest.raises(ValueError, match="frozen"):
        builder.emit(_readout("later"))


def _compile_scoped(
    soccfg: QickConfig,
    frontend: str,
    body: Callable[[v2.TimelineBuilder | v2.TimelineProgramV2], None],
    modules: Sequence[v2.Module] = (),
    *,
    loops: Sequence[tuple[str, int]] = (),
) -> v2.TimelineProgramV2:
    if frontend == "builder":
        builder = v2.TimelineBuilder(soccfg, initial_delay=0, relax_delay=0)
        for name, count in loops:
            builder.add_loop(name, count)
        body(builder)
        return builder.build()

    class Native(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            for name, count in loops:
                self.add_loop(name, count)
            for module in modules:
                self.declare(module)

        def _body(self, cfg):
            body(self)

    return Native(soccfg, initial_delay=0, relax_delay=0)


@pytest.mark.parametrize("frontend", ["builder", "native"])
@pytest.mark.parametrize("align_timeline", [True, False])
@pytest.mark.parametrize("explicit_parent", [True, False])
def test_branch_case_timeline_forks_parent_and_merges_selected_or_aligned_end(
    mock_soccfg, frontend, align_timeline, explicit_parent
):
    pulse = _pulse("drive")
    names = []

    def body(program: v2.TimelineBuilder | v2.TimelineProgramV2):
        program.fork("default", into="parent")
        program.timeline_inc(10 / 430.08, timeline="parent")
        with (
            program.timeline("parent" if not explicit_parent else "default"),
            program.branch(
                by="scan",
                align_timeline=align_timeline,
                timeline="parent" if explicit_parent else None,
            ) as cases,
            cases.case(1) as name,
        ):
            names.append(name)
            program.timeline_inc(20 / 430.08)
            program.emit(pulse)
        program.emit(pulse, timeline="parent")

    prog = _compile_scoped(
        mock_soccfg, frontend, body, modules=(pulse,), loops=(("scan", 3),)
    )
    assert names == ["scan[1]"]
    assert [waves(shot) for shot in shots(prog)] == (
        [[73], [30, 73], [73]] if align_timeline else [[10], [30, 73], [10]]
    )


@pytest.mark.parametrize("frontend", ["builder", "native"])
@pytest.mark.parametrize("operation", ["emit", "fork", "join", "scope", "parent"])
def test_branch_case_rejects_access_to_outer_timelines_at_build(
    mock_soccfg, frontend, operation
):
    pause = v2.Delay("pause", 0.1)

    def body(program: v2.TimelineBuilder | v2.TimelineProgramV2):
        program.fork("default", into="outside")
        with (
            program.branch(by="scan", align_timeline=True) as cases,
            cases.case(0) as case,
        ):
            if operation == "emit":
                program.emit(pause, timeline="outside")
            elif operation == "fork":
                program.fork("outside", into="local")
            elif operation == "join":
                program.join("outside", into=case)
            elif operation == "scope":
                with program.timeline("outside"):
                    pass
            else:
                program.timeline_inc(0, timeline="default")

    with pytest.raises(ValueError, match="(outside|default).*unknown timeline"):
        _compile_scoped(
            mock_soccfg, frontend, body, modules=(pause,), loops=(("scan", 2),)
        )


@pytest.mark.parametrize("frontend", ["builder", "native"])
def test_branch_case_local_forks_and_nested_parent_merge_without_leaking_names(
    mock_soccfg, frontend
):
    pulse = _pulse("drive")

    def body(program: v2.TimelineBuilder | v2.TimelineProgramV2):
        with (
            program.branch(by="scan", align_timeline=True) as cases,
            cases.case(1) as case,
        ):
            program.fork(case, into="local")
            program.timeline_inc(10 / 430.08, timeline="local")
            with (
                program.branch(
                    by="inner", align_timeline=True, timeline="local"
                ) as nested,
                nested.case(0),
            ):
                program.timeline_inc(20 / 430.08)
            program.join("local", into=case)
        # Both implicit and explicit local names expire, allowing fresh forks.
        program.fork("default", into=["scan[1]", "inner[0]", "local"])
        program.emit(pulse)

    prog = _compile_scoped(
        mock_soccfg,
        frontend,
        body,
        modules=(pulse,),
        loops=(("scan", 2), ("inner", 2)),
    )
    assert [waves(shot) for shot in shots(prog)] == [[30]] * 4


@pytest.mark.parametrize("frontend", ["builder", "native"])
def test_timeline_scope_nested_override_and_restore_match_explicit_program(
    mock_soccfg, frontend
):
    pause = v2.Delay("pause", 0.2)

    def scoped(program: v2.TimelineBuilder | v2.TimelineProgramV2) -> None:
        program.fork("default", into=["outer", "inner"])
        with program.timeline("outer"):
            program.timeline_inc(0.1)
            program.emit(pause)
            program.timeline_inc(0.3, timeline="default")
            with program.timeline("inner"):
                program.emit(pause, timeline=None)
                program.emit(pause, timeline="outer")
                program.delay_by_timeline()
            program.timeline_inc(0.4)
            program.delay_by_timeline()
        program.timeline_inc(0.5)
        program.delay_by_timeline()

    class Explicit(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(pause)

        def _body(self, cfg):
            self.fork("default", into=["outer", "inner"])
            self.timeline_inc(0.1, timeline="outer")
            self.emit(pause, timeline="outer")
            self.timeline_inc(0.3, timeline="default")
            self.emit(pause, timeline="inner")
            self.emit(pause, timeline="outer")
            self.delay_by_timeline(timeline="inner")
            self.timeline_inc(0.4, timeline="outer")
            self.delay_by_timeline(timeline="outer")
            self.timeline_inc(0.5, timeline="default")
            self.delay_by_timeline(timeline="default")

    expected = Explicit(mock_soccfg, initial_delay=0, relax_delay=0)
    actual = _compile_scoped(mock_soccfg, frontend, scoped, (pause,))
    assert actual.prog_list == expected.prog_list
    actual.compile()
    assert actual.prog_list == expected.prog_list


@pytest.mark.parametrize("frontend", ["builder", "native"])
def test_timeline_scope_join_native_outputs_and_reference_delays_match_explicit(
    mock_soccfg, frontend
):
    readout, drive = _readout(), _pulse("drive", pulse_id="scoped_drive")
    joined = v2.Join(readout, drive)

    def scoped(program: v2.TimelineBuilder | v2.TimelineProgramV2) -> None:
        program.fork("default", into="work")
        with program.timeline("work"):
            program.timeline_inc(0.1)
            program.emit(joined)
            program.pulse(1, "scoped_drive", t=0.1, tag="scoped_pulse")
            program.trigger(ros=[0], t=0.05, tag="scoped_trigger")
            program.delay(0.02, tag="native_delay")
            program.delay_auto(0.03, tag="native_auto")
            program.pulse(1, "scoped_drive", timeline="default", tag="override")
        program.delay_auto()
        program.pulse(1, "scoped_drive", tag="native_after")

    class Explicit(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(joined)

        def _body(self, cfg):
            self.fork("default", into="work")
            self.timeline_inc(0.1, timeline="work")
            self.emit(joined, timeline="work")
            self.pulse(1, "scoped_drive", t=0.1, timeline="work", tag="scoped_pulse")
            self.trigger(ros=[0], t=0.05, timeline="work", tag="scoped_trigger")
            self.delay(0.02, tag="native_delay")
            self.delay_auto(0.03, tag="native_auto")
            self.pulse(1, "scoped_drive", timeline="default", tag="override")
            self.delay_auto()
            self.pulse(1, "scoped_drive", tag="native_after")

    expected = Explicit(mock_soccfg, initial_delay=0, relax_delay=0)
    actual = _compile_scoped(mock_soccfg, frontend, scoped, (joined,))
    assert actual.prog_list == expected.prog_list
    actual.compile()
    assert actual.prog_list == expected.prog_list


@pytest.mark.parametrize("frontend", ["builder", "native"])
def test_timeline_scope_repeat_parent_and_case_emissions_match_explicit_program(
    mock_soccfg, frontend
):
    readout = _readout()

    def scoped(program: v2.TimelineBuilder | v2.TimelineProgramV2) -> None:
        program.fork("default", into="work")
        with program.timeline("work"):
            program.timeline_inc(0.1)
            program.delay_by_timeline()
            with program.repeat("body", 2):
                with program.branch(by="scan", align_timeline=True) as branch:
                    with branch.case(0):
                        program.emit(readout)
                        program.timeline_inc(0.02)
                    with branch.case(1):
                        program.timeline_inc(0.3)
                        program.emit(readout, timeline=None)
                program.delay_by_timeline()
            program.delay_by_timeline(timeline="default")
            with program.repeat("override", 2, timeline="default"):
                program.timeline_inc(0.05, timeline="default")
                program.delay_by_timeline(timeline="default")
        program.emit(readout)

    class Explicit(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(readout)

        def _body(self, cfg):
            self.fork("default", into="work")
            self.timeline_inc(0.1, timeline="work")
            self.delay_by_timeline(timeline="work")
            with self.repeat("body", 2, timeline="work"):
                with self.branch(
                    by="scan", align_timeline=True, timeline="work"
                ) as branch:
                    with branch.case(0) as case:
                        self.emit(readout, timeline=case)
                        self.timeline_inc(0.02, timeline=case)
                    with branch.case(1) as case:
                        self.timeline_inc(0.3, timeline=case)
                        self.emit(readout, timeline=case)
                self.delay_by_timeline(timeline="work")
            self.delay_by_timeline(timeline="default")
            with self.repeat("override", 2, timeline="default"):
                self.timeline_inc(0.05, timeline="default")
                self.delay_by_timeline(timeline="default")
            self.emit(readout, timeline="default")

    expected = Explicit(mock_soccfg, initial_delay=0, relax_delay=0)
    actual = _compile_scoped(
        mock_soccfg, frontend, scoped, (readout,), loops=(("scan", 2),)
    )
    assert actual.prog_list == expected.prog_list
    actual.compile()
    assert actual.prog_list == expected.prog_list


@pytest.mark.parametrize("frontend", ["builder", "native"])
@pytest.mark.parametrize("scenario", ["missing", "future", "expired", "empty-name"])
def test_timeline_scope_requires_a_currently_declared_name_even_when_empty(
    mock_soccfg, frontend, scenario
):
    def scoped(program: v2.TimelineBuilder | v2.TimelineProgramV2) -> None:
        name = "" if scenario == "empty-name" else "work"
        if scenario == "expired":
            program.delay_by_timeline()
            with program.repeat("body", 1):
                program.fork("default", into=name)
                program.delay_by_timeline()
        with program.timeline(name):
            pass
        if scenario == "future":
            program.fork("default", into=name)

    with pytest.raises(ValueError, match="unknown timeline; declare it with fork"):
        _compile_scoped(mock_soccfg, frontend, scoped)


@pytest.mark.parametrize("frontend", ["builder", "native"])
def test_empty_timeline_scopes_do_not_fork_join_or_close_branch_ends(
    mock_soccfg, frontend
):
    pause = v2.Delay("pause", 0.1)

    def scoped(program: v2.TimelineBuilder | v2.TimelineProgramV2) -> None:
        with program.timeline("default"):
            pass
        program.fork("default", into="work")
        program.timeline_inc(0.5, timeline="work")
        with (
            program.timeline("work"),
            program.branch(by="scan", align_timeline=True) as branch,
        ):
            with branch.case(0) as case, program.timeline(case):
                pass
            with branch.case(1):
                program.emit(pause)
        program.delay_by_timeline(timeline="default")
        program.join("work", into="default")

    class Explicit(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(pause)

        def _body(self, cfg):
            self.fork("default", into="work")
            self.timeline_inc(0.5, timeline="work")
            with self.branch(by="scan", align_timeline=True, timeline="work") as branch:
                with branch.case(0):
                    pass
                with branch.case(1):
                    self.emit(pause)
            self.delay_by_timeline(timeline="default")
            self.join("work", into="default")

    expected = Explicit(mock_soccfg, initial_delay=0, relax_delay=0)
    actual = _compile_scoped(
        mock_soccfg, frontend, scoped, (pause,), loops=(("scan", 2),)
    )
    assert actual.prog_list == expected.prog_list
    actual.compile()
    assert actual.prog_list == expected.prog_list


def test_builder_timeline_between_branch_and_cases_matches_native(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg, initial_delay=0, relax_delay=0)
    builder.add_loop("scan", 2)
    pause = v2.Delay("pause", 0.1)
    builder.fork("default", into="work")
    builder.timeline_inc(0.3, timeline="work")
    with builder.branch(by="scan", align_timeline=True) as branch:
        with builder.timeline("work"), branch.case(0) as case:
            builder.emit(pause)
            builder.emit(pause, timeline=case)
            with builder.timeline(case):
                builder.emit(pause)
            builder.emit(pause)
        with branch.case(1):
            builder.emit(pause)
    builder.emit(pause)
    actual = builder.build()

    class Explicit(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.pause = v2.Delay("pause", 0.1)
            self.declare(self.pause)

        def _body(self, cfg):
            self.fork("default", into="work")
            self.timeline_inc(0.3, timeline="work")
            with self.branch(by="scan", align_timeline=True) as branch:
                with branch.case(0) as case:
                    self.emit(self.pause, timeline=case)
                    self.emit(self.pause, timeline=case)
                    self.emit(self.pause, timeline=case)
                    self.emit(self.pause, timeline=case)
                with branch.case(1) as case:
                    self.emit(self.pause, timeline=case)
            self.emit(self.pause, timeline="default")

    expected = Explicit(mock_soccfg, initial_delay=0, relax_delay=0)
    assert actual.prog_list == expected.prog_list
    actual.compile()
    assert actual.prog_list == expected.prog_list


def test_builder_rejects_unknown_trigger_keywords_at_recording(mock_soccfg):
    builder = v2.TimelineBuilder(mock_soccfg)
    operation: Callable[..., None] = builder.trigger
    with pytest.raises(TypeError, match="unexpected.*typo"):
        operation(typo=True)
