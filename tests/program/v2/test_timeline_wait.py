"""Shell WAIT bounds through public timeline compilation and point scheduling."""

import math
from typing import Literal

import pytest
from qick.asm_v2 import QickSweep1D
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.mocksoc import make_mock_soccfg, make_mux_mock_soccfg
from zcu_tools.program.v2.modules import ComputedPulse, DelayAuto, LoadValue, PulseCfg
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import CycleBounds, TimelineProgramV2

from .timing_support import shots, waves


def t1_program(
    mux: bool,
    *,
    delay_kind: Literal["uniform", "nonuniform", "static"] = "uniform",
    sequence: Literal["t1", "ramsey", "echo"] = "t1",
) -> TimelineProgramV2:
    """Build synthetic T1 with a 3441-cycle origin span and ten-cycle read post."""
    if mux:
        soccfg = make_mux_mock_soccfg(2, 2, gen_ch=0, ro_chs=(0, 1))
        readout = MuxReadoutCfg(
            gen_ch=0,
            nqz=1,
            mixer_freq=1000,
            length=0.1,
            ro_length=55 / 76.8,
            trig_offset=12 / 430.08,
            post_delay=10 / 430.08,
            tones=[
                MuxReadoutTone(ro_ch=ch, freq=1000 + 80 * ch, gain=0.1) for ch in (0, 1)
            ],
        ).build("read")
    else:
        soccfg = make_mock_soccfg(2, 1)
        readout = PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            ro_ch=0,
            freq=100,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=127 / 307.2,
            trig_offset=12 / 430.08,
            post_delay=10 / 430.08,
        ).build("read")

    loader = LoadValue("delays", [43, 300, 3484], "delay", "delay_cycles")
    native_gap = DelayAuto("gap", "delay_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("delay", 3)
            if delay_kind == "nonuniform":
                self.declare(loader)
                self.declare(native_gap)
            self.declare_gen(1, nqz=1)
            self.add_pulse(
                1, "pi", style="const", freq=1000, phase=0, gain=0.5, length=25 / 384
            )
            self.declare(readout)

        def _body(self, cfg):
            self.pulse(1, "pi", timeline="default")
            if delay_kind == "nonuniform":
                self.emit(loader)
                self.emit(native_gap)
            else:
                gap = (
                    QickSweep1D("delay", 43 / 430.08, 3484 / 430.08)
                    if delay_kind == "uniform"
                    else 43 / 430.08
                )
                if sequence == "echo":
                    self.delay_timeline(gap / 2)
                    self.pulse(1, "pi", timeline="default")
                    self.delay_timeline(gap / 2)
                else:
                    self.delay_timeline(gap)
                if sequence != "t1":
                    self.pulse(1, "pi", timeline="default")
            self.emit(readout)

    return Program(soccfg, reps=1, initial_delay=0, relax_delay=1)


def shell_wait(prog: TimelineProgramV2) -> int:
    """Read the reference-relative shell WAIT literal from public assembly."""
    (wait,) = [inst for inst in prog.prog_list if inst["CMD"] == "WAIT"]
    return int(str(wait["TIME"]).removeprefix("@"))


@pytest.mark.parametrize("mux", [False, True], ids=["single", "mux"])
def test_uniform_t1_shell_wait_cancels_shared_readout_origin(mux):
    prog = t1_program(mux)
    assert shell_wait(prog) == 0


def test_two_sweep_reference_debits_keep_each_axis_correlated():
    loader = LoadValue("origin", [40, 60, 50], "origin_axis", "origin_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("origin_axis", 3)
            self.add_loop("advance_axis", 3)
            self.declare(loader)
            self.declare_readout(0, length=50 / 307.2)

        def _body(self, cfg):
            self.emit(loader)
            self.delay_timeline("origin_cycles")
            self.trigger(ros=[0], width=1 / 430.08, timeline="default")
            self.delay_timeline(10 / 430.08)
            self.delay(QickSweep1D("advance_axis", 10 / 430.08, 30 / 430.08))

    prog = Program(make_mock_soccfg(0, 1), initial_delay=0, relax_delay=1)
    # Each ADC ends 70 cycles after origin; body closes ten cycles later.
    expected_body = [origin + 80 for origin in (40, 60, 50) for _ in range(3)]
    result = shots(prog)
    assert [s.body_end_cycle - s.start_cycle for s in result] == expected_body
    assert shell_wait(prog) == 0
    prog.compile()
    assert shots(prog) == result
    assert shell_wait(prog) == 0


@pytest.mark.parametrize("mux", [False, True], ids=["single", "mux"])
@pytest.mark.parametrize("delay_kind", ["nonuniform", "static"])
def test_nonuniform_and_static_t1_keep_zero_shell_wait(mux, delay_kind):
    prog = t1_program(mux, delay_kind=delay_kind)
    assert shell_wait(prog) == 0


@pytest.mark.parametrize("mux", [False, True], ids=["single", "mux"])
@pytest.mark.parametrize("sequence", ["ramsey", "echo"])
def test_coherence_shell_wait_matches_pointwise_scheduling_oracle(mux, sequence):
    prog = t1_program(mux, sequence=sequence)
    expected_waves = []
    expected_body = []
    remaining = []
    for gap in (43, 1763.5, 3484):
        starts = [0]
        end = 28
        if sequence == "echo":
            end = math.ceil(end + gap / 2)
            starts.append(end)
            end += 28
            end = math.ceil(end + gap / 2)
        else:
            end = math.ceil(end + gap)
        starts.append(end)
        end += 28
        starts.append(end)  # Readout starts at the last drive end.
        if not mux:
            starts.append(
                end
            )  # Dynamic readout configuration is also a wave port event.
        expected_waves.append(starts)
        adc_end = end + (320 if mux else 190)
        body_end = end + (330 if mux else 200)
        expected_body.append(body_end)
        remaining.append(max(0, adc_end - body_end))
    result = shots(prog)
    assert [waves(s) for s in result] == expected_waves
    assert [s.body_end_cycle - s.start_cycle for s in result] == expected_body
    # At every point body closure is ten cycles later than the ceiled ADC end.
    assert shell_wait(prog) == max(remaining)


@pytest.mark.filterwarnings("error")
def test_overlapping_pulses_swept_start_keeps_exact_wait_after_all_timeline_sync():
    readout = PulseReadoutCfg(
        gen_ch=2,
        nqz=1,
        ro_ch=0,
        freq=100,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=127 / 307.2,
        trig_offset=12 / 430.08,
        post_delay=10 / 430.08,
    ).build("read")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.start = self.add_sweep(
                "start",
                SweepCfg(
                    start=28 / 430.08,
                    stop=84 / 430.08,
                    step=28 / 430.08,
                    expts=3,
                ),
            )
            for ch, length in ((0, 100 / 384), (1, 50 / 384)):
                self.declare_gen(ch, nqz=1)
                self.add_pulse(
                    ch,
                    f"drive{ch}",
                    style="const",
                    freq=1000,
                    phase=0,
                    gain=0.5,
                    length=length,
                )
            self.declare(readout)

        def _body(self, cfg):
            self.fork("default", into="other")
            self.pulse(0, "drive0", timeline="default")
            self.pulse(1, "drive1", t=self.start, timeline="other")
            self.join("other", into="default")
            self.sync()
            self.emit(readout)

    prog = Program(make_mock_soccfg(3, 1), initial_delay=0, relax_delay=1)
    result = shots(prog)
    # A lasts 112 tProc cycles; B lasts 56. All three points overlap A.
    # max(112, t + 56) is 112, 112, 140: left side, corner, right side.
    # Readout emits a pulse and config at that maximum, then triggers at +12.
    assert [waves(s) for s in result] == [
        [0, 28, 112, 112],
        [0, 56, 112, 112],
        [0, 84, 140, 140],
    ]
    trigger_starts = [
        e.cycle - s.start_cycle
        for s in result
        for e in s.events
        if e.kind == "data" and e.value
    ]
    assert trigger_starts == [124, 124, 152]
    body_ends = [s.body_end_cycle - s.start_cycle for s in result]
    # The ADC lasts 177.8 cycles. Its ceiled end is ten before body closure.
    assert body_ends == [312, 312, 340]
    remaining = [
        max(0, math.ceil(trigger + 177.8) - body)
        for trigger, body in zip(trigger_starts, body_ends, strict=True)
    ]
    assert remaining == [0, 0, 0]
    assert shell_wait(prog) == max(remaining)


def variable_tail_program(
    *, relax_cycles: int = 100, tail_cycles: tuple[int, int] = (10, 30)
) -> TimelineProgramV2:
    """Leave the chosen ADC tail after a shared runtime origin cancels."""
    origin = LoadValue("origin", [40, 60, 50], "origin_axis", "origin_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("origin_axis", 3)
            self.add_loop("post_axis", 3)
            self.declare(origin)
            self.declare_readout(0, length=50 / 307.2)

        def _body(self, cfg):
            self.emit(origin)
            self.delay_timeline("origin_cycles")
            self.trigger(ros=[0], width=1 / 430.08, timeline="default")
            self.delay_timeline(
                QickSweep1D(
                    "post_axis", -tail_cycles[0] / 430.08, -tail_cycles[1] / 430.08
                )
            )

    return Program(
        make_mock_soccfg(0, 1), initial_delay=0, relax_delay=relax_cycles / 430.08
    )


def test_variable_shell_wait_warns_with_only_uncancelled_axis_and_cycle_slack():
    with pytest.warns(UserWarning, match=r"WAIT.*post_axis.*slack=20 cycles") as caught:
        prog = variable_tail_program()
    assert all("origin" not in str(item.message) for item in caught)
    assert shell_wait(prog) == 30
    # ADC end is origin + 70, while the negative post closes the body earlier.
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        100,
        90,
        80,
        120,
        110,
        100,
        110,
        100,
        90,
    ]


def test_shell_wait_slack_above_relax_rejects_at_compile_time():
    with pytest.raises(
        ValueError, match=r"WAIT.*post_axis.*slack=20 cycles.*relax=19 cycles"
    ):
        variable_tail_program(relax_cycles=19)


def test_shell_wait_slack_equal_to_relax_warns_without_rejection():
    with pytest.warns(UserWarning, match=r"WAIT.*post_axis.*slack=20 cycles"):
        prog = variable_tail_program(relax_cycles=20)
    assert shell_wait(prog) == 30


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("tail", [0, 30])
def test_constant_shell_wait_has_zero_slack_even_with_no_relax(tail):
    prog = variable_tail_program(relax_cycles=0, tail_cycles=(tail, tail))
    assert shell_wait(prog) == tail


def opposing_tail_program(
    values: tuple[int, ...], *, relax_cycles: int = 100, feedback: bool = False
):
    """Two ADCs share an index but use opposite table origins; body ends at zero."""
    left = LoadValue("left", values, "axis", "left_cycles")
    right = LoadValue("right", [100 - x for x in values], "axis", "right_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("axis", len(values))
            self.declare(left)
            self.declare(right)
            for ch in (0, 1):
                self.declare_readout(ch, length=50 / 307.2)
            if feedback:
                self.add_reg("feedback_cycles")

        def _body(self, cfg):
            self.fork("default", into="other")
            for loader, register, timeline, ro in (
                (left, "left_cycles", "default", 0),
                (right, "right_cycles", "other", 1),
            ):
                self.emit(loader)
                self.delay_timeline(register, timeline=timeline)
                self.trigger(ros=[ro], width=1 / 430.08, timeline=timeline)
                self.delay_timeline(-200 / 430.08, timeline=timeline)
            if feedback:
                with self.time_producer("feedback_cycles", CycleBounds(0, 31)):
                    self.write_reg_op("feedback_cycles", "s11", "AND", 31)
                self.delay_timeline("feedback_cycles")

    return Program(
        make_mock_soccfg(0, 2), initial_delay=0, relax_delay=relax_cycles / 430.08
    )


def test_shared_axis_lookup_maximum_warns_with_exact_discrete_slack():
    with pytest.warns(
        UserWarning, match=r"WAIT.*left_cycles.*right_cycles.*slack=50 cycles"
    ):
        prog = opposing_tail_program((0, 50, 100))
    assert shell_wait(prog) == 170
    result = shots(prog)
    assert [s.body_end_cycle - s.start_cycle for s in result] == [0, 0, 0]


@pytest.mark.filterwarnings("error")
def test_shared_axis_lookup_maximum_is_constant_on_two_legal_points():
    prog = opposing_tail_program((0, 100), relax_cycles=0)
    assert shell_wait(prog) == 170


def test_unknown_wait_warns_with_slack_bound_without_rejecting_above_relax():
    with pytest.warns(
        UserWarning,
        match=r"WAIT.*feedback_cycles.*slack=31 cycles.*upper bound.*not actual",
    ):
        prog = opposing_tail_program((0, 100), relax_cycles=0, feedback=True)
    assert shell_wait(prog) == 170


def test_shell_bound_does_not_replace_native_swept_final_reference_advance():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("axis", 3)
            for ch in (0, 1):
                self.declare_readout(ch, length=50 / 307.2)

        def _body(self, cfg):
            self.trigger(
                ros=[0], t=QickSweep1D("axis", 0, 100 / 430.08), width=1 / 430.08
            )
            self.trigger(
                ros=[1], t=QickSweep1D("axis", 100 / 430.08, 0), width=1 / 430.08
            )

    with pytest.warns(UserWarning, match=r"WAIT.*axis.*slack=50 cycles"):
        prog = Program(
            make_mock_soccfg(0, 2), initial_delay=0, relax_delay=100 / 430.08
        )
    assert shell_wait(prog) == 170
    assert [s.end_cycle - s.start_cycle for s in shots(prog)] == [270, 220, 270]


def test_table_derived_repeat_count_keeps_exact_wait_slack():
    origin = LoadValue("origin", [40, 40, 40], "axis", "origin_cycles")
    count = LoadValue("counts", [1, 3, 2], "axis", "count_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("axis", 3)
            self.declare(origin)
            self.declare(count)
            self.declare_readout(0, length=50 / 307.2)

        def _body(self, cfg):
            self.emit(origin)
            self.emit(count)
            self.delay_timeline("origin_cycles")
            self.trigger(ros=[0], width=1 / 430.08, timeline="default")
            self.delay_timeline(-200 / 430.08)
            with self.repeat("repeat", "count_cycles", range_hint=(1, 3)):
                self.delay_timeline(10 / 430.08)

    with pytest.warns(
        UserWarning, match=r"WAIT.*count_cycles.*slack=20 cycles"
    ) as caught:
        prog = Program(make_mock_soccfg(0, 1), initial_delay=0, relax_delay=20 / 430.08)
    assert all("not actual" not in str(w.message) for w in caught)
    assert shell_wait(prog) == 100
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [10, 30, 20]


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("reuse_after", [False, True])
def test_fixed_repeat_table_advance_sums_legal_iterations_before_wait_projection(
    reuse_after,
):
    delays = LoadValue("delays", [10, 20], "inner", "delay_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(delays)
            self.declare_readout(0, length=50 / 307.2)

        def _body(self, cfg):
            self.trigger(ros=[0], width=1 / 430.08)
            with self.repeat("inner", 2):
                self.emit(delays)
                self.delay_timeline("delay_cycles")
            if reuse_after:
                self.delay_timeline("delay_cycles")

    prog = Program(make_mock_soccfg(0, 1), initial_delay=0, relax_delay=0)
    assert shell_wait(prog) == (20 if reuse_after else 40)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        50 if reuse_after else 30
    ]


@pytest.mark.filterwarnings("error")
def test_fixed_repeat_readout_keeps_only_the_last_iteration_adc_tail():
    delays = LoadValue("delays", [10, 20], "inner", "delay_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(delays)
            self.declare_readout(0, length=50 / 307.2)

        def _body(self, cfg):
            with self.repeat("inner", 2):
                self.emit(delays)
                self.delay_timeline(60 / 430.08)
                self.trigger(ros=[0], width=1 / 430.08, timeline="default")
                self.delay_timeline(-40 / 430.08)
                self.delay_timeline("delay_cycles")

    prog = Program(make_mock_soccfg(0, 1), initial_delay=0, relax_delay=0)
    assert shell_wait(prog) == 20
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [210]


@pytest.mark.parametrize("branch", [False, True])
def test_native_scalar_adc_retains_table_reference_debit(branch):
    delays = LoadValue("delays", [100, 110], "scan", "delay_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(delays)
            self.declare_readout(0, length=50 / 307.2)

        def _body(self, cfg):
            self.trigger(ros=[0], t=60 / 430.08, width=1 / 430.08)
            self.emit(delays)
            if branch:
                with self.branch(by="scan") as cases:
                    for case in range(2):
                        with cases.case():
                            self.delay_timeline("delay_cycles")
                            self.delay_timeline(case * 5 / 430.08)
            else:
                self.delay_timeline("delay_cycles")

    slack = 15 if branch else 10
    with pytest.warns(UserWarning, match=rf"slack={slack} cycles"):
        prog = Program(
            make_mock_soccfg(0, 1), initial_delay=0, relax_delay=slack / 430.08
        )
    assert shell_wait(prog) == 30
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        100,
        115 if branch else 110,
    ]


def test_selected_playback_keeps_known_selector_in_exact_relax_guard():
    selector = LoadValue("selector", [0, 1], "scan", "selected")
    playback = ComputedPulse(
        "playback",
        val_reg="selected",
        pulses=[
            PulseCfg(
                ch=0,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=length / 307.2),
            )
            for length in (50, 100)
        ],
    )

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(selector)
            self.declare(playback)
            self.declare_readout(0, length=150 / 307.2)

        def _body(self, cfg):
            self.trigger(ros=[0], width=1 / 430.08)
            self.emit(selector)
            self.emit(playback)

    # Native waveform rounding gives aggregate ends 71 and 140 cycles.
    with pytest.raises(ValueError, match=r"selected.*slack=69 cycles.*relax=68 cycles"):
        Program(make_mock_soccfg(1, 1), initial_delay=0, relax_delay=68 / 430.08)
    with pytest.warns(UserWarning, match=r"selected.*slack=69 cycles"):
        prog = Program(make_mock_soccfg(1, 1), initial_delay=0, relax_delay=69 / 430.08)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [71, 140]
    assert shell_wait(prog) == 139


def test_crossing_body_maximum_retains_positive_adc_remainder():
    origins = (40, 60, 50)
    others = (30, 140, 200)
    left = LoadValue("left", origins, "left_axis", "left_cycles")
    right = LoadValue("right", others, "right_axis", "right_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("left_axis", 3)
            self.add_loop("right_axis", 3)
            self.declare(left)
            self.declare(right)
            self.declare_readout(0, length=50 / 307.2)

        def _body(self, cfg):
            self.fork("default", into="other")
            self.emit(left)
            self.emit(right)
            self.delay_timeline("left_cycles")
            self.trigger(ros=[0], width=1 / 430.08, timeline="default")
            # A negative post leaves a real ADC tail beyond the body boundary.
            self.delay_timeline(-30 / 430.08)
            self.delay_timeline("right_cycles", timeline="other")

    prog = Program(make_mock_soccfg(0, 1), initial_delay=0, relax_delay=1)
    expected_body = []
    remaining = []
    for origin in origins:
        for other in others:
            body = max(origin + 40, other)
            expected_body.append(body)
            remaining.append(max(0, origin + 70 - body))
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == expected_body
    assert shell_wait(prog) == max(remaining) == 30
