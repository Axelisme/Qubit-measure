"""Shell WAIT bounds through public timeline compilation and point scheduling."""

import math
import random
import warnings
from dataclasses import dataclass
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


@pytest.mark.filterwarnings("error")
def test_fractional_native_adc_tail_never_rounds_below_physical_end():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare_readout(0, length=51 / 307.2)

        def _body(self, cfg):
            self.trigger(ros=[0], t=0, width=1 / 430.08)
            self.delay_timeline(42 / 430.08)

    prog = Program(
        make_mock_soccfg(0, 1), initial_delay=0, final_wait=0, relax_delay=0
    )
    (shot,) = shots(prog)
    assert shot.body_end_cycle - shot.start_cycle == 42
    # Independent ADC-clock oracle: 51 samples end at 71.4 tProc cycles.
    physical_tail = 51 / 307.2 * 430.08 - 42
    assert shell_wait(prog) >= physical_tail
    assert shell_wait(prog) == math.ceil(physical_tail) == 30


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


@pytest.mark.parametrize("relax_cycles", [10, 9])
def test_zero_count_repeat_preserves_entry_source_and_exact_wait_guard(relax_cycles):
    counts = LoadValue("counts", [0, 1], "scan", "count_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(counts)
            self.add_reg("duration")
            self.declare_readout(0, length=50 / 307.2)

        def _body(self, cfg):
            self.emit(counts)
            self.write_reg("duration", 10)
            self.trigger(ros=[0], width=1 / 430.08)
            with self.repeat("inner", "count_cycles", range_hint=(0, 1)):
                self.write_reg("duration", 20)
            self.delay_timeline("duration")

    if relax_cycles < 10:
        with pytest.raises(
            ValueError, match=r"WAIT.*count_cycles.*slack=10 cycles.*relax=9 cycles"
        ):
            Program(make_mock_soccfg(0, 1), initial_delay=0, relax_delay=9 / 430.08)
    else:
        with pytest.warns(
            UserWarning, match=r"WAIT.*count_cycles.*slack=10 cycles"
        ) as caught:
            prog = Program(
                make_mock_soccfg(0, 1), initial_delay=0, relax_delay=10 / 430.08
            )
        assert all("upper bound" not in str(item.message) for item in caught)
        body = [s.body_end_cycle - s.start_cycle for s in shots(prog)]
        assert body == [10, 20]
        # The ADC ends at 70: skipping the write leaves a 60-cycle tail.
        tails = [70 - end for end in body]
        assert tails == [60, 50]
        assert shell_wait(prog) == max(tails) == 60


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


@dataclass(frozen=True)
class WaitRecipe:
    """Reproducible public program inputs; all delays/tails are tProc cycles.

    sweep_steps has one step per three-point axis. counts and branch_delays
    share axis0's index. pulse_samples holds the two native DAC lengths at
    384 MHz; adc_samples is the ADC length at 307.2 MHz. read_at selects the
    trigger's position; tail subtracts from its timeline end, not its ADC end.
    fixed_count and counts give the fixed and table-selected Repeat counts.
    """

    sweep_steps: tuple[int, ...]
    pulse_samples: tuple[int, int]
    adc_samples: int
    fixed_count: int
    counts: tuple[int, ...]
    branch_delays: tuple[int, ...]
    loop_delay: int
    read_at: Literal["before", "fixed", "runtime", "branch", "after"]
    tail: int


def wait_recipes(seed: int) -> list[WaitRecipe]:
    """Generate 30 recipes within the mock tProc's 16-data-register limit.

    One-axis recipes include fractional-cycle DAC lengths and crossing forks.
    Two-axis recipes use integral-cycle lengths and a noncrossing fork to leave
    registers for both sweeps. The measured whole-shape budgets below apply
    to this seed, not additive costs of individual constructs:

    Axes  Read position       Branch  Fixed/runtime Repeat  Peak registers
    1     all                 yes     both                  8..16
    2     before              no      neither               8..14
    2     branch              yes     neither               8..11
    2     fixed/runtime/after no      both                 10..16

    Unrestricted recipes exhausted the same 16 slots and failed allocation 17
    both before and after the shell-WAIT changes. Capacity errors are not caught
    or skipped by the property test.
    """
    rng = random.Random(seed)
    recipes = []
    for repeat in range(3):
        for axes in (1, 2):
            for read_at in ("before", "fixed", "runtime", "branch", "after"):
                adc_samples = rng.choice((50, 100, 150))
                sweep_steps = tuple(rng.randint(5, 30) for _ in range(axes))
                fractional_samples = (rng.randint(20, 70), rng.randint(10, 50))
                recipes.append(
                    WaitRecipe(
                        sweep_steps=sweep_steps,
                        pulse_samples=fractional_samples if axes == 1 else (25, 50),
                        adc_samples=adc_samples,
                        fixed_count=rng.randint(1, 3),
                        counts=tuple(rng.sample((0, 1, 3), 3)),
                        branch_delays=tuple(rng.sample(range(5, 40), 3)),
                        loop_delay=rng.randint(5, 25),
                        read_at=read_at,
                        # Include fully consumed ADCs as well as positive tails.
                        tail=0 if repeat == 0 else rng.randint(10, adc_samples),
                    )
                )
    return recipes


class MixedWaitProgram(TimelineProgramV2):
    """Compile a generated recipe; relax_cycles controls the shell slack guard."""

    def __init__(self, recipe: WaitRecipe, *, relax_cycles: int = 1000):
        self.recipe = recipe
        super().__init__(
            make_mock_soccfg(2, 1), initial_delay=0, relax_delay=relax_cycles / 430.08
        )

    def _initialize(self, cfg):
        recipe = self.recipe
        self.gaps = [
            self.add_sweep(
                f"axis{axis}",
                SweepCfg(
                    start=step / 430.08,
                    stop=3 * step / 430.08,
                    step=step / 430.08,
                    expts=3,
                ),
            )
            for axis, step in enumerate(recipe.sweep_steps)
        ]
        self.counts = LoadValue("counts", recipe.counts, "axis0", "count_cycles")
        self.declare(self.counts)
        self.declare_readout(0, length=recipe.adc_samples / 307.2)
        for ch, samples in enumerate(recipe.pulse_samples):
            self.declare_gen(ch, nqz=1)
            self.add_pulse(
                ch,
                f"pulse{ch}",
                style="const",
                freq=100,
                phase=0,
                gain=0.1,
                length=samples / 384,
            )

    def read(self):
        """Schedule one ADC, then leave the requested tail beyond body end."""
        self.trigger(ros=[0], width=1 / 430.08, timeline="default")
        self.delay_timeline(-self.recipe.tail / 430.08)

    def emit_controls(self):
        """Emit the recipe's budgeted unequal Branch and fixed/runtime Repeats."""
        recipe = self.recipe
        if len(self.gaps) == 1 or recipe.read_at == "branch":
            with self.branch(by="axis0") as branch:
                for delay in recipe.branch_delays:
                    with branch.case():
                        if recipe.read_at == "branch":
                            self.read()
                        self.delay_timeline(delay / 430.08)
        if len(self.gaps) == 1 or recipe.read_at not in ("branch", "before"):
            with self.repeat("fixed", recipe.fixed_count):
                if recipe.read_at == "fixed":
                    self.read()
                self.delay_timeline(recipe.loop_delay / 430.08)
            with self.repeat("runtime", "count_cycles", range_hint=(0, 3)):
                if recipe.read_at == "runtime":
                    self.read()
                self.delay_timeline(recipe.loop_delay / 430.08)

    def _body(self, cfg):
        self.emit(self.counts)
        if self.recipe.read_at == "before":
            self.read()
        self.fork("default", into="parallel")
        self.pulse(0, "pulse0", timeline="default")
        self.delay_timeline(self.gaps[0], timeline="parallel")
        self.pulse(1, "pulse1", timeline="parallel")
        self.join("parallel", into="default")
        self.sync()
        if len(self.gaps) == 2:
            # Both forked timelines are consumed. Advance the reference
            # directly to keep this mixed program within 16 data registers.
            self.delay(self.gaps[1])
        self.emit_controls()
        if self.recipe.read_at == "after":
            self.read()


WAIT_RECIPE_SEED = 0x5EED


@pytest.mark.parametrize(
    "recipe",
    wait_recipes(WAIT_RECIPE_SEED),
    ids=lambda recipe: f"seed={WAIT_RECIPE_SEED}-{recipe!r}",
)
def test_mixed_schedules_shell_wait_matches_every_legal_point(recipe: WaitRecipe):
    # The public analyzer executes linked instructions. It supplies event starts
    # and reference only, so lengths come independently from mock clock specs:
    # 430.08/307.2 = 7/5 ADC cycles; 430.08/384 = 28/25 DAC cycles.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prog = MixedWaitProgram(recipe)
    result = shots(prog)
    assert len(result) == 3 ** len(recipe.sweep_steps)
    remainders = []
    for shot in result:
        triggers = [e.cycle for e in shot.events if e.kind == "data" and e.value]
        adc_end = max(
            (start + (recipe.adc_samples * 7 + 4) // 5 for start in triggers),
            default=shot.start_cycle,
        )
        remainders.append(max(0, adc_end - shot.body_end_cycle))
        # Both drive pulses finish before later control structures. Use their
        # known lengths, not waveform objects or compiler timestamp estimates.
        for start, samples in zip(waves(shot), recipe.pulse_samples, strict=True):
            assert (
                shot.start_cycle + start + (samples * 28 + 24) // 25
                <= shot.body_end_cycle
            ), (shot.loop_indices, recipe)
    upper, lower = max(remainders), min(remainders)
    wait = shell_wait(prog)
    diagnostic = (recipe, remainders, wait, [str(w.message) for w in caught])
    assert wait >= upper, diagnostic
    assert wait == upper, diagnostic
    bound_warnings = [str(w.message) for w in caught if "WAIT" in str(w.message)]
    if upper == lower:
        assert not bound_warnings, diagnostic
    else:
        assert len(bound_warnings) == 1, diagnostic
        assert f"slack={upper - lower} cycles" in bound_warnings[0], diagnostic
        assert "variables=" in bound_warnings[0], diagnostic
        assert "upper bound" not in bound_warnings[0], diagnostic
        # Test both guard boundaries using the independent legal-point extrema.
        with pytest.warns(UserWarning, match=rf"slack={upper - lower} cycles"):
            equal_relax = MixedWaitProgram(recipe, relax_cycles=upper - lower)
        assert shell_wait(equal_relax) == upper, diagnostic
        with pytest.raises(ValueError, match=rf"slack={upper - lower} cycles"):
            MixedWaitProgram(recipe, relax_cycles=upper - lower - 1)


class SweptLengthWaitProgram(TimelineProgramV2):
    """Sweep ten consecutive DAC lengths, starting at samples (384 MHz clocks).

    post_tenths adds tenths of one tProc cycle after the pulse. relax_cycles
    bounds allowed shell slack. A 100-sample ADC starts at the pulse origin.
    """

    def __init__(self, samples: int, post_tenths: int, *, relax_cycles: int = 1000):
        self.samples = samples
        self.post_tenths = post_tenths
        super().__init__(
            make_mock_soccfg(1, 1), initial_delay=0, relax_delay=relax_cycles / 430.08
        )

    def _initialize(self, cfg):
        length = self.add_sweep(
            "length",
            SweepCfg(
                start=self.samples / 384,
                stop=(self.samples + 9) / 384,
                step=1 / 384,
                expts=10,
            ),
        )
        self.drive = PulseCfg(
            ch=0,
            nqz=1,
            freq=100,
            gain=0.1,
            waveform=ConstWaveformCfg(length=length),
            post_delay=self.post_tenths / (10 * 430.08),
        ).build("drive")
        self.declare(self.drive)
        self.declare_readout(0, length=100 / 307.2)

    def _body(self, cfg):
        self.trigger(ros=[0], width=1 / 430.08)
        self.emit(self.drive)


def fractional_length_cases(seed: int) -> list[tuple[int, int]]:
    """Pick six shallow sweeps whose ten points cross fractional ceil steps.

    No Branch or Repeat consumes data registers here, leaving space for the
    nonaffine aggregate-ceil lookup at the 384/430.08 MHz clock seam.
    """
    rng = random.Random(seed)
    return [(samples, rng.randint(1, 9)) for samples in rng.sample(range(20, 60), 6)]


@pytest.mark.parametrize(
    ("samples", "post_tenths"),
    fractional_length_cases(WAIT_RECIPE_SEED + 1),
    ids=lambda value: f"seed={WAIT_RECIPE_SEED + 1}-{value}",
)
def test_fractional_length_shell_wait_uses_exact_nonaffine_point_tails(
    samples: int, post_tenths: int
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prog = SweptLengthWaitProgram(samples, post_tenths)
    result = shots(prog)
    assert len(result) == 10
    tails = []
    for index, shot in enumerate(result):
        (start,) = waves(shot)
        # ceil((samples + index) * 28/25 + post_tenths/10), once, not
        # separate waveform/post ceils. ADC duration is 100 * 7/5 = 140.
        duration = ((samples + index) * 280 + post_tenths * 25 + 249) // 250
        assert shot.body_end_cycle == shot.start_cycle + start + duration
        (trigger,) = [e.cycle for e in shot.events if e.kind == "data" and e.value]
        tails.append(max(0, trigger + 140 - shot.body_end_cycle))
    upper, lower = max(tails), min(tails)
    diagnostic = (WAIT_RECIPE_SEED + 1, samples, post_tenths, tails)
    assert shell_wait(prog) >= upper, diagnostic
    assert shell_wait(prog) == upper, diagnostic
    messages = [str(w.message) for w in caught if "WAIT" in str(w.message)]
    assert len(messages) == 1, diagnostic
    assert f"slack={upper - lower} cycles" in messages[0], diagnostic
    assert "variables=" in messages[0], diagnostic
    assert "upper bound" not in messages[0], diagnostic
    with pytest.warns(UserWarning, match=rf"slack={upper - lower} cycles"):
        equal_relax = SweptLengthWaitProgram(
            samples, post_tenths, relax_cycles=upper - lower
        )
    assert shell_wait(equal_relax) == upper, diagnostic
    with pytest.raises(ValueError, match=rf"slack={upper - lower} cycles"):
        SweptLengthWaitProgram(samples, post_tenths, relax_cycles=upper - lower - 1)
