"""Shell WAIT bounds through public timeline compilation and point scheduling."""

import math
import random
import warnings
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import pytest
from qick.asm_v2 import QickSweep1D
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import (
    ComputedPulse,
    DelayAuto,
    LoadValue,
    LoadWord,
    PulseCfg,
)
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import CycleBounds, TimelineProgramV2

from tests.program._profiles import (
    GEN_MHZ,
    RO_MHZ,
    TPROC_MHZ,
    load_mux_soccfg,
    load_soccfg,
)
from tests.program._timing_expectations import (
    generator_duration,
    native_sweep,
    readout_duration,
    runtime_cycles,
    tproc_ceil,
)

from .timing_support import shots, waves

PROFILE = load_soccfg()
ADC50 = tproc_ceil(PROFILE, readout_duration(PROFILE, 50 / RO_MHZ))


def t1_program(
    mux: bool,
    *,
    delay_kind: Literal["uniform", "nonuniform", "static"] = "uniform",
    sequence: Literal["t1", "ramsey", "echo"] = "t1",
) -> TimelineProgramV2:
    """Build synthetic T1 with a 3441-cycle origin span and ten-cycle read post."""
    if mux:
        soccfg = load_mux_soccfg(2, 2, gen_ch=0, ro_chs=(0, 1))
        readout = MuxReadoutCfg(
            gen_ch=0,
            nqz=1,
            mixer_freq=1000,
            length=0.1,
            ro_length=55 / soccfg["readouts"][0]["f_output"],
            trig_offset=12 / TPROC_MHZ,
            post_delay=10 / TPROC_MHZ,
            tones=[
                MuxReadoutTone(ro_ch=ch, freq=1000 + 80 * ch, gain=0.1) for ch in (0, 1)
            ],
        ).build("read")
    else:
        soccfg = load_soccfg(2, 1)
        readout = PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            ro_ch=0,
            freq=100,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=127 / RO_MHZ,
            trig_offset=12 / TPROC_MHZ,
            post_delay=10 / TPROC_MHZ,
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
                1,
                "pi",
                style="const",
                freq=1000,
                phase=0,
                gain=0.5,
                length=25 / GEN_MHZ,
            )
            self.declare(readout)

        def _body(self, cfg):
            self.pulse(1, "pi", timeline="default")
            if delay_kind == "nonuniform":
                self.emit(loader)
                self.emit(native_gap)
            else:
                gap = (
                    QickSweep1D("delay", 43 / TPROC_MHZ, 3484 / TPROC_MHZ)
                    if delay_kind == "uniform"
                    else 43 / TPROC_MHZ
                )
                if sequence == "echo":
                    self.timeline_inc(gap / 2)
                    self.pulse(1, "pi", timeline="default")
                    self.timeline_inc(gap / 2)
                else:
                    self.timeline_inc(gap)
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
            self.declare_readout(0, length=51 / RO_MHZ)

        def _body(self, cfg):
            self.trigger(ros=[0], t=0, width=1 / TPROC_MHZ)
            self.timeline_inc(42 / TPROC_MHZ)

    prog = Program(load_soccfg(0, 1), initial_delay=0, relax_delay=0)
    (shot,) = shots(prog)
    assert shot.body_end_cycle - shot.start_cycle == 42
    physical_tail = (
        readout_duration(prog.soccfg, 51 / RO_MHZ) * Fraction(str(TPROC_MHZ)) - 42
    )
    assert shell_wait(prog) >= physical_tail
    assert shell_wait(prog) == math.ceil(physical_tail)


@pytest.mark.parametrize("relax_cycles", [9, 10])
def test_loadword_overwrite_keeps_exact_table_tail_and_relax_guard(relax_cycles):
    loader = LoadWord("words", [10, 20], "index", "duration", use_existed=True)

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("index", 2)
            self.add_reg("duration")
            self.declare(loader)
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.write_reg("duration", 20)
            self.emit(loader)
            self.trigger(ros=[0], t=0, width=1 / TPROC_MHZ)
            self.timeline_inc("duration")

    if relax_cycles < 10:
        with pytest.raises(
            ValueError, match=r"WAIT.*duration.*slack=10 cycles.*relax=9 cycles"
        ):
            Program(load_soccfg(0, 1), initial_delay=0, relax_delay=9 / TPROC_MHZ)
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prog = Program(
                load_soccfg(0, 1), initial_delay=0, relax_delay=10 / TPROC_MHZ
            )
        result = shots(prog)
        body = [shot.body_end_cycle - shot.start_cycle for shot in result]
        assert body == [10, 20]
        tails = [ADC50 - end for end in body]
        assert tails == pytest.approx([ADC50 - 10, ADC50 - 20])
        messages = [str(warning.message) for warning in caught]
        assert shell_wait(prog) == math.ceil(max(tails)), messages
        assert len(messages) == 1
        assert "duration" in messages[0] and "slack=10 cycles" in messages[0]
        assert "upper bound" not in messages[0]


@pytest.mark.parametrize("mux", [False, True], ids=["single", "mux"])
def test_uniform_t1_shell_wait_cancels_shared_readout_origin(mux):
    prog = t1_program(mux)
    adc_length = 55 / prog.soccfg["readouts"][0]["f_output"] if mux else 127 / RO_MHZ
    adc_end = 12 + tproc_ceil(prog.soccfg, readout_duration(prog.soccfg, adc_length))
    gen_end = tproc_ceil(
        prog.soccfg,
        generator_duration(prog.soccfg, 0.1) + Fraction(10) / Fraction(str(TPROC_MHZ)),
    )
    assert shell_wait(prog) == adc_end - gen_end


def test_two_sweep_reference_debits_keep_each_axis_correlated():
    loader = LoadValue("origin", [40, 60, 50], "origin_axis", "origin_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("origin_axis", 3)
            self.add_loop("advance_axis", 3)
            self.declare(loader)
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.emit(loader)
            self.timeline_inc("origin_cycles")
            self.trigger(ros=[0], width=1 / TPROC_MHZ, timeline="default")
            self.timeline_inc(79 / TPROC_MHZ)
            self.delay(QickSweep1D("advance_axis", 10 / TPROC_MHZ, 30 / TPROC_MHZ))

    prog = Program(load_soccfg(0, 1), initial_delay=0, relax_delay=1)
    # The authored body advance consumes each ADC tail at every Cartesian point.
    expected_body = [origin + 80 for origin in (40, 60, 50) for _ in range(3)]
    result = shots(prog)
    assert [s.body_end_cycle - s.start_cycle for s in result] == expected_body
    assert shell_wait(prog) == 0
    prog.compile()
    assert shots(prog) == result
    assert shell_wait(prog) == 0


@pytest.mark.parametrize("mux", [False, True], ids=["single", "mux"])
@pytest.mark.parametrize("delay_kind", ["nonuniform", "static"])
def test_nonuniform_and_static_t1_keep_constant_adc_shell_wait(mux, delay_kind):
    prog = t1_program(mux, delay_kind=delay_kind)
    adc_length = 55 / prog.soccfg["readouts"][0]["f_output"] if mux else 127 / RO_MHZ
    adc_end = 12 + tproc_ceil(prog.soccfg, readout_duration(prog.soccfg, adc_length))
    gen_end = tproc_ceil(
        prog.soccfg,
        generator_duration(prog.soccfg, 0.1) + Fraction(10) / Fraction(str(TPROC_MHZ)),
    )
    assert shell_wait(prog) == adc_end - gen_end


@pytest.mark.parametrize("mux", [False, True], ids=["single", "mux"])
@pytest.mark.parametrize("sequence", ["ramsey", "echo"])
def test_coherence_shell_wait_matches_pointwise_scheduling_oracle(mux, sequence):
    prog = t1_program(mux, sequence=sequence)
    expected_waves = []
    expected_body = []
    remaining = []
    # Author delay sweeps use native start round and truncated step.
    divisor = 2 if sequence == "echo" else 1
    gaps = native_sweep(
        prog.soccfg, 43 / (divisor * TPROC_MHZ), 3484 / (divisor * TPROC_MHZ), 3
    )
    drive_end = tproc_ceil(
        prog.soccfg, generator_duration(prog.soccfg, 25 / GEN_MHZ, 1)
    )
    read_end = tproc_ceil(
        prog.soccfg,
        generator_duration(prog.soccfg, 0.1) + Fraction(10) / Fraction(str(TPROC_MHZ)),
    )
    adc_length = 55 / prog.soccfg["readouts"][0]["f_output"] if mux else 127 / RO_MHZ
    adc_duration = tproc_ceil(prog.soccfg, readout_duration(prog.soccfg, adc_length))
    for gap in gaps:
        starts = [0]
        end = drive_end
        if sequence == "echo":
            end += gap
            starts.append(end)
            end += drive_end
            end += gap
        else:
            end = math.ceil(end + gap)
        starts.append(end)
        end += drive_end
        starts.append(end)  # Readout starts at the last drive end.
        if not mux:
            starts.append(
                end
            )  # Dynamic readout configuration is also a wave port event.
        expected_waves.append(starts)
        adc_end = end + 12 + adc_duration
        body_end = end + read_end
        expected_body.append(body_end)
        remaining.append(max(0, adc_end - body_end))
    result = shots(prog)
    assert [waves(s) for s in result] == expected_waves
    assert [s.body_end_cycle - s.start_cycle for s in result] == expected_body
    # The shell covers only the ADC remainder beyond generator-plus-post closure.
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
        ro_length=127 / RO_MHZ,
        trig_offset=12 / TPROC_MHZ,
        post_delay=10 / TPROC_MHZ,
    ).build("read")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.start = self.add_sweep(
                "start",
                SweepCfg(
                    start=28 / TPROC_MHZ,
                    stop=84 / TPROC_MHZ,
                    step=28 / TPROC_MHZ,
                    expts=3,
                ),
            )
            for ch, length in ((0, 100 / GEN_MHZ), (1, 50 / GEN_MHZ)):
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
            self.delay_by_timeline()
            self.emit(readout)

    prog = Program(load_soccfg(3, 1), initial_delay=0, relax_delay=1)
    result = shots(prog)
    a_end = tproc_ceil(prog.soccfg, generator_duration(prog.soccfg, 100 / GEN_MHZ))
    b_end = tproc_ceil(prog.soccfg, generator_duration(prog.soccfg, 50 / GEN_MHZ, 1))
    origins = [max(a_end, start + b_end) for start in (28, 56, 84)]
    assert [waves(s) for s in result] == [
        [0, start, origin, origin]
        for start, origin in zip((28, 56, 84), origins, strict=True)
    ]
    trigger_starts = [
        e.cycle - s.start_cycle
        for s in result
        for e in s.events
        if e.kind == "data" and e.value
    ]
    assert trigger_starts == [origin + 12 for origin in origins]
    body_ends = [s.body_end_cycle - s.start_cycle for s in result]
    read_end = tproc_ceil(
        prog.soccfg,
        generator_duration(prog.soccfg, 0.1, 2)
        + Fraction(10) / Fraction(str(TPROC_MHZ)),
    )
    adc_duration = readout_duration(prog.soccfg, 127 / RO_MHZ) * Fraction(
        str(TPROC_MHZ)
    )
    assert body_ends == [origin + read_end for origin in origins]
    remaining = [
        max(0, math.ceil(trigger + adc_duration) - body)
        for trigger, body in zip(trigger_starts, body_ends, strict=True)
    ]
    assert remaining == [12 + math.ceil(adc_duration) - read_end] * 3
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
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.emit(origin)
            self.timeline_inc("origin_cycles")
            self.trigger(ros=[0], width=1 / TPROC_MHZ, timeline="default")
            self.timeline_inc(
                (ADC50 - 1) / TPROC_MHZ
            )  # Explicitly reach the ADC end before post.
            self.timeline_inc(
                QickSweep1D(
                    "post_axis",
                    -tail_cycles[0] / TPROC_MHZ,
                    -tail_cycles[1] / TPROC_MHZ,
                )
            )

    return Program(
        load_soccfg(0, 1), initial_delay=0, relax_delay=relax_cycles / TPROC_MHZ
    )


def test_variable_shell_wait_warns_with_only_uncancelled_axis_and_cycle_slack():
    with pytest.warns(UserWarning, match=r"WAIT.*post_axis.*slack=20 cycles") as caught:
        prog = variable_tail_program()
    assert all("origin" not in str(item.message) for item in caught)
    assert shell_wait(prog) == 30
    # The negative authored post leaves exactly the chosen tail after ADC end.
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        origin + ADC50 - tail for origin in (40, 60, 50) for tail in (10, 20, 30)
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
                self.declare_readout(ch, length=50 / RO_MHZ)
            if feedback:
                self.add_reg("feedback_cycles")

        def _body(self, cfg):
            self.fork("default", into="other")
            for loader, register, timeline, ro in (
                (left, "left_cycles", "default", 0),
                (right, "right_cycles", "other", 1),
            ):
                self.emit(loader)
                self.timeline_inc(register, timeline=timeline)
                self.trigger(ros=[ro], width=1 / TPROC_MHZ, timeline=timeline)
                self.timeline_inc(-200 / TPROC_MHZ, timeline=timeline)
            if feedback:
                with self.time_producer("feedback_cycles", CycleBounds(0, 31)):
                    self.write_reg_op("feedback_cycles", "s11", "AND", 31)
                self.timeline_inc("feedback_cycles")

    return Program(
        load_soccfg(0, 2), initial_delay=0, relax_delay=relax_cycles / TPROC_MHZ
    )


def test_shared_axis_lookup_maximum_warns_with_exact_discrete_slack():
    with pytest.warns(
        UserWarning, match=r"WAIT.*left_cycles.*right_cycles.*slack=50 cycles"
    ):
        prog = opposing_tail_program((0, 50, 100))
    assert shell_wait(prog) == 100 + ADC50
    result = shots(prog)
    assert [s.body_end_cycle - s.start_cycle for s in result] == [0, 0, 0]


@pytest.mark.filterwarnings("error")
def test_shared_axis_lookup_maximum_is_constant_on_two_legal_points():
    prog = opposing_tail_program((0, 100), relax_cycles=0)
    assert shell_wait(prog) == 100 + ADC50


def test_unknown_wait_warns_with_slack_bound_without_rejecting_above_relax():
    with pytest.warns(
        UserWarning,
        match=r"WAIT.*feedback_cycles.*slack=31 cycles.*upper bound.*not actual",
    ):
        prog = opposing_tail_program((0, 100), relax_cycles=0, feedback=True)
    assert shell_wait(prog) == 100 + ADC50


def test_shell_bound_does_not_replace_native_swept_final_reference_advance():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("axis", 3)
            for ch in (0, 1):
                self.declare_readout(ch, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.trigger(
                ros=[0], t=QickSweep1D("axis", 0, 100 / TPROC_MHZ), width=1 / TPROC_MHZ
            )
            self.trigger(
                ros=[1], t=QickSweep1D("axis", 100 / TPROC_MHZ, 0), width=1 / TPROC_MHZ
            )

    with pytest.warns(UserWarning, match=r"WAIT.*axis.*slack=50 cycles"):
        prog = Program(load_soccfg(0, 2), initial_delay=0, relax_delay=100 / TPROC_MHZ)
    assert shell_wait(prog) == 100 + ADC50
    assert [s.end_cycle - s.start_cycle for s in shots(prog)] == [
        max(a, b) + ADC50 + 100 for a, b in ((0, 100), (50, 50), (100, 0))
    ]


def test_table_derived_repeat_count_keeps_exact_wait_slack():
    origin = LoadValue("origin", [40, 40, 40], "axis", "origin_cycles")
    count = LoadValue("counts", [1, 3, 2], "axis", "count_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("axis", 3)
            self.declare(origin)
            self.declare(count)
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.emit(origin)
            self.emit(count)
            self.timeline_inc("origin_cycles")
            self.trigger(ros=[0], width=1 / TPROC_MHZ, timeline="default")
            self.timeline_inc(-200 / TPROC_MHZ)
            self.delay_by_timeline()
            with self.repeat("repeat", "count_cycles", range_hint=(1, 3)):
                self.timeline_inc(10 / TPROC_MHZ)
                self.delay_by_timeline()

    with pytest.warns(
        UserWarning, match=r"WAIT.*count_cycles.*slack=20 cycles"
    ) as caught:
        prog = Program(load_soccfg(0, 1), initial_delay=0, relax_delay=20 / TPROC_MHZ)
    assert all("not actual" not in str(w.message) for w in caught)
    assert shell_wait(prog) == 40 + ADC50 - 10
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [10, 30, 20]


@pytest.mark.parametrize("relax_cycles", [10, 9])
def test_zero_count_repeat_preserves_entry_source_and_exact_wait_guard(relax_cycles):
    counts = LoadValue("counts", [0, 1], "scan", "count_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(counts)
            self.add_reg("duration")
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.emit(counts)
            self.write_reg("duration", 10)
            self.trigger(ros=[0], width=1 / TPROC_MHZ)
            self.delay_by_timeline()
            with self.repeat("inner", "count_cycles", range_hint=(0, 1)):
                self.write_reg("duration", 20)
                self.delay_by_timeline()
            self.timeline_inc("duration")

    if relax_cycles < 10:
        with pytest.raises(
            ValueError, match=r"WAIT.*count_cycles.*slack=10 cycles.*relax=9 cycles"
        ):
            Program(load_soccfg(0, 1), initial_delay=0, relax_delay=9 / TPROC_MHZ)
    else:
        with pytest.warns(
            UserWarning, match=r"WAIT.*count_cycles.*slack=10 cycles"
        ) as caught:
            prog = Program(
                load_soccfg(0, 1), initial_delay=0, relax_delay=10 / TPROC_MHZ
            )
        assert all("upper bound" not in str(item.message) for item in caught)
        body = [s.body_end_cycle - s.start_cycle for s in shots(prog)]
        assert body == [10, 20]
        tails = [ADC50 - end for end in body]
        assert tails == [ADC50 - 10, ADC50 - 20]
        assert shell_wait(prog) == max(tails)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("reuse_after", [False, True])
def test_fixed_repeat_table_advance_sums_legal_iterations_before_wait_projection(
    reuse_after,
):
    delays = LoadValue("delays", [10, 20], "inner", "delay_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(delays)
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.trigger(ros=[0], width=1 / TPROC_MHZ)
            self.delay_by_timeline()
            with self.repeat("inner", 2):
                self.emit(delays)
                self.timeline_inc("delay_cycles")
                self.delay_by_timeline()
            if reuse_after:
                self.timeline_inc("delay_cycles")

    prog = Program(load_soccfg(0, 1), initial_delay=0, relax_delay=0)
    assert shell_wait(prog) == ADC50 - (50 if reuse_after else 30)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        50 if reuse_after else 30
    ]


@pytest.mark.filterwarnings("error")
def test_fixed_repeat_readout_keeps_only_the_last_iteration_adc_tail():
    delays = LoadValue("delays", [10, 20], "inner", "delay_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(delays)
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.delay_by_timeline()
            with self.repeat("inner", 2):
                self.emit(delays)
                self.timeline_inc(60 / TPROC_MHZ)
                self.trigger(ros=[0], width=1 / TPROC_MHZ, timeline="default")
                self.timeline_inc(
                    (ADC50 - 1) / TPROC_MHZ
                )  # Preserve a nonoverlapping ADC cadence.
                self.timeline_inc(-40 / TPROC_MHZ)
                self.timeline_inc("delay_cycles")
                self.delay_by_timeline()

    prog = Program(load_soccfg(0, 1), initial_delay=0, relax_delay=0)
    assert shell_wait(prog) == 20
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [
        2 * (60 + ADC50 - 40) + 10 + 20
    ]


@pytest.mark.parametrize("branch", [False, True])
def test_native_scalar_adc_retains_table_reference_debit(branch):
    delays = LoadValue("delays", [100, 110], "scan", "delay_cycles")

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(delays)
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.trigger(ros=[0], t=60 / TPROC_MHZ, width=1 / TPROC_MHZ)
            self.emit(delays)
            if branch:
                with self.branch(by="scan", align_timeline=False) as cases:
                    for case in range(2):
                        with cases.case(case):
                            self.timeline_inc("delay_cycles")
                            self.timeline_inc(case * 5 / TPROC_MHZ)
            else:
                self.timeline_inc("delay_cycles")

    slack = 15 if branch else 10
    with pytest.warns(UserWarning, match=rf"slack={slack} cycles"):
        prog = Program(
            load_soccfg(0, 1), initial_delay=0, relax_delay=slack / TPROC_MHZ
        )
    assert shell_wait(prog) == 60 + ADC50 - 100
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
                waveform=ConstWaveformCfg(length=length / RO_MHZ),
            )
            for length in (50, 100)
        ],
    )

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("scan", 2)
            self.declare(selector)
            self.declare(playback)
            self.declare_readout(0, length=150 / RO_MHZ)

        def _body(self, cfg):
            self.trigger(ros=[0], width=1 / TPROC_MHZ)
            self.emit(selector)
            self.emit(playback)

    profile = load_soccfg(1, 1)
    # Integer tProc candidates remain exact; only cross-grid candidates use k=8.
    physical = [
        generator_duration(profile, n / RO_MHZ) * Fraction(str(TPROC_MHZ))
        for n in (50, 100)
    ]
    ends = [
        int(ticks)
        if ticks.denominator == 1
        else math.ceil(
            runtime_cycles(profile, int(profile.us2cycles(n / RO_MHZ, gen_ch=0)))
        )
        for n, ticks in zip((50, 100), physical, strict=True)
    ]
    slack = max(ends) - min(ends)
    with pytest.raises(
        ValueError, match=rf"selected.*slack={slack} cycles.*relax={slack - 1} cycles"
    ):
        Program(profile, initial_delay=0, relax_delay=(slack - 1) / TPROC_MHZ)
    with pytest.warns(UserWarning, match=rf"selected.*slack={slack} cycles"):
        prog = Program(profile, initial_delay=0, relax_delay=slack / TPROC_MHZ)
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == ends
    assert shell_wait(prog) == tproc_ceil(
        profile, readout_duration(profile, 150 / RO_MHZ)
    ) - min(ends)


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
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.fork("default", into="other")
            self.emit(left)
            self.emit(right)
            self.timeline_inc("left_cycles")
            self.trigger(ros=[0], width=1 / TPROC_MHZ, timeline="default")
            # Advance 40 authored cycles after origin; the ADC may still have a tail.
            self.timeline_inc(39 / TPROC_MHZ)
            self.timeline_inc("right_cycles", timeline="other")

    prog = Program(load_soccfg(0, 1), initial_delay=0, relax_delay=1)
    expected_body = []
    remaining = []
    for origin in origins:
        for other in others:
            body = max(origin + 40, other)
            expected_body.append(body)
            remaining.append(max(0, origin + ADC50 - body))
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == expected_body
    assert shell_wait(prog) == max(remaining)


@pytest.mark.filterwarnings("error")
def test_same_axis_maximum_keeps_constant_tail_after_cross_axis_join():
    """Opposing two-point sweeps both end at 40, leaving 30 ADC cycles."""

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("i", 2)
            self.add_loop("j", 2)
            self.declare_readout(0, length=50 / RO_MHZ)

        def _body(self, cfg):
            self.trigger(ros=[0], t=0, width=1 / TPROC_MHZ)
            self.fork("default", into=["right", "extra"])
            self.timeline_inc(QickSweep1D("i", 0, 40 / TPROC_MHZ))
            self.timeline_inc(QickSweep1D("i", 40 / TPROC_MHZ, 0), timeline="right")
            self.timeline_inc(QickSweep1D("j", 0, 10 / TPROC_MHZ), timeline="extra")
            self.join("right", "extra", into="default")

    prog = Program(load_soccfg(0, 1), initial_delay=0, relax_delay=0)
    assert shell_wait(prog) == ADC50 - 40
    assert [s.body_end_cycle - s.start_cycle for s in shots(prog)] == [40] * 4


@dataclass(frozen=True)
class WaitRecipe:
    """Reproducible public program inputs; all delays/tails are tProc cycles.

    sweep_steps has one step per three-point axis. counts and branch_delays
    share axis0's index. pulse_samples holds the two native DAC lengths in
    fabric cycles; adc_samples uses ADC output-clock cycles. read_at selects the
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
            load_soccfg(2, 1), initial_delay=0, relax_delay=relax_cycles / TPROC_MHZ
        )

    def _initialize(self, cfg):
        recipe = self.recipe
        self.gaps = [
            self.add_sweep(
                f"axis{axis}",
                SweepCfg(
                    start=step / TPROC_MHZ,
                    stop=3 * step / TPROC_MHZ,
                    step=step / TPROC_MHZ,
                    expts=3,
                ),
            )
            for axis, step in enumerate(recipe.sweep_steps)
        ]
        self.counts = LoadValue("counts", recipe.counts, "axis0", "count_cycles")
        self.declare(self.counts)
        self.declare_readout(0, length=recipe.adc_samples / RO_MHZ)
        for ch, samples in enumerate(recipe.pulse_samples):
            self.declare_gen(ch, nqz=1)
            self.add_pulse(
                ch,
                f"pulse{ch}",
                style="const",
                freq=100,
                phase=0,
                gain=0.1,
                length=samples / GEN_MHZ,
            )

    def read(self, timeline="default"):
        """Schedule one ADC, then leave the requested tail beyond body end."""
        self.trigger(ros=[0], width=1 / TPROC_MHZ, timeline=timeline)
        adc_cycles = tproc_ceil(
            self.soccfg, readout_duration(self.soccfg, self.recipe.adc_samples / RO_MHZ)
        )
        self.timeline_inc((adc_cycles - 1) / TPROC_MHZ)
        self.timeline_inc(-self.recipe.tail / TPROC_MHZ)

    def emit_controls(self):
        """Emit the recipe's budgeted unequal Branch and fixed/runtime Repeats."""
        recipe = self.recipe
        if len(self.gaps) == 1 or recipe.read_at == "branch":
            with self.branch(by="axis0", align_timeline=True) as branch:
                for index, delay in enumerate(recipe.branch_delays):
                    with branch.case(index) as case:
                        if recipe.read_at == "branch":
                            self.read(timeline=case)
                        self.timeline_inc(delay / TPROC_MHZ)
        if len(self.gaps) == 1 or recipe.read_at not in ("branch", "before"):
            self.delay_by_timeline()
            with self.repeat("fixed", recipe.fixed_count):
                if recipe.read_at == "fixed":
                    self.timeline_inc(recipe.tail / TPROC_MHZ)
                    self.read()
                self.timeline_inc(recipe.loop_delay / TPROC_MHZ)
                self.delay_by_timeline()
            self.delay_by_timeline()
            with self.repeat("runtime", "count_cycles", range_hint=(0, 3)):
                if recipe.read_at == "runtime":
                    self.timeline_inc(recipe.tail / TPROC_MHZ)
                    self.read()
                self.timeline_inc(recipe.loop_delay / TPROC_MHZ)
                self.delay_by_timeline()

    def _body(self, cfg):
        self.emit(self.counts)
        if self.recipe.read_at == "before":
            self.read()
        self.fork("default", into="parallel")
        self.pulse(0, "pulse0", timeline="default")
        self.timeline_inc(self.gaps[0], timeline="parallel")
        self.pulse(1, "pulse1", timeline="parallel")
        self.join("parallel", into="default")
        self.delay_by_timeline()
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
    # Linked events supply start/reference, not durations. Physical lengths come
    # independently from caller YAML and public native clock conversion.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prog = MixedWaitProgram(recipe)
    result = shots(prog)
    assert len(result) == 3 ** len(recipe.sweep_steps)
    remainders = []
    for shot in result:
        triggers = [e.cycle for e in shot.events if e.kind == "data" and e.value]
        adc_end = max(
            (
                start
                + tproc_ceil(
                    prog.soccfg,
                    readout_duration(prog.soccfg, recipe.adc_samples / RO_MHZ),
                )
                for start in triggers
            ),
            default=shot.start_cycle,
        )
        remainders.append(max(0, adc_end - shot.body_end_cycle))
        # Both drive pulses finish before later control structures. Use their
        # known lengths, not waveform objects or compiler timestamp estimates.
        for start, samples in zip(waves(shot), recipe.pulse_samples, strict=True):
            assert (
                shot.start_cycle
                + start
                + tproc_ceil(
                    prog.soccfg, generator_duration(prog.soccfg, samples / GEN_MHZ)
                )
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
    """Sweep ten consecutive DAC lengths, starting at samples fabric cycles.

    post_tenths adds tenths of one tProc cycle after the pulse. relax_cycles
    bounds allowed shell slack. A 100-sample ADC starts at the pulse origin.
    """

    def __init__(self, samples: int, post_tenths: int, *, relax_cycles: int = 1000):
        self.samples = samples
        self.post_tenths = post_tenths
        super().__init__(
            load_soccfg(1, 1), initial_delay=0, relax_delay=relax_cycles / TPROC_MHZ
        )

    def _initialize(self, cfg):
        length = self.add_sweep(
            "length",
            SweepCfg(
                start=self.samples / GEN_MHZ,
                stop=(self.samples + 9) / GEN_MHZ,
                step=1 / GEN_MHZ,
                expts=10,
            ),
        )
        self.drive = PulseCfg(
            ch=0,
            nqz=1,
            freq=100,
            gain=0.1,
            waveform=ConstWaveformCfg(length=length),
            post_delay=self.post_tenths / (10 * TPROC_MHZ),
        ).build("drive")
        self.declare(self.drive)
        self.declare_readout(0, length=100 / RO_MHZ)

    def _body(self, cfg):
        self.trigger(ros=[0], width=1 / TPROC_MHZ)
        self.emit(self.drive)


def fractional_length_cases(seed: int) -> list[tuple[int, int]]:
    """Pick six shallow sweeps whose ten points cross fractional ceil steps.

    No Branch or Repeat consumes data registers here, leaving space for the
    fixed-point aggregate conversion between generator and tProc clock grids.
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
        # One aggregate ceil, not separate waveform/post ceils.
        duration = math.ceil(
            runtime_cycles(prog.soccfg, samples + index) + Fraction(post_tenths, 10)
        )
        assert shot.body_end_cycle == shot.start_cycle + start + duration
        (trigger,) = [e.cycle for e in shot.events if e.kind == "data" and e.value]
        tails.append(
            max(
                0,
                trigger
                + tproc_ceil(prog.soccfg, readout_duration(prog.soccfg, 100 / RO_MHZ))
                - shot.body_end_cycle,
            )
        )
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
