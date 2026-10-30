"""Shell WAIT bounds through public timeline compilation and point scheduling."""

import math
from typing import Literal

import pytest
from qick.asm_v2 import QickSweep1D
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.mocksoc import make_mock_soccfg, make_mux_mock_soccfg
from zcu_tools.program.v2.modules import DelayAuto, LoadValue
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

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
