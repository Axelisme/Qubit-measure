"""Read counts at timeline compilation and QICK acquisition boundaries."""

import pytest
from qick.asm_v2 import Macro, Trigger
from zcu_tools.program.v2 import IQ, TimelineProgramV2, shot_sums
from zcu_tools.program.v2.modules import ConstWaveformCfg, PulseReadout

from tests.program._profiles import load_soccfg


class _ReadProgram(TimelineProgramV2):
    def _initialize(self, cfg):
        self.add_loop("choice", 2)
        self.add_reg("count")
        self.declare_readout(0, length=0.1)


def test_complementary_branches_fail_at_the_first_branch_exit():
    class Program(_ReadProgram):
        def _body(self, cfg):
            for index in (0, 1):
                with (
                    self.branch(by="choice", align_timeline=True) as branch,
                    branch.case(index),
                ):
                    self.trigger([0])
                    self.timeline_inc(0.2)

    with pytest.raises(
        ValueError,
        match=r"Branch 'choice': ADC 0 reads differ across cases: case 0 -> 1, case 1 -> 0",
    ):
        Program(load_soccfg())


class _NativeRead(Macro):
    def __init__(self, ch):
        self.trigger = Trigger(
            ros=[ch], tts=[], pins=[], t=0, width=0.01, ddr4=False, mr=False
        )

    def preprocess(self, prog):
        self.trigger.preprocess(prog)

    def translate(self, prog):
        self.trigger.translate(prog)


@pytest.mark.parametrize("align_timeline", [False, True])
def test_balanced_branch_counts_only_one_case_per_shot(align_timeline):
    class Program(_ReadProgram):
        def _initialize(self, cfg):
            super()._initialize(cfg)
            self.declare_readout(1, length=0.1)

        def _body(self, cfg):
            self.trigger([1])
            self.delay(0.2)
            with self.branch(by="choice", align_timeline=align_timeline) as branch:
                with branch.case(0):
                    self.trigger([0, 1])
                with branch.case(1):
                    self.append_macro(_NativeRead(0))
                    self.trigger([1])

    prog = Program(load_soccfg(n_readouts=2), reps=3)
    assert {ch: ro["trigs"] for ch, ro in prog.ro_chs.items()} == {0: 1, 1: 2}


@pytest.mark.parametrize("count", [3, "count"])
@pytest.mark.parametrize("synchronize", [False, True])
def test_fixed_repeat_adds_its_body_reads_once_per_iteration(count, synchronize):
    class Program(_ReadProgram):
        def _body(self, cfg):
            self.trigger([0])
            self.delay(0.2)
            self.write_reg("count", 3)
            hint = (3, 3) if isinstance(count, str) else None
            with self.repeat("reads", count, range_hint=hint):
                self.trigger([0], timeline="default")
                self.timeline_inc(0.2)
                if synchronize:
                    self.delay_by_timeline()
            self.trigger([0], timeline="default")

    prog = Program(load_soccfg())
    assert prog.ro_chs[0]["trigs"] == 5


@pytest.mark.parametrize("hint", [None, (1, 3)])
def test_register_repeat_rejects_body_reads_even_with_timing_bound(hint):
    class Program(_ReadProgram):
        def _body(self, cfg):
            self.write_reg_op("count", "choice", "+", 1)
            with self.repeat("reads", "count", range_hint=hint):
                self.trigger([0], timeline="default")
                self.timeline_inc(0.2)
                self.delay_by_timeline()

    with pytest.raises(
        ValueError,
        match=r"Repeat 'reads': ADC 0 has 1 reads per iteration.*'count'",
    ):
        Program(load_soccfg())


@pytest.mark.parametrize("supplied", [2, [1, 2], []])
def test_set_reads_per_shot_cannot_override_compiled_values(supplied):
    class Program(_ReadProgram):
        def _body(self, cfg):
            self.trigger([0])

    prog = Program(load_soccfg())
    prog.set_reads_per_shot(1)
    prog.set_reads_per_shot([1])
    assert prog.ro_chs[0]["trigs"] == 1
    with pytest.raises(ValueError, match=r"reads_per_shot.*requested.*compiled.*1"):
        prog.set_reads_per_shot(supplied)
    assert prog.ro_chs[0]["trigs"] == 1


def test_opaque_trigger_is_counted_but_read_label_cannot_resolve_its_index():
    readout = PulseReadout(
        "selected",
        PulseReadout.Cfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=1.0,
        ),
    )

    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(readout)

        def _body(self, cfg):
            self.append_macro(_NativeRead(0))
            self.delay(1.5)
            self.emit(readout)

    prog = Program(load_soccfg())
    with pytest.raises(
        ValueError, match=r"ADC 0.*compiled 2.*observed 1.*opaque triggers"
    ):
        shot_sums(prog, [IQ("selected")])
