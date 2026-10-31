"""Temporary semantic RED through the pre-migration public Repeat signature.

Retire this probe when Repeat accepts the new signature. Durable acceptance lives
at tests/program/v2/test_timeline_repeat.py and test_timeline_builder.py, without
advance_ref. Keeping the old required keyword here avoids mistaking a signature
TypeError for the missing reference/tail behavior.
"""

from contextlib import nullcontext

import pytest
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import TimelineProgramV2

from tests.program.v2.timing_support import shots, waves


def compile_program(body, *, branch=False):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            for ch in range(2):
                self.declare_gen(ch, nqz=1)
                self.add_pulse(
                    ch,
                    f"p{ch}",
                    style="const",
                    freq=1000,
                    phase=0,
                    gain=0.5,
                    length=25 / 384,
                )
            if branch:
                self.add_loop("scan", 2)

        def _body(self, cfg):
            body(self)

    return Program(make_mock_soccfg(2, 1), initial_delay=0, relax_delay=0, reps=1)


def test_explicit_native_loop_confirms_fixed_point_oracle():
    def body(prog):
        prog.open_loop(2, name="r")
        prog.pulse(0, "p0", timeline="default")
        prog.delay_timeline(7 / 430.08)
        prog.sync()
        prog.close_loop()
        prog.pulse(1, "p1", timeline="default")

    prog = compile_program(body)
    (shot,) = shots(prog)
    assert waves(shot) == [0, 35, 70]
    assert (shot.body_end_cycle, shot.end_cycle) == (98, 98)
    assert len(prog.reg_dict) == 2
    assert any(
        inst["CMD"] == "TIME"
        and inst["C_OP"] == "inc_ref"
        and inst.get("LIT") == "#35"
        for inst in prog.prog_list
    )


def test_fixed_point_with_explicit_sync_uses_no_tail_carrier():
    def body(prog):
        with prog.repeat("r", 2, advance_ref=False):
            prog.pulse(0, "p0")
            prog.delay_timeline(7 / 430.08)
            prog.sync()
        prog.pulse(1, "p1", timeline="default")

    prog = compile_program(body)
    (shot,) = shots(prog)
    assert waves(shot) == [0, 35, 70]
    assert (shot.body_end_cycle, shot.end_cycle) == (98, 98)
    assert len(prog.reg_dict) == 2
    assert any(
        inst["CMD"] == "TIME"
        and inst["C_OP"] == "inc_ref"
        and inst.get("LIT") == "#35"
        for inst in prog.prog_list
    )


@pytest.mark.parametrize("sync_in_middle", [False, True])
def test_nonfixed_tail_retains_author_reference_position(sync_in_middle):
    def body(prog):
        with prog.repeat("r", 2, advance_ref=False):
            prog.pulse(0, "p0")
            if sync_in_middle:
                prog.sync()
            prog.delay_timeline(7 / 430.08)
        prog.pulse(1, "p1")

    prog = compile_program(body)
    (shot,) = shots(prog)
    assert waves(shot) == [0, 35, 63 if sync_in_middle else 0]
    assert (shot.body_end_cycle, shot.end_cycle) == (
        (70, 91) if sync_in_middle else (70, 70)
    )
    assert len(prog.reg_dict) > 2


@pytest.mark.parametrize("operation", ["sync", "delay", "delay_auto"])
def test_reference_owner_advance_is_allowed_outside_case(operation):
    def body(prog):
        with prog.repeat("r", 2, advance_ref=False):
            prog.pulse(0, "p0")
            if operation == "sync":
                prog.sync()
            elif operation == "delay":
                prog.delay(28 / 430.08)
            else:
                prog.delay_auto()
        prog.pulse(1, "p1")

    (shot,) = shots(compile_program(body))
    assert waves(shot) == [0, 28, 56]
    assert (shot.body_end_cycle, shot.end_cycle) == (56, 84)


@pytest.mark.parametrize("operation", ["sync", "delay", "delay_auto"])
@pytest.mark.parametrize("nested", [False, True])
def test_case_still_rejects_reference_advance_including_repeat(operation, nested):
    def body(prog):
        with prog.branch(by="scan", align_timeline=False) as cases:
            with (
                cases.case(0),
                prog.repeat("r", 2, advance_ref=False) if nested else nullcontext(),
            ):
                prog.delay_timeline(28 / 430.08)
                if operation == "delay":
                    prog.delay(28 / 430.08)
                else:
                    getattr(prog, operation)()

    with pytest.raises(ValueError, match="cannot advance reference"):
        compile_program(body, branch=True)
