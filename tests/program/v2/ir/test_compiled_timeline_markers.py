"""Timeline compilation preserves source ownership before IR optimization."""

import pytest
from qick.qick_asm import AbsQickProgram
from zcu_tools.program.v2.ir.instructions import MetaInst, WaitInst
from zcu_tools.program.v2.ir.io_instructions import PortWriteInst
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import Pulse, PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.timeline import TimelineProgramV2


def _invalidate_binary(prog: AbsQickProgram) -> None:
    # The QICK base owns the nullable public compilation cache.
    prog.binprog = None


def test_shared_pulse_resources_keep_each_emission_source_on_waveform_outputs():
    cfg = PulseCfg(
        ch=0, nqz=1, freq=1000, gain=0.5, waveform=ConstWaveformCfg(length=0.1)
    )
    first = Pulse("first", cfg)
    second = Pulse("second", cfg)

    class Program(TimelineProgramV2):
        def _initialize(self, cfg: None) -> None:
            self.declare(first)
            self.declare(second)

        def _body(self, cfg: None) -> None:
            self.emit(first, tag="first_occurrence")
            self.emit(second, tag="second_occurrence")
            self.emit(first, tag="third_occurrence")

    prog = Program(make_mock_soccfg(), initial_delay=0, relax_delay=0)
    assert first.pulse_name(prog) == second.pulse_name(prog)
    sources = []
    active = []
    boundaries = []
    for record in prog.compiled_stream.records:
        inst = record.instruction
        if isinstance(inst, MetaInst):
            if inst.type == "SOURCE_START":
                active.append(inst.name)
                boundaries.append((inst.type, inst.name))
            elif inst.type == "SOURCE_END":
                assert active.pop() == inst.name
                boundaries.append((inst.type, inst.name))
        elif isinstance(inst, PortWriteInst):
            sources.append(tuple(active))
    assert sources == [("first",), ("second",), ("first",)]
    assert boundaries == [
        ("SOURCE_START", "first"),
        ("SOURCE_END", "first"),
        ("SOURCE_START", "second"),
        ("SOURCE_END", "second"),
        ("SOURCE_START", "first"),
        ("SOURCE_END", "first"),
    ]
    assert active == []


def test_nested_repeats_keep_outer_scan_shot_and_register_count_structure():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg: None) -> None:
            self.add_loop("scan", 3)
            self.add_reg("count")
            self.write_reg("count", 2)

        def _body(self, cfg: None) -> None:
            self.delay_by_timeline()
            with self.repeat("fixed", 2):
                self.delay_by_timeline()
                with self.repeat("variable", "count", range_hint=(0, 4)):
                    self.timeline_inc(0.1)
                    self.delay_by_timeline()
                self.delay_by_timeline()

    prog = Program(make_mock_soccfg(), reps=2)
    starts = {}
    structure = []
    for record in prog.compiled_stream.records:
        inst = record.instruction
        if isinstance(inst, MetaInst) and inst.type.startswith("LOOP_"):
            structure.append((inst.type, inst.name))
            if inst.type == "LOOP_START":
                starts[inst.name] = inst.info
    assert structure == [
        ("LOOP_START", "reps"),
        ("LOOP_BODY_START", "reps"),
        ("LOOP_START", "scan"),
        ("LOOP_BODY_START", "scan"),
        ("LOOP_START", "fixed"),
        ("LOOP_BODY_START", "fixed"),
        ("LOOP_START", "variable"),
        ("LOOP_BODY_START", "variable"),
        ("LOOP_BODY_END", "variable"),
        ("LOOP_END", "variable"),
        ("LOOP_BODY_END", "fixed"),
        ("LOOP_END", "fixed"),
        ("LOOP_BODY_END", "scan"),
        ("LOOP_END", "scan"),
        ("LOOP_BODY_END", "reps"),
        ("LOOP_END", "reps"),
    ]
    for name, count in [("reps", 2), ("scan", 3), ("fixed", 2)]:
        assert starts[name]["n"] == count
        assert starts[name]["counter_reg"] == prog.reg_dict[name].full_addr()
    assert starts["variable"]["counter_reg"] == prog.reg_dict["variable"].full_addr()
    assert starts["variable"]["n"] == prog.reg_dict["count"].full_addr()
    assert starts["variable"]["range_hint"] == (0, 4)


def test_timeline_branch_preserves_order_compare_register_and_body_shell_boundary():
    class Program(TimelineProgramV2):
        def _initialize(self, cfg: None) -> None:
            self.add_loop("selector", 2)

        def _body(self, cfg: None) -> None:
            with self.branch(by="selector", align_timeline=True) as branch:
                with branch.case(0):
                    self.timeline_inc(0.1)
                with branch.case(1):
                    self.timeline_inc(0.2)
            self.wait(t=0.3)

    prog = Program(make_mock_soccfg(), initial_delay=0, relax_delay=0.4)
    markers = [
        inst
        for record in prog.compiled_stream.records
        if isinstance(inst := record.instruction, MetaInst)
    ]
    branches = [inst for inst in markers if inst.type.startswith("BRANCH_")]
    name = branches[0].name
    assert [(inst.type, inst.name) for inst in branches] == [
        ("BRANCH_START", name),
        ("BRANCH_CASE_START", "0"),
        ("BRANCH_CASE_END", "0"),
        ("BRANCH_CASE_START", "1"),
        ("BRANCH_CASE_END", "1"),
        ("BRANCH_END", name),
    ]
    assert branches[0].info["compare_reg"] == prog.reg_dict["selector"].full_addr()
    boundaries = {
        inst.name: record.p_addr
        for record in prog.compiled_stream.records
        if isinstance(inst := record.instruction, MetaInst)
        and inst.type == "TIMING_BOUNDARY"
    }
    waits = [
        record.p_addr
        for record in prog.compiled_stream.records
        if isinstance(record.instruction, WaitInst)
    ]
    assert len(waits) == 2
    assert boundaries["body_start"] <= waits[0] < boundaries["body_end"] <= waits[1]


@pytest.mark.parametrize("timeline", [None, "default"])
def test_native_pulse_tag_and_call_order_sources_reset_only_on_actual_compile(timeline):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg: None) -> None:
            self.declare_gen(0, nqz=1)
            self.add_pulse(
                0, "native", style="const", freq=1000, phase=0, gain=0.5, length=0.1
            )

        def _body(self, cfg: None) -> None:
            self.pulse(0, "native", t="auto", timeline=timeline)
            self.pulse(0, "native", t="auto", tag="tagged", timeline=timeline)
            self.pulse(0, "native", t="auto", timeline=timeline)

    prog = Program(make_mock_soccfg(), initial_delay=0, relax_delay=0)
    saved = prog.compiled_stream
    prog.compile()
    assert prog.compiled_stream is saved
    _invalidate_binary(prog)
    prog.compile()
    assert prog.compiled_stream is not saved
    for snapshot in [saved, prog.compiled_stream]:
        labels = []
        for record in snapshot.records:
            inst = record.instruction
            if isinstance(inst, MetaInst) and inst.type.startswith("SOURCE_"):
                labels.append((inst.type, inst.name))
        assert labels == [
            ("SOURCE_START", "native:pulse:1"),
            ("SOURCE_END", "native:pulse:1"),
            ("SOURCE_START", "tagged"),
            ("SOURCE_END", "tagged"),
            ("SOURCE_START", "native:pulse:3"),
            ("SOURCE_END", "native:pulse:3"),
        ]
