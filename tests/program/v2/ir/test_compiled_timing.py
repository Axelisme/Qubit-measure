"""Scheduled-cycle analysis through the public compiler and IR analysis seams."""

from copy import deepcopy

import pytest
from qick.asm_v2 import AsmInst, Pulse, QickSweep1D, Trigger
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.base import MyProgramV2

from tests.program._profiles import GEN_MHZ, RO_MHZ, TPROC_MHZ, load_soccfg
from tests.program._timing_expectations import generator_duration, readout_duration


class ScalarProgram(MyProgramV2):
    first_cycle = 3

    def _initialize(self, cfg):
        self.declare_gen(ch=0, nqz=1)
        self.add_pulse(
            ch=0,
            name="p",
            style="const",
            freq=100,
            phase=0,
            gain=0.1,
            length=10 / GEN_MHZ,
        )

    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.append_macro(Pulse(ch=0, name="p", t=self.first_cycle / TPROC_MHZ))
        self.delay(5 / TPROC_MHZ)
        self.append_macro(Pulse(ch=0, name="p", t=20 / TPROC_MHZ))
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_compiled_scalar_shots_include_shell_and_preserve_boundaries():
    layout_from_metadata = ir.TimingLayout.from_metadata
    prog = ScalarProgram(
        load_soccfg(),
        reps=2,
        initial_delay=20 / TPROC_MHZ,
        relax_delay=30 / TPROC_MHZ,
    )
    layout = layout_from_metadata(prog.meta_infos, (ir.TimingLoop("reps", 2),))
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    # Native shell rounds the final waveform tail plus relax after body advance.
    tail = generator_duration(prog.soccfg, 10 / GEN_MHZ)
    shell = round(20 + tail * TPROC_MHZ + 30)
    length = 5 + shell
    starts = [20 + i * length for i in range(2)]
    assert [(s.start_cycle, s.body_end_cycle, s.end_cycle) for s in result.shots] == [
        (start, start + 5, start + length) for start in starts
    ]
    assert [[e.cycle for e in s.events] for s in result.shots] == [
        [start + 3, start + 25] for start in starts
    ]
    assert [s.loop_indices for s in result.shots] == [(0,), (1,)]
    assert [(e.kind, e.port) for e in result.shots[0].events] == [
        ("wave", 0),
        ("wave", 0),
    ]


class DelayOnlyProgram(MyProgramV2):
    def _initialize(self, cfg):
        self.declare_gen(ch=0, nqz=1)

    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.delay(5 / TPROC_MHZ)
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_linked_boundary_separates_body_advance_from_mergeable_shell_delay():
    prog = DelayOnlyProgram(
        load_soccfg(),
        reps=2,
        initial_delay=20 / TPROC_MHZ,
        relax_delay=30 / TPROC_MHZ,
    )
    layout = ir.TimingLayout.from_metadata(prog.meta_infos, (ir.TimingLoop("reps", 2),))
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    assert [(s.start_cycle, s.body_end_cycle, s.end_cycle) for s in result.shots] == [
        (20, 25, 55),
        (55, 60, 90),
    ]
    assert [s.events for s in result.shots] == [(), ()]


class SweptProgram(ScalarProgram):
    def _initialize(self, cfg):
        super()._initialize(cfg)
        self.declare_readout(ch=0, length=10 / RO_MHZ)
        self.add_loop("scan", 3)

    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.append_macro(Pulse(ch=0, name="p", t=0))
        self.delay(5 / TPROC_MHZ)
        self.delay(QickSweep1D("scan", 10 / TPROC_MHZ, 14 / TPROC_MHZ))
        self.append_macro(
            Pulse(
                ch=0,
                name="p",
                t=QickSweep1D("scan", 20 / TPROC_MHZ, 24 / TPROC_MHZ),
            )
        )
        self.append_macro(
            Trigger(
                ros=[0],
                pins=None,
                tts=None,
                t=QickSweep1D("scan", 30 / TPROC_MHZ, 34 / TPROC_MHZ),
                width=2 / TPROC_MHZ,
                ddr4=False,
                mr=False,
                tag=None,
            )
        )
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_compiled_sweeps_step_and_reset_register_times_and_trigger_edges():
    prog = SweptProgram(
        load_soccfg(),
        reps=2,
        initial_delay=20 / TPROC_MHZ,
        relax_delay=30 / TPROC_MHZ,
    )
    before = deepcopy((prog.prog_list, prog.labels, prog.meta_infos))
    layout = ir.TimingLayout.from_metadata(
        prog.meta_infos,
        (ir.TimingLoop("reps", 2), ir.TimingLoop("scan", 3)),
    )
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    # Body advances and trigger offsets share scan; only ADC tail is cross-grid.
    adc_tail = readout_duration(prog.soccfg, 10 / RO_MHZ) * TPROC_MHZ
    expected_boundaries = []
    expected_events = []
    start = 20
    for scan in [0, 1, 2] * 2:
        body_end = start + 15 + 2 * scan
        trigger = body_end + 30 + 2 * scan
        end = body_end + round(30 + 2 * scan + adc_tail + 30)
        expected_boundaries.append((start, body_end, end))
        expected_events.append([start, body_end + 20 + 2 * scan, trigger, trigger + 2])
        start = end
    assert [
        (s.start_cycle, s.body_end_cycle, s.end_cycle) for s in result.shots
    ] == expected_boundaries
    assert [[e.cycle for e in s.events] for s in result.shots] == expected_events
    assert [s.loop_indices for s in result.shots] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert [(e.kind, e.port, e.value) for e in result.shots[0].events] == [
        ("wave", 0, None),
        ("wave", 0, None),
        ("data", 0, 1),
        ("data", 0, 0),
    ]
    assert (prog.prog_list, prog.labels, prog.meta_infos) == before


class RuntimeProgram(ScalarProgram):
    def _initialize(self, cfg):
        super()._initialize(cfg)
        self.add_reg("count")
        self.add_reg("counter")
        self.add_reg("gain")
        self.write_reg("count", 2)
        self.write_reg("gain", 100)

    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.open_inner_loop("repeat", "counter", "count")
        self.pulse_from_regs(ch=0, name="p", gain_reg="gain", t=3 / TPROC_MHZ)
        self.delay(20 / TPROC_MHZ)
        self.close_inner_loop("repeat", "counter", "count")
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_known_runtime_repeat_and_wave_patch_keep_scheduled_times():
    prog = RuntimeProgram(
        load_soccfg(),
        reps=2,
        initial_delay=20 / TPROC_MHZ,
        relax_delay=0,
    )
    layout = ir.TimingLayout.from_metadata(prog.meta_infos, (ir.TimingLoop("reps", 2),))
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    assert [(s.start_cycle, s.body_end_cycle, s.end_cycle) for s in result.shots] == [
        (20, 60, 60),
        (60, 100, 100),
    ]
    assert [[e.cycle for e in s.events] for s in result.shots] == [[23, 43], [63, 83]]


class RegisterDispatchProgram(ScalarProgram):
    def _initialize(self, cfg):
        super()._initialize(cfg)
        self.add_reg("count")
        self.add_reg("counter")
        self.write_reg("count", 3)

    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.open_inner_loop("repeat", "counter", "count", range_hint=(2, 4))
        self.append_macro(Pulse(ch=0, name="p", t=0))
        self.delay(25 / TPROC_MHZ)
        self.close_inner_loop("repeat", "counter", "count")
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_native_register_loop_dispatch_preserves_events_and_reference():
    prog = RegisterDispatchProgram(load_soccfg(), initial_delay=0, relax_delay=0)
    layout = ir.TimingLayout.from_metadata(prog.meta_infos)
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    assert [e.cycle for e in result.shots[0].events] == [0, 25, 50]
    assert result.shots[0].body_end_cycle == 75
    assert result.shots[0].end_cycle == 75


class AbsoluteTimeProgram(MyProgramV2):
    input_value: int | str = 7

    def _initialize(self, cfg):
        # Reserve explicit registers for the native assembly macro below.
        self.add_reg("operand", addr=10)
        self.add_reg("magnitude", addr=11)
        self.add_reg("r15", addr=15)
        self.write_reg("operand", self.input_value)

    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.asm_inst({"CMD": "REG_WR", "DST": "r11", "SRC": "op", "OP": "ABS r10"})
        self.write_reg("s_out_time", "magnitude")
        self.asm_inst({"CMD": "TRIG", "DST": "2", "SRC": "set"})
        self.cond_jump("zero_magnitude", "magnitude", "Z")
        self.delay(5 / TPROC_MHZ)
        self.label("zero_magnitude")
        self.asm_inst({"CMD": "TIME", "C_OP": "inc_ref", "R1": "r11"})
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def compile_absolute_time(value: int | str) -> ir.CompiledTimingInput:
    class Program(AbsoluteTimeProgram):
        input_value = value

    prog = Program(load_soccfg(), initial_delay=0, relax_delay=0)
    return ir.CompiledTimingInput(
        prog.prog_list, prog.labels, ir.TimingLayout.from_metadata(prog.meta_infos)
    )


@pytest.mark.parametrize(
    "value,event_cycle,end_cycle",
    [
        (7, 7, 12),
        (-7, 7, 12),
        (0, 0, 0),
        (2147483647, 2147483647, 2147483652),
        (-2147483647, 2147483647, 2147483652),
    ],
)
def test_native_absolute_register_drives_event_reference_and_control(
    value, event_cycle, end_cycle
):
    result = ir.analyze_timing(compile_absolute_time(value))
    shot = result.shots[0]
    assert [(e.kind, e.port, e.value, e.cycle) for e in shot.events] == [
        ("trigger", 2, 1, event_cycle)
    ]
    assert (shot.start_cycle, shot.body_end_cycle, shot.end_cycle) == (
        0,
        end_cycle,
        end_cycle,
    )
    assert result.end_cycle == end_cycle


def test_native_absolute_minimum_signed_word_rejects_uncertain_overflow():
    with pytest.raises(ValueError, match="ABS.*-2147483648.*overflow"):
        ir.analyze_timing(compile_absolute_time(-2147483648))


@pytest.mark.parametrize(
    "value,message",
    [
        ("r15", "Unknown register value: s14"),
        ("s11", "s11"),
    ],
)
def test_native_absolute_unknown_time_sources_are_not_zero(value, message):
    with pytest.raises(ValueError, match=message):
        ir.analyze_timing(compile_absolute_time(value))


def compile_immediate_flags(operand: str, initial: int, literal: int, condition: str):
    class Program(MyProgramV2):
        def _initialize(self, cfg):
            self.add_reg("value", addr=10)
            self.write_reg("value", initial)

        def _body(self, cfg):
            self.meta_macro("TIMING_BOUNDARY", "body_start")
            self.asm_inst(
                {
                    "CMD": "REG_WR",
                    "DST": "r10",
                    "SRC": "imm",
                    "LIT": f"#{literal}",
                    "OP": operand,
                    "UF": "1",
                }
            )
            self.asm_inst({"CMD": "JUMP", "LABEL": "selected", "IF": condition})
            self.delay(7 / TPROC_MHZ)
            self.label("selected")
            self.asm_inst({"CMD": "TIME", "C_OP": "inc_ref", "R1": "r10"})
            self.asm_inst({"CMD": "TRIG", "DST": "2", "SRC": "set", "TIME": "@3"})
            self.meta_macro("TIMING_BOUNDARY", "body_end")

    prog = Program(load_soccfg(), initial_delay=0, relax_delay=0)
    return ir.CompiledTimingInput(
        prog.prog_list, prog.labels, ir.TimingLayout.from_metadata(prog.meta_infos)
    )


@pytest.mark.parametrize(
    "operand,initial,literal,condition,end",
    [
        ("s0", 9, 5, "Z", 5),
        ("r10", 0, 5, "Z", 5),
        ("r10", -3, 5, "S", 5),
        ("r10", 9, 0, "NZ", 0),
        ("r10", 9, 5, "Z", 12),
    ],
)
def test_immediate_write_uses_prewrite_alu_flags_and_literal_destination(
    operand, initial, literal, condition, end
):
    result = ir.analyze_timing(
        compile_immediate_flags(operand, initial, literal, condition)
    )
    (shot,) = result.shots
    # The ALU reads the old operand, independently of literal writeback.
    # A taken branch skips seven cycles; reference then adds the new literal.
    assert (shot.start_cycle, shot.body_end_cycle, shot.end_cycle) == (0, end, end)
    assert [(e.kind, e.cycle) for e in shot.events] == [("trigger", end + 3)]


@pytest.mark.parametrize(
    "operand,message",
    [("s11", "s11.*s_usr_time"), ("r20", "Unknown condition flags")],
)
def test_immediate_flag_alu_rejects_hardware_or_unknown_branch_sources(
    operand, message
):
    with pytest.raises(ValueError, match=message):
        ir.analyze_timing(compile_immediate_flags(operand, 0, 5, "Z"))


class TriggerProgram(MyProgramV2):
    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.write_reg("s_out_time", 7)
        self.delay(5 / TPROC_MHZ)
        self.append_macro(
            AsmInst(
                inst={"CMD": "TRIG", "DST": "2", "SRC": "set", "TIME": "@3"},
                addr_inc=1,
            )
        )
        self.append_macro(
            AsmInst(
                inst={"CMD": "TRIG", "DST": "2", "SRC": "clr"},
                addr_inc=1,
            )
        )
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_literal_time_overrides_register_time_and_trigger_levels_are_distinct():
    prog = TriggerProgram(load_soccfg(), initial_delay=20 / TPROC_MHZ, relax_delay=0)
    layout = ir.TimingLayout.from_metadata(prog.meta_infos)
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    assert [(e.kind, e.port, e.value, e.cycle) for e in result.shots[0].events] == [
        ("trigger", 2, 1, 28),
        ("trigger", 2, 0, 32),
    ]
    assert result.shots[0].body_end_cycle == 25


@pytest.mark.parametrize(
    "bad_instruction, message",
    [
        (
            {"CMD": "WPORT_WR", "DST": "0", "SRC": "wmem", "ADDR": "&0"},
            "Unknown register value: s14",
        ),
        (
            {"CMD": "TIME", "C_OP": "inc_ref", "R1": "r31"},
            "Unknown register value: r31",
        ),
        (
            {"CMD": "TIME", "C_OP": "updt", "LIT": "#0"},
            "Unsupported reference operation",
        ),
        ({"CMD": "JUMP", "LABEL": "HERE", "IF": "NZ"}, "Unknown condition flags"),
        ({"CMD": "WAIT", "C_OP": "port_dt"}, "Unsupported wait control"),
        ({"CMD": "CALL", "LABEL": "HERE"}, "Unsupported timing instruction"),
        ({"CMD": "WMEM_WR", "DST": "&0", "IF": "NZ"}, "Unknown condition flags"),
    ],
)
def test_unknown_timing_or_control_fails_instead_of_returning_precise_output(
    bad_instruction,
    message,
):
    compiled = ir.CompiledTimingInput(
        [dict(bad_instruction, P_ADDR=0)],
        {},
        ir.TimingLayout(0, 0),
    )
    with pytest.raises(ValueError, match=message):
        ir.analyze_timing(compiled)


def test_wait_rejects_a_target_that_would_change_control_flow():
    compiled = ir.CompiledTimingInput(
        [
            {"CMD": "REG_WR", "DST": "s15", "SRC": "imm", "LIT": "#100", "P_ADDR": 0},
            {"CMD": "WAIT", "C_OP": "time", "TIME": "@0", "ADDR": "s15", "P_ADDR": 1},
            {"CMD": "JUMP", "LABEL": "HERE", "P_ADDR": 3},
        ],
        {},
        ir.TimingLayout(0, 3),
    )
    with pytest.raises(ValueError, match="WAIT target"):
        ir.analyze_timing(compiled)


@pytest.mark.parametrize(
    ("left", "right", "minimum", "maximum"),
    [
        (20, 100, 20, 100),
        (100, 20, 20, 100),
        (-20, 100, -20, 100),
        (100, -20, -20, 100),
    ],
)
def test_sign_condition_selects_pointwise_minimum_and_is_sensitive_to_inversion(
    left, right, minimum, maximum
):
    instructions = [
        {"CMD": "REG_WR", "DST": "r0", "SRC": "imm", "LIT": f"#{left}"},
        {"CMD": "REG_WR", "DST": "r1", "SRC": "imm", "LIT": f"#{right}"},
        {"CMD": "TEST", "OP": "r0 - r1", "UF": "1"},
        {"CMD": "REG_WR", "DST": "r2", "SRC": "op", "OP": "r0"},
        {"CMD": "REG_WR", "DST": "r2", "SRC": "op", "OP": "r1", "IF": "NS"},
        {"CMD": "REG_WR", "DST": "r2", "SRC": "op", "OP": "r2 + #200"},
        {"CMD": "TIME", "C_OP": "inc_ref", "R1": "r2"},
        {"CMD": "JUMP", "LABEL": "HERE"},
    ]
    compiled = [
        {**inst, "P_ADDR": address} for address, inst in enumerate(instructions)
    ]
    normal = ir.analyze_timing(
        ir.CompiledTimingInput(compiled, {}, ir.TimingLayout(0, 7))
    )
    assert normal.shots[0].body_end_cycle == 200 + minimum
    compiled[4] = {**compiled[4], "IF": "S"}
    inverted = ir.analyze_timing(
        ir.CompiledTimingInput(compiled, {}, ir.TimingLayout(0, 7))
    )
    assert inverted.shots[0].body_end_cycle == 200 + maximum


def test_known_false_output_condition_does_not_emit_an_event():
    compiled = ir.CompiledTimingInput(
        [
            {"CMD": "REG_WR", "DST": "r0", "SRC": "imm", "LIT": "#0", "P_ADDR": 0},
            {"CMD": "TEST", "OP": "r0", "P_ADDR": 1},
            {
                "CMD": "WPORT_WR",
                "DST": "0",
                "SRC": "wmem",
                "ADDR": "&0",
                "TIME": "@9",
                "IF": "NZ",
                "P_ADDR": 2,
            },
            {
                "CMD": "WPORT_WR",
                "DST": "0",
                "SRC": "wmem",
                "ADDR": "&0",
                "TIME": "@3",
                "P_ADDR": 3,
            },
            {"CMD": "JUMP", "LABEL": "HERE", "P_ADDR": 4},
        ],
        {},
        ir.TimingLayout(0, 4),
    )
    result = ir.analyze_timing(compiled)
    assert [e.cycle for e in result.shots[0].events] == [3]


def test_changing_one_compiled_pulse_time_is_observable():
    class LaterPulse(ScalarProgram):
        first_cycle = 4

    prog = LaterPulse(load_soccfg(), initial_delay=20 / TPROC_MHZ, relax_delay=0)
    layout = ir.TimingLayout.from_metadata(prog.meta_infos)
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    # Independent negative control: the scalar acceptance above expects 23,
    # and must reject this program's deliberately late first pulse at cycle24.
    assert result.shots[0].events[0].cycle == 24
    assert result.shots[0].events[0].cycle != 23


def test_resync_rejects_hardware_time_and_identifies_its_source():
    class ResyncProgram(MyProgramV2):
        def _body(self, cfg):
            self.meta_macro("TIMING_BOUNDARY", "body_start")
            self.resync(0.1)
            self.meta_macro("TIMING_BOUNDARY", "body_end")

    prog = ResyncProgram(load_soccfg(), initial_delay=0, relax_delay=0)
    layout = ir.TimingLayout.from_metadata(prog.meta_infos)
    compiled = ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    with pytest.raises(ValueError, match="s11"):
        ir.analyze_timing(compiled)


@pytest.mark.parametrize(
    "loops, message",
    [
        ((), "shot count"),
        ((ir.TimingLoop("scan", 0),), "positive"),
        ((ir.TimingLoop("scan", 1), ir.TimingLoop("scan", 1)), "unique"),
    ],
)
def test_layout_axes_validate_but_do_not_override_executed_shots(loops, message):
    prog = ScalarProgram(load_soccfg(), reps=2, initial_delay=0, relax_delay=0)
    layout = ir.TimingLayout.from_metadata(prog.meta_infos, loops)
    with pytest.raises(ValueError, match=message):
        ir.analyze_timing(ir.CompiledTimingInput(prog.prog_list, prog.labels, layout))


def test_analysis_rejects_nonterminating_control_flow_with_a_step_limit():
    compiled = ir.CompiledTimingInput(
        [
            {"CMD": "JUMP", "LABEL": "HERE", "P_ADDR": 0},
            {"CMD": "NOP", "P_ADDR": 1},
        ],
        {},
        ir.TimingLayout(0, 0),
    )
    with pytest.raises(ValueError, match="max_steps"):
        ir.analyze_timing(compiled, max_steps=3)


@pytest.mark.parametrize(
    "metadata, message",
    [
        ([], "Missing"),
        (
            [
                {
                    "kind": "meta",
                    "type": "TIMING_BOUNDARY",
                    "name": "body_start",
                    "p_addr": 0,
                }
            ]
            * 2,
            "Duplicate",
        ),
        (
            [
                {
                    "kind": "meta",
                    "type": "TIMING_BOUNDARY",
                    "name": "body_end",
                    "p_addr": "0",
                }
            ],
            "integer",
        ),
    ],
)
def test_layout_reports_unusable_boundary_metadata(metadata, message):
    with pytest.raises(ValueError, match=message):
        ir.TimingLayout.from_metadata(metadata)


def test_compiled_wmem_attaches_scalar_wave_words_without_mutating_inputs():
    prog = ScalarProgram(load_soccfg(), reps=2, initial_delay=0, relax_delay=0)
    layout = ir.TimingLayout.from_metadata(prog.meta_infos, (ir.TimingLoop("reps", 2),))
    expected = tuple(
        int(prog.waves[0][name])
        for name in ("freq", "phase", "env", "gain", "length", "conf")
    )
    wmem = ir.compiled_wmem(prog)
    before = deepcopy((prog.prog_list, prog.labels, wmem))
    compiled = ir.CompiledTimingInput(prog.prog_list, prog.labels, layout, wmem=wmem)
    result = ir.analyze_timing(compiled)
    assert [[event.wave_words for event in shot.events] for shot in result.shots] == [
        [expected, expected],
        [expected, expected],
    ]
    assert (prog.prog_list, prog.labels, wmem) == before
    assert ir.analyze_timing(compiled) == result


@pytest.mark.parametrize(
    "parameter,start,stop", [("gain", -0.2, 0.3), ("freq", 97, 103)]
)
def test_wave_words_follow_native_sweep_updates_and_reset(parameter, start, stop):
    def compile_parameter(value, *, swept=False, reps=1):
        class ParameterProgram(ScalarProgram):
            def _initialize(self, cfg):
                self.declare_gen(ch=0, nqz=1)
                if swept:
                    self.add_loop("scan", 3)
                params = {"freq": 100, "gain": 0.1}
                params[parameter] = value
                self.add_pulse(
                    ch=0,
                    name="p",
                    style="const",
                    phase=0,
                    length=10 / GEN_MHZ,
                    **params,
                )

        return ParameterProgram(
            load_soccfg(), reps=reps, initial_delay=0, relax_delay=0
        )

    prog = compile_parameter(QickSweep1D("scan", start, stop), swept=True, reps=2)
    expected = []
    for point in prog.get_pulse_param("p", parameter, as_array=True).flatten():
        scalar = compile_parameter(float(point))
        expected.append(ir.compiled_wmem(scalar)[0])
    layout = ir.TimingLayout.from_metadata(
        prog.meta_infos, (ir.TimingLoop("reps", 2), ir.TimingLoop("scan", 3))
    )
    wmem = ir.compiled_wmem(prog)
    before = deepcopy(wmem)
    compiled = ir.CompiledTimingInput(prog.prog_list, prog.labels, layout, wmem=wmem)
    result = ir.analyze_timing(compiled)
    assert [[event.wave_words for event in shot.events] for shot in result.shots] == [
        [words, words] for words in expected * 2
    ]
    assert wmem == before
    assert ir.analyze_timing(compiled) == result


def test_native_waveform_sweep_updates_do_not_require_waveform_contents():
    class GainSweepProgram(ScalarProgram):
        def _initialize(self, cfg):
            self.declare_gen(ch=0, nqz=1)
            self.add_loop("scan", 3)
            self.add_pulse(
                ch=0,
                name="p",
                style="const",
                freq=100,
                phase=0,
                gain=QickSweep1D("scan", 0.1, 0.3),
                length=10 / GEN_MHZ,
            )

    prog = GainSweepProgram(
        load_soccfg(),
        initial_delay=20 / TPROC_MHZ,
        relax_delay=30 / TPROC_MHZ,
    )
    layout = ir.TimingLayout.from_metadata(prog.meta_infos, (ir.TimingLoop("scan", 3),))
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    tail = generator_duration(prog.soccfg, 10 / GEN_MHZ)
    length = 5 + round(20 + tail * TPROC_MHZ + 30)
    starts = [20 + i * length for i in range(3)]
    assert [[e.cycle for e in s.events] for s in result.shots] == [
        [start + 3, start + 25] for start in starts
    ]
    assert [s.end_cycle for s in result.shots] == [start + length for start in starts]


def test_unknown_wave_data_invalidates_updated_flags_before_control_flow():
    compiled = ir.CompiledTimingInput(
        [
            {"CMD": "REG_WR", "DST": "r0", "SRC": "imm", "LIT": "#0", "P_ADDR": 0},
            {"CMD": "TEST", "OP": "r0", "P_ADDR": 1},
            {
                "CMD": "REG_WR",
                "DST": "r_wave",
                "SRC": "wmem",
                "ADDR": "&0",
                "UF": "1",
                "P_ADDR": 2,
            },
            {"CMD": "JUMP", "LABEL": "HERE", "IF": "Z", "P_ADDR": 3},
        ],
        {},
        ir.TimingLayout(0, 3),
    )
    with pytest.raises(ValueError, match="Unknown condition flags"):
        ir.analyze_timing(compiled)
