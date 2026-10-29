"""Scheduled-cycle analysis through the public compiler and IR analysis seams."""

from copy import deepcopy

import pytest
from qick.asm_v2 import AsmInst, Pulse, QickSweep1D, Trigger
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.base import MyProgramV2
from zcu_tools.program.v2.mocksoc import make_mock_soccfg


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
            length=10 / 384,
        )

    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.append_macro(Pulse(ch=0, name="p", t=self.first_cycle / 430.08))
        self.delay(5 / 430.08)
        self.append_macro(Pulse(ch=0, name="p", t=20 / 430.08))
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_compiled_scalar_shots_include_shell_and_preserve_boundaries():
    layout_from_metadata = ir.TimingLayout.from_metadata
    prog = ScalarProgram(
        make_mock_soccfg(),
        reps=2,
        initial_delay=20 / 430.08,
        relax_delay=30 / 430.08,
    )
    layout = layout_from_metadata(prog.meta_infos, (ir.TimingLoop("reps", 2),))
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    # Generator length10 / 384 us = 11.2 tProc cycles.
    # Ref20, pulse23, advance5, pulse45; round(20 + 11.2 + relax30)=61.
    assert [(s.start_cycle, s.body_end_cycle, s.end_cycle) for s in result.shots] == [
        (20, 25, 86),
        (86, 91, 152),
    ]
    assert [[e.cycle for e in s.events] for s in result.shots] == [[23, 45], [89, 111]]
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
        self.delay(5 / 430.08)
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_linked_boundary_separates_body_advance_from_mergeable_shell_delay():
    prog = DelayOnlyProgram(
        make_mock_soccfg(),
        reps=2,
        initial_delay=20 / 430.08,
        relax_delay=30 / 430.08,
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
        self.declare_readout(ch=0, length=10 / 307.2)
        self.add_loop("scan", 3)

    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.append_macro(Pulse(ch=0, name="p", t=0))
        self.delay(5 / 430.08)
        self.delay(QickSweep1D("scan", 10 / 430.08, 14 / 430.08))
        self.append_macro(
            Pulse(
                ch=0,
                name="p",
                t=QickSweep1D("scan", 20 / 430.08, 24 / 430.08),
            )
        )
        self.append_macro(
            Trigger(
                ros=[0],
                pins=None,
                tts=None,
                t=QickSweep1D("scan", 30 / 430.08, 34 / 430.08),
                width=2 / 430.08,
                ddr4=False,
                mr=False,
                tag=None,
            )
        )
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_compiled_sweeps_step_and_reset_register_times_and_trigger_edges():
    prog = SweptProgram(
        make_mock_soccfg(),
        reps=2,
        initial_delay=20 / 430.08,
        relax_delay=30 / 430.08,
    )
    before = deepcopy((prog.prog_list, prog.labels, prog.meta_infos))
    layout = ir.TimingLayout.from_metadata(
        prog.meta_infos,
        (ir.TimingLoop("reps", 2), ir.TimingLoop("scan", 3)),
    )
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    # Body adds 5 + (10,12,14); trigger offset=(30,32,34).
    # ADC10 / 307.2 us = 14 tProc cycles; shell=(74,76,78).
    assert [(s.start_cycle, s.body_end_cycle, s.end_cycle) for s in result.shots] == [
        (20, 35, 109),
        (109, 126, 202),
        (202, 221, 299),
        (299, 314, 388),
        (388, 405, 481),
        (481, 500, 578),
    ]
    assert [[e.cycle for e in s.events] for s in result.shots] == [
        [20, 55, 65, 67],
        [109, 148, 158, 160],
        [202, 245, 255, 257],
        [299, 334, 344, 346],
        [388, 427, 437, 439],
        [481, 524, 534, 536],
    ]
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
        self.pulse_from_regs(ch=0, name="p", gain_reg="gain", t=3 / 430.08)
        self.delay(20 / 430.08)
        self.close_inner_loop("repeat", "counter", "count")
        self.meta_macro("TIMING_BOUNDARY", "body_end")


def test_known_runtime_repeat_and_wave_patch_keep_scheduled_times():
    prog = RuntimeProgram(
        make_mock_soccfg(),
        reps=2,
        initial_delay=20 / 430.08,
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


class TriggerProgram(MyProgramV2):
    def _body(self, cfg):
        self.meta_macro("TIMING_BOUNDARY", "body_start")
        self.write_reg("s_out_time", 7)
        self.delay(5 / 430.08)
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
    prog = TriggerProgram(make_mock_soccfg(), initial_delay=20 / 430.08, relax_delay=0)
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

    prog = LaterPulse(make_mock_soccfg(), initial_delay=20 / 430.08, relax_delay=0)
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

    prog = ResyncProgram(make_mock_soccfg(), initial_delay=0, relax_delay=0)
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
    prog = ScalarProgram(make_mock_soccfg(), reps=2, initial_delay=0, relax_delay=0)
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
                length=10 / 384,
            )

    prog = GainSweepProgram(
        make_mock_soccfg(),
        initial_delay=20 / 430.08,
        relax_delay=30 / 430.08,
    )
    layout = ir.TimingLayout.from_metadata(prog.meta_infos, (ir.TimingLoop("scan", 3),))
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout)
    )
    assert [[e.cycle for e in s.events] for s in result.shots] == [
        [23, 45],
        [89, 111],
        [155, 177],
    ]
    assert [s.end_cycle for s in result.shots] == [86, 152, 218]


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
