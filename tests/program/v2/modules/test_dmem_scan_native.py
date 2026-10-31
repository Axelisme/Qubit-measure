"""ScanWith native composition through linked timing and dmem consumers."""

from typing import override

import pytest
from qick.asm_v2 import Macro, QickParam, QickSweep1D, WriteReg
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules import Delay, LoadValue, Module, ScanWith
from zcu_tools.program.v2.timeline import TimelineProgramV2


class ScalarConsumer(Module):
    """Expose a scalar on a data port and use it as a timeline delay in cycles."""

    name = "consumer"

    def __init__(self, register: str):
        self.register = register

    @override
    def init(self, prog):
        raise NotImplementedError("Native-only synthetic consumer")

    @override
    def run(self, prog, t=0.0):
        raise NotImplementedError("Native-only synthetic consumer")

    @override
    def declare(self, prog):
        pass

    @override
    def lower(self, prog, *, timeline, tag):
        parent = "default" if timeline is None else timeline
        prog.sync(timeline=parent)
        if self.register != "output":
            prog.write_reg("output", self.register)
        prog.asm_inst(
            {"CMD": "DPORT_WR", "DST": "0", "SRC": "reg", "DATA": "r10", "TIME": "@0"}
        )
        prog.delay_timeline(self.register, timeline=parent)


def analyze(prog: TimelineProgramV2) -> ir.TimingAnalysis:
    return ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list, prog.labels, prog.timing_layout, prog.compile_datamem()
        )
    )


def compile_scan(
    scan: ScanWith,
    *,
    timeline: str = "default",
    reps: int = 1,
    axis_count: int = 1,
    emissions: int = 1,
    prefix: float | QickParam = 7 / 430.08,
) -> TimelineProgramV2:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("axis", axis_count)
            self.add_reg("output", addr=10)
            self.declare(scan)

        def _body(self, cfg):
            if timeline != "default":
                self.fork("default", into=timeline)
            self.delay_timeline(prefix, timeline=timeline)
            for _ in range(emissions):
                self.emit(scan, timeline=timeline)

    return Program(
        make_mock_soccfg(), reps=reps, initial_delay=5 / 430.08, relax_delay=7 / 430.08
    )


def test_scan_loads_each_value_before_ordered_children_and_closes_each_point():
    scan = ScanWith("scan", [4, 9, 2], "value").add_content(
        [
            Delay("before", 3 / 430.08),
            ScalarConsumer("value"),
            Delay("after", 2 / 430.08),
        ]
    )

    prog = compile_scan(scan)
    (shot,) = analyze(prog).shots
    words = prog.compile_datamem()
    assert words is not None
    assert words.tolist() == [4, 9, 2]
    # Entry 5+7=12. Each point adds before=3, value, after=2:
    # 12 -> event 15 -> 21; 21 -> event 24 -> 35; 35 -> event 38 -> 42.
    assert [(event.cycle, event.value) for event in shot.events] == [
        (15, 4),
        (24, 9),
        (38, 2),
    ]
    assert (shot.body_end_cycle, shot.end_cycle) == (42, 49)


def test_repeated_scan_emissions_restart_index_and_share_only_declared_resources():
    scan = ScanWith("scan", [4, 9, 2], "value").add_content(ScalarConsumer("value"))
    prog = compile_scan(scan, emissions=2, reps=2)
    result = analyze(prog)
    assert [
        [(event.cycle - shot.start_cycle, event.value) for event in shot.events]
        for shot in result.shots
    ] == [[(7, 4), (11, 9), (20, 2), (22, 4), (26, 9), (35, 2)]] * 2
    assert [
        (shot.body_end_cycle - shot.start_cycle, shot.end_cycle - shot.start_cycle)
        for shot in result.shots
    ] == [(37, 44)] * 2
    words = prog.compile_datamem()
    assert words is not None
    assert words.tolist() == [4, 9, 2]
    prog.compile()
    assert analyze(prog) == result
    rebuilt = compile_scan(scan, emissions=2, reps=2)
    assert analyze(rebuilt) == result


class TranslateCounterOverwrite(Macro):
    def translate(self, prog):
        WriteReg(dst="scan_count", src=1).translate(prog)


class CounterOverwrite(ScalarConsumer):
    def __init__(self, mode: str):
        super().__init__("value")
        self.mode = mode

    @override
    def lower(self, prog, *, timeline, tag):
        if self.mode == "translate":
            prog.append_macro(TranslateCounterOverwrite())
        elif self.mode == "arithmetic":
            prog.write_reg_op("scan_count", "scan_count", "+", 0)
        elif self.mode == "branch":
            with prog.branch(by="axis", align_timeline=True) as branch, branch.case(0):
                prog.write_reg("scan_count", 1)
        else:
            prog.write_reg("scan_count", 1)
        super().lower(prog, timeline=timeline, tag=tag)
        if self.mode == "reset":
            # A later valid reset cannot hide the unauthorized earlier write.
            prog.write_reg("scan_count", 0)


def compile_fixed_lookup(loader: LoadValue, consumer: ScalarConsumer):
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_loop("axis", 1)
            self.add_reg("output", addr=10)
            self.declare(loader)
            self.declare(consumer)

        def _body(self, cfg):
            with self.repeat("scan_count", 3):
                self.emit(loader)
                self.emit(consumer)

    return Program(make_mock_soccfg(), initial_delay=0, relax_delay=0)


@pytest.mark.parametrize(
    "mode", ["native", "translate", "arithmetic", "branch", "reset"]
)
def test_fixed_counter_rejects_even_in_range_unrelated_native_write(mode):
    with pytest.raises(
        ValueError, match="scan_count: unregistered.*write at instruction"
    ):
        compile_fixed_lookup(
            LoadValue("table", [4, 9, 2], "scan_count", "value"), CounterOverwrite(mode)
        )


@pytest.mark.parametrize("timeline", ["default", "scan_timeline"])
def test_nested_scans_reset_at_each_outer_index_sweep_shot_and_recompile(timeline):
    inner = ScanWith("inner", [1, 3], "inner_value").add_content(
        [Delay("inner_before", 1 / 430.08), ScalarConsumer("inner_value")]
    )
    scan = ScanWith("outer", [2, 5], "outer_value").add_content(
        [ScalarConsumer("outer_value"), None, inner]
    )
    prog = compile_scan(
        scan,
        timeline=timeline,
        reps=2,
        axis_count=2,
        prefix=QickSweep1D("axis", 7 / 430.08, 11 / 430.08),
    )
    shots = analyze(prog).shots
    # First point: outer 2 then inner (1+1)+(1+3) => 8 cycles.
    # Second point: outer 5 then inner (1+1)+(1+3) => 11 cycles.
    # Prefix 7/11 plus 19, then shell 7. Every outer iteration restarts inner 0.
    assert [
        [(event.cycle - shot.start_cycle, event.value) for event in shot.events]
        for shot in shots
    ] == [
        [(7, 2), (10, 1), (12, 3), (15, 5), (21, 1), (23, 3)],
        [(11, 2), (14, 1), (16, 3), (19, 5), (25, 1), (27, 3)],
    ] * 2
    assert [
        (shot.body_end_cycle - shot.start_cycle, shot.end_cycle - shot.start_cycle)
        for shot in shots
    ] == [(26, 33), (30, 37)] * 2
    assert [(loop.name, loop.count) for loop in prog.timing_layout.loops] == [
        ("reps", 2),
        ("axis", 2),
    ]
    assert len(prog.fixed_loop_indices) == 2
    for index in prog.fixed_loop_indices:
        assert index.counter == index.name
        assert (index.count, index.bounds.minimum, index.bounds.maximum) == (2, 0, 1)
    prog.compile()
    assert analyze(prog).shots == shots


def test_fixed_count_proves_full_index_range_not_just_first_lookup():
    loader = LoadValue("short_table", [7, 8], "scan_count", "other")
    with pytest.raises(
        ValueError,
        match=r"other: index dependency scan_count allowed \[0, 1\], actual \[0, 2\]",
    ):
        compile_fixed_lookup(loader, ScalarConsumer("other"))


def test_single_point_scan_accepts_zero_value_and_reuses_existing_destination():
    # The output register already belongs to the caller, so the table must not
    # allocate it again. A single point has the valid index range [0, 0].
    scan = ScanWith("scan", [0], "output", use_existed=True).add_content(
        ScalarConsumer("output")
    )
    prog = compile_scan(scan, reps=2)
    shots = analyze(prog).shots
    assert [
        [(event.cycle - shot.start_cycle, event.value) for event in shot.events]
        for shot in shots
    ] == [[(7, 0)], [(7, 0)]]
    assert [
        (shot.body_end_cycle - shot.start_cycle, shot.end_cycle - shot.start_cycle)
        for shot in shots
    ] == [(7, 14), (7, 14)]
    (index,) = prog.fixed_loop_indices
    assert (index.count, index.bounds.minimum, index.bounds.maximum) == (1, 0, 0)
