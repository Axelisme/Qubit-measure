"""Known data-memory timing/control through real native compilation."""

from tests.program._profiles import load_soccfg
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from qick.asm_v2 import AsmInst
from zcu_tools.program.v2 import ir
from zcu_tools.program.v2.macro.loop import CloseInnerLoop, OpenInnerLoop

from zcu_tools.program.v2.timeline import TimelineProgramV2


class DispatchProgram(TimelineProgramV2):
    count = 3

    def _initialize(self, cfg):
        self.declare_gen(ch=0, nqz=1)
        self.add_pulse(
            ch=0,
            name="p",
            style="const",
            freq=100,
            phase=0,
            gain=0.1,
            length=5 / 384,
        )
        self.add_reg("count")
        self.add_reg("counter")
        self.write_reg("count", self.count)

    def _body(self, cfg):
        self.append_macro(OpenInnerLoop("inner", "counter", "count", range_hint=(2, 4)))
        self.pulse(ch=0, name="p", t=0)
        self.delay(15 / 430.08)
        self.append_macro(CloseInnerLoop("inner", "counter", "count"))


def test_native_compiler_dispatch_uses_public_dmem_without_mutating_it():
    prog = DispatchProgram(
        load_soccfg(),
        reps=2,
        initial_delay=20 / 430.08,
        relax_delay=7 / 430.08,
    )
    layout = prog.timing_layout
    dmem = prog.compile_datamem()
    before = deepcopy((prog.prog_list, prog.labels, prog.meta_infos, dmem))
    result = ir.analyze_timing(
        ir.CompiledTimingInput(prog.prog_list, prog.labels, layout, dmem)
    )
    # Three 15-cycle iterations; pulse duration 5.6 fits inside each iteration.
    # Initial ref20, body45, shell7; repeat starts72, ends124.
    assert [(s.start_cycle, s.body_end_cycle, s.end_cycle) for s in result.shots] == [
        (20, 65, 72),
        (72, 117, 124),
    ]
    assert [[e.cycle for e in s.events] for s in result.shots] == [
        [20, 35, 50],
        [72, 87, 102],
    ]
    assert (prog.prog_list, prog.labels, prog.meta_infos) == before[:3]
    np.testing.assert_array_equal(dmem, before[3])


@pytest.mark.parametrize(
    "count,cycles,end", [(2, [0, 15], 30), (4, [0, 15, 30, 45], 60)]
)
def test_native_dispatch_range_edges_and_transported_words(count, cycles, end):
    class Program(DispatchProgram):
        pass

    Program.count = count
    prog = Program(load_soccfg(), initial_delay=0, relax_delay=0)
    result = ir.analyze_timing(
        ir.CompiledTimingInput(
            prog.prog_list, prog.labels, prog.timing_layout, prog.binprog["dmem"]
        )
    )
    assert [e.cycle for e in result.shots[0].events] == cycles
    assert (result.shots[0].body_end_cycle, result.shots[0].end_cycle) == (end, end)


def compile_memory(
    instructions: Sequence[dict[str, str]],
    words: Sequence[int] = (7,),
) -> ir.CompiledTimingInput:
    class Program(TimelineProgramV2):
        def _initialize(self, cfg):
            self.add_dmem(words)
            self.add_reg("data", addr=10)

        def _body(self, cfg):
            for instruction in instructions:
                self.append_macro(AsmInst(inst=instruction, addr_inc=1))

    prog = Program(load_soccfg(), initial_delay=0, relax_delay=0)
    return ir.CompiledTimingInput(
        prog.prog_list,
        prog.labels,
        prog.timing_layout,
        prog.compile_datamem(),
    )


def read_and_output(address: str) -> list[dict[str, str]]:
    return [
        {"CMD": "REG_WR", "DST": "r10", "SRC": "dmem", "ADDR": address},
        {"CMD": "DPORT_WR", "DST": "0", "SRC": "reg", "DATA": "r10", "TIME": "@3"},
    ]


@pytest.mark.parametrize("address", ["&0", "&2", "s15"])
@pytest.mark.parametrize("word", [0, -7, -(2**31), 2**31 - 1])
def test_dmem_word_offsets_and_signed_words_reach_public_output(address, word):
    # Distinct words catch byte addressing or accidentally reading the first word.
    instructions = read_and_output(address)
    if address == "s15":
        instructions.insert(
            0, {"CMD": "REG_WR", "DST": "s15", "SRC": "imm", "LIT": "#2"}
        )
    compiled = compile_memory(instructions, [11, 99, word])
    result = ir.analyze_timing(compiled)
    assert [(e.kind, e.port, e.cycle, e.value) for e in result.shots[0].events] == [
        ("data", 0, 3, 11 if address == "&0" else word),
    ]


@pytest.mark.parametrize("dmem", [None, []])
def test_executed_read_requires_actual_compiled_data(dmem):
    compiled = compile_memory(read_and_output("&0"))
    with pytest.raises(
        ValueError, match="Missing compiled dmem|Dmem word address out of range"
    ):
        ir.analyze_timing(replace(compiled, dmem=dmem))


@pytest.mark.parametrize("address", [-1, 1])
def test_dmem_address_does_not_wrap_or_read_zero_outside_the_buffer(address):
    compiled = compile_memory(
        [
            {"CMD": "REG_WR", "DST": "s15", "SRC": "imm", "LIT": f"#{address}"},
            *read_and_output("s15"),
        ]
    )
    with pytest.raises(ValueError, match="Dmem word address out of range"):
        ir.analyze_timing(compiled)


@pytest.mark.parametrize(
    "address,message", [("s15", "Unknown register value: s15"), ("s11", "s11")]
)
def test_dmem_address_must_be_known_without_hardware_time(address, message):
    with pytest.raises(ValueError, match=message):
        ir.analyze_timing(compile_memory(read_and_output(address)))


def test_dmem_writes_are_not_modeled_as_read_only_data():
    compiled = compile_memory(
        [
            {"CMD": "DMEM_WR", "DST": "[&0]", "SRC": "imm", "LIT": "#12"},
            *read_and_output("&0"),
        ]
    )
    with pytest.raises(
        ValueError, match="Unsupported timing instruction: DmemWriteInst"
    ):
        ir.analyze_timing(compiled)


@pytest.mark.parametrize(
    "dmem",
    [
        np.array([[7]], dtype=np.int32),
        np.array([7], dtype=np.int64),
        [2**31],
        [-(2**31) - 1],
    ],
)
def test_read_rejects_data_outside_the_compiled_int32_word_contract(dmem):
    compiled = compile_memory(read_and_output("&0"))
    with pytest.raises(ValueError, match="int32"):
        ir.analyze_timing(replace(compiled, dmem=dmem))


def test_known_false_read_needs_no_data_and_preserves_the_destination():
    instructions = [
        {"CMD": "REG_WR", "DST": "r10", "SRC": "imm", "LIT": "#9"},
        {"CMD": "TEST", "OP": "s0", "UF": "1"},
        {"CMD": "REG_WR", "DST": "r10", "SRC": "dmem", "ADDR": "&0", "IF": "NZ"},
        {"CMD": "DPORT_WR", "DST": "0", "SRC": "reg", "DATA": "r10", "TIME": "@3"},
    ]
    result = ir.analyze_timing(compile_memory(instructions, []))
    assert [(e.cycle, e.value) for e in result.shots[0].events] == [(3, 9)]


def test_unknown_read_condition_is_not_assumed_false():
    instructions = read_and_output("&0")
    instructions[0]["IF"] = "NZ"
    with pytest.raises(ValueError, match="Unknown condition flags"):
        ir.analyze_timing(compile_memory(instructions))


def test_dmem_read_with_alu_flag_side_effects_is_rejected():
    instructions = read_and_output("&0")
    instructions[0].update(OP="s0", UF="1")
    with pytest.raises(ValueError, match="Unsupported dmem read side effects"):
        ir.analyze_timing(compile_memory(instructions))


@pytest.mark.parametrize("words", [[], [7]])
def test_programs_without_reads_do_not_require_data(words):
    compiled = compile_memory([{"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#5"}], words)
    result = ir.analyze_timing(replace(compiled, dmem=None))
    assert (result.shots[0].body_end_cycle, result.shots[0].end_cycle) == (5, 5)
