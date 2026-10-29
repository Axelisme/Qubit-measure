"""Register arithmetic observed through compiled numeric pulse outputs."""

import numpy as np
import pytest

from tests.program.v2.simulation._compiled_support import program, request


@pytest.mark.parametrize(
    "operation,a,b,expected",
    [
        ("MSH", -1985229329, 0, 0x89AB),
        ("LSH", -1985229329, 0, 0xCDEF),
        ("SWP", -1985229329, 0, -839939669),
        ("PAR", -1, 0, 0),
        ("PAR", 0x10001, 0, 0),
        ("PAR", -2, 0, 1),
        ("CAT", 0x12345678, -1985229329, 0x5678CDEF),
        ("::", -1, 0x12345678, -43400),
        ("ABS", -2147483648, 0, -2147483648),
        ("NOT", 0x12345678, 0, -305419897),
        ("+", 2147483647, 1, -2147483648),
        ("-", -2147483648, 1, 2147483647),
        ("AND", -1, 0x12345678, 0x12345678),
        ("OR", 0x10000000, 0x2345678, 0x12345678),
        ("XOR", -1, 0x12345678, -305419897),
        ("ASR", -2147483648, 31, -65536),
        ("SR", -2147483648, 31, 65536),
        ("SL", 0x12345678, 16, 0x12345678),
        (">>", -1, 16, -1),
        ("<<", -2147483648, 1, 0),
    ],
)
def test_alu_bitvector_results_reach_output_and_latch_only_zero_sign(
    operation, a, b, expected
):
    def initialize(p):
        p.add_reg("a")
        p.add_reg("b")

    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("a", a)
        p.write_reg("b", b)
        ra, rb = (p.reg_dict[name].full_addr() for name in ("a", "b"))
        expression = (
            f"{operation} {ra}"
            if operation in {"MSH", "LSH", "SWP", "PAR", "ABS", "NOT"}
            else f"{ra} {operation} {rb}"
        )
        p.asm_inst(
            {"CMD": "REG_WR", "DST": "w3", "SRC": "op", "OP": expression, "UF": "1"}
        )
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})
        p.write_reg("w_gain", 10)
        p.asm_inst(
            {"CMD": "REG_WR", "DST": "w3", "SRC": "imm", "LIT": "#20", "IF": "Z"}
        )
        p.asm_inst(
            {"CMD": "REG_WR", "DST": "w3", "SRC": "imm", "LIT": "#30", "IF": "S"}
        )
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@50"})

    prog, env = program(body, initialize, reps=1)
    events = request(prog, env).materialize_point(0).events
    flag_gain = 20 if expected == 0 else 30 if expected < 0 else 10
    assert [e.gain for e in events] == pytest.approx(
        np.array([expected, flag_gain]) / 32766
    )


@pytest.mark.parametrize(
    "opcode,operands,expected",
    [
        ("T", [-3, 7], -21),
        ("TP", [-3, 7, -5], -26),
        ("TM", [-3, 7, -5], -16),
        ("PT", [10, -3, 7], 49),
        ("PTP", [10, -3, 7, -5], 44),
        ("PTM", [10, -3, 7, -5], 54),
        ("MT", [10, -3, 7], 91),
        ("MTP", [10, -3, 7, -5], 86),
        ("MTM", [10, -3, 7, -5], 96),
        ("TP", [134217725, 262151, -5], -26),
        ("TM", [-67108864, -131072, -2147483648], -2147483648),
        ("T", [65536, 65536], 0),
    ],
)
def test_arith_signed_lane_widths_and_low_word_reach_pulse_gain(
    opcode, operands, expected
):
    def initialize(p):
        for i in range(len(operands)):
            p.add_reg(f"operand{i}")

    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        instruction = {"CMD": "ARITH", "C_OP": opcode}
        for i, value in enumerate(operands):
            p.write_reg(f"operand{i}", value)
            instruction[f"R{i + 1}"] = p.reg_dict[f"operand{i}"].full_addr()
        p.asm_inst(instruction)
        p.write_reg("w_gain", "s_arith_l")
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})

    prog, env = program(body, initialize, reps=1)
    assert request(prog, env).materialize_point(0).events[0].gain == pytest.approx(
        expected / 32766
    )


@pytest.mark.parametrize("opcode,d,a", [("PT", 67108863, 1), ("MT", -67108864, 1)])
def test_arith_pre_adder_overflow_fails_before_request(opcode, d, a):
    def initialize(p):
        for name in ("d", "a", "b"):
            p.add_reg(name)

    def body(p):
        for name, value in (("d", d), ("a", a), ("b", 1)):
            p.write_reg(name, value)
        p.asm_inst(
            {
                "CMD": "ARITH",
                "C_OP": opcode,
                **{
                    f"R{i + 1}": p.reg_dict[n].full_addr()
                    for i, n in enumerate(("d", "a", "b"))
                },
            }
        )
        p.pulse(0, "p")

    prog, env = program(body, initialize, reps=1)
    with pytest.raises(ValueError, match="ArithInst.*preopt.*reps.*pre-adder.*27"):
        request(prog, env)


@pytest.mark.parametrize(
    "numerator,denominator,immediate,expected",
    [
        (17, 5, True, [3, 2]),
        (-1, 2, False, [2147483647, 1]),
        (5, -1, False, [0, 5]),
        (-1, 1, True, [-1, 0]),
    ],
)
def test_div_uint32_quotient_and_remainder_reach_outputs(
    numerator, denominator, immediate, expected
):
    def initialize(p):
        p.add_reg("num")
        p.add_reg("den")

    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.write_reg("num", numerator)
        p.write_reg("den", denominator)
        p.asm_inst(
            {
                "CMD": "DIV",
                "NUM": p.reg_dict["num"].full_addr(),
                "DEN": f"#{denominator}"
                if immediate
                else p.reg_dict["den"].full_addr(),
            }
        )
        for time, name in ((0, "s_div_q"), (50, "s_div_r")):
            p.write_reg("w_gain", name)
            p.asm_inst(
                {"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": f"@{time}"}
            )

    prog, env = program(body, initialize, reps=1)
    assert [
        e.gain for e in request(prog, env).materialize_point(0).events
    ] == pytest.approx(np.array(expected) / 32766)


@pytest.mark.parametrize(
    "destination,value,expected",
    [
        ("s0", 1234, 0),
        ("s3", 1234, 0),
        ("s4", 1234, 0),
        ("s15", 305419896, 22136),
        ("s2", 376, 120),
    ],
)
def test_register_destinations_preserve_readonly_and_sfr_width_rules(
    destination, value, expected
):
    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        p.asm_inst({"CMD": "ARITH", "C_OP": "T", "R1": "s0", "R2": "s0"})
        p.asm_inst({"CMD": "DIV", "NUM": "s0", "DEN": "#1"})
        p.write_reg(destination, value)
        p.write_reg("w_gain", destination)
        p.asm_inst({"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": "@0"})

    prog, env = program(body, reps=1)
    assert request(prog, env).materialize_point(0).events[0].gain == pytest.approx(
        expected / 32766
    )


def test_div_zero_fails_before_request_with_instruction_context():
    def body(p):
        p.asm_inst({"CMD": "DIV", "NUM": "s0", "DEN": "#0"})
        p.pulse(0, "p")

    prog, env = program(body, reps=1)
    with pytest.raises(ValueError, match="DivInst.*preopt.*reps.*zero"):
        request(prog, env)


@pytest.mark.parametrize(
    "opcode,operands,low,high",
    [
        ("TM", [-67108864, -131072, -2147483648], -2147483648, 2048),
        ("TP", [67108863, 131071, 2147483647], 2080243712, 2048),
        ("PTP", [-67108864, 0, 131071, -2147483648], -2080374784, -2049),
        ("MTM", [0, -67108863, -131072, 2147483647], -2147352575, -2049),
    ],
)
def test_arith_extreme_signed_values_are_available_through_configured_core_words(
    opcode, operands, low, high
):
    def initialize(p):
        for i in range(len(operands)):
            p.add_reg(f"operand{i}")

    def body(p):
        p.read_wmem(p.pulses["p"].get_wavenames()[0])
        instruction = {"CMD": "ARITH", "C_OP": opcode}
        for i, value in enumerate(operands):
            p.write_reg(f"operand{i}", value)
            instruction[f"R{i + 1}"] = p.reg_dict[f"operand{i}"].full_addr()
        p.asm_inst(instruction)
        p.write_reg("s_cfg", 1)
        for time, name in ((0, "s_core_r1"), (50, "s_core_r2")):
            p.write_reg("w_gain", name)
            p.asm_inst(
                {"CMD": "WPORT_WR", "DST": "0", "SRC": "r_wave", "TIME": f"@{time}"}
            )

    prog, env = program(body, initialize, reps=1)
    assert [
        e.gain for e in request(prog, env).materialize_point(0).events
    ] == pytest.approx(np.array([low, high]) / 32766)
