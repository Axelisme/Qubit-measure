from collections.abc import Callable

import pytest
from zcu_tools.program.v2 import TimelineProgramV2
from zcu_tools.program.v2.ir.instructions import RegWriteInst
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    Register,
    SrcKeyword,
)
from zcu_tools.program.v2.modules.dmem import LoadValue, LoadWord

from tests.program._profiles import load_soccfg
from tests.program._timeline import compile_modules


@pytest.mark.parametrize("invalid_call", [LoadValue], ids=["LoadValue"])
def test_load_value_rejects_positional_storage_options(
    invalid_call: Callable[..., object],
) -> None:
    with pytest.raises(TypeError, match="positional"):
        invalid_call("lookup", [1, 2], "idx", "value", False, True)


def _execute_lookup_alu(expr: AluExpr, registers: dict[str, int]) -> int:
    lhs = registers[expr.lhs.name]
    rhs = expr.rhs
    if expr.op == AluOp.NONE:
        return lhs
    assert isinstance(rhs, (Register, Immediate))
    operand = registers[rhs.name] if isinstance(rhs, Register) else rhs.value
    match expr.op:
        case AluOp.ADD:
            return lhs + operand
        case AluOp.AND:
            return lhs & operand
        case AluOp.SL:
            return lhs << (operand & 15)
        case AluOp.SR:
            return (lhs & 0xFFFFFFFF) >> (operand & 15)
        case AluOp.ASR:
            signed = lhs if lhs < 2**31 else lhs - 2**32
            return signed >> (operand & 15)
        case _:
            pytest.fail(f"Unsupported lookup ALU operation: {expr}")


def _execute_lookup_body(program: TimelineProgramV2, index: int) -> int:
    """Replay straight-line lookup ASM up to WAIT with tProc's 4-bit shifts.

    This deliberately narrow oracle executes the compiled register/data-memory
    instructions, not LoadValue's packing metadata or its Python simulation.
    The outer sweep's index is supplied by the test; control flow is not emulated.
    """
    index_reg = f"r{program.reg_dict['idx'].addr}"
    output_reg = f"r{program.reg_dict['value'].addr}"
    ignored = {index_reg}
    if "unused" in program.reg_dict:
        ignored.add(f"r{program.reg_dict['unused'].addr}")
    registers = {index_reg: index}
    dmem = program.compile_datamem()
    assert dmem is not None
    for raw in program.prog_list:
        if raw["CMD"] == "WAIT":
            break
        if raw["CMD"] != "REG_WR":
            continue
        inst = RegWriteInst.from_dict(raw)
        dst = inst.dst.name
        if dst in ignored or inst.src == SrcKeyword.LABEL:
            continue
        if inst.src == SrcKeyword.IMM:
            assert inst.lit is not None
            result = inst.lit.value
        elif inst.src == SrcKeyword.DMEM:
            assert isinstance(inst.addr, Register)
            result = int(dmem[registers[inst.addr.name]])
        else:
            assert inst.src == SrcKeyword.OP and inst.op is not None
            result = _execute_lookup_alu(inst.op, registers)
        registers[dst] = result & 0xFFFFFFFF
    return registers[output_reg]


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 8, 9, 16])
@pytest.mark.parametrize("padding", [False, True])
def test_compiled_lookup_recovers_every_slot_with_hardware_shift_width(
    bits: int, padding: bool
) -> None:
    import numpy as np

    rng = np.random.default_rng(1906)
    values = rng.integers(0, 1 << bits, size=131).tolist()
    values[0] = (1 << bits) - 1
    lookup = LoadValue("lookup", values, idx_reg="idx", val_reg="value")
    modules = [LoadValue("padding", [31, 17, 9], "idx", "unused")] if padding else []
    modules.append(lookup)
    program = compile_modules(
        load_soccfg(),
        modules,
        sweep=[("idx", len(values))],
        reps=1,
    )
    assert lookup.is_compressed
    assert [_execute_lookup_body(program, i) for i in range(len(values))] == values


def test_negative_values_rejected():
    with pytest.raises(ValueError, match=r"\[0,"):
        LoadValue("x", [1, -1, 2], idx_reg="i", val_reg="v")


def test_value_exceeding_int32_max_rejected():
    with pytest.raises(ValueError, match=r"\[0,"):
        LoadValue("x", [0, 2**31], idx_reg="i", val_reg="v")


def test_int32_max_value_accepted():
    lv = LoadValue("x", [2**31 - 1], idx_reg="i", val_reg="v")
    assert lv.values == [2**31 - 1]


def test_auto_compress_off_keeps_values():
    values = [i % 4 for i in range(64)]
    lv = LoadValue("x", values, idx_reg="i", val_reg="v", auto_compress=False)
    assert lv.is_compressed is False


# ---------------------------------------------------------------------------
# LoadValue.run — uncompressed path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("padding", [False, True])
def test_uncompressed_scalar_lookup_reaches_legacy_register_consumer(padding):
    values = [10, 20, 30]
    modules = [LoadValue("padding", [31, 17, 9], "idx", "unused")] if padding else []
    modules.append(LoadValue("lookup", values, "idx", "value"))
    program = compile_modules(
        load_soccfg(), modules, sweep=[("idx", len(values))], reps=1
    )
    assert [_execute_lookup_body(program, index) for index in range(3)] == values


# ---------------------------------------------------------------------------
# LoadValue.run — compressed path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Table bounds
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _plan_compression — edge cases
# ---------------------------------------------------------------------------


def test_large_value_max_17bits_not_compressed():
    # max value = 65536 (17-bit) → bits rounds to 32 → values_per_word = 1 < 2 → no compression
    values = [65536] * 64
    lv = LoadValue("x", values, idx_reg="i", val_reg="v")
    assert lv.is_compressed is False


def test_large_value_max_16bits_gets_compressed():
    # max value = 65535 (16-bit) → bits = 16 → values_per_word = 2 → compressed
    values = [65535] * 64
    lv = LoadValue("x", values, idx_reg="i", val_reg="v")
    assert lv.is_compressed is True


# ---------------------------------------------------------------------------
# LoadWord
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-1, 2**32])
def test_load_word_rejects_values_outside_uint32_range(value):
    with pytest.raises(ValueError, match="raw uint32"):
        LoadWord("w", [value], idx_reg="i", val_reg="v")
