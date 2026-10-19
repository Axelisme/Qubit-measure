"""Shared instruction dispatch, strict parsing and immutability contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from zcu_tools.program.v2.ir.instructions import (
    BaseInst,
    DivInst,
    FlagInst,
    JumpInst,
    LabelInst,
    RegWriteInst,
    TestInst,
    TimeInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.io_instructions import (
    DportWriteInst,
    PortWriteInst,
)
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    ImmValue,
    Register,
    SrcKeyword,
    parse_register,
)


@pytest.mark.parametrize(
    "inst, field, value",
    [
        (TimeInst(c_op="inc_ref"), "c_op", "inc_ref"),
        (
            TestInst(op=AluExpr(Register("r1"), AluOp.SUB, Register("r2"))),
            "op",
            AluExpr(Register("r1"), AluOp.ADD, Register("r2")),
        ),
        (JumpInst(label=LabelRef(Label("loop"))), "label", Label("exit")),
        (RegWriteInst(dst=Register("s1"), src=SrcKeyword.IMM), "dst", "s2"),
        (PortWriteInst(dst=ImmValue(0)), "dst", "1"),
    ],
    ids=["time", "test", "jump", "regwr", "wport_wr"],
)
def test_instruction_fields_are_immutable(inst: BaseInst, field: str, value: object):
    with pytest.raises(FrozenInstanceError, match=f"cannot assign to field '{field}'"):
        setattr(inst, field, value)


class TestUnknownOpcode:
    def test_dispatch_unknown_opcode_raises(self):
        with pytest.raises(ValueError, match="Unknown instruction opcode"):
            BaseInst.from_dict({"CMD": "UNKNOWN_OP", "FIELD1": "value1"})

    def test_dispatch_dport_wr_to_specific(self):
        """DPORT_WR is specially handled; should be DportWriteInst."""
        d = {"CMD": "DPORT_WR", "DST": "0", "SRC": "imm", "DATA": "1"}
        inst = BaseInst.from_dict(d)
        assert isinstance(inst, DportWriteInst)
        assert str(inst.dst) == "0"
        assert str(inst.data) == "1"


class TestInstructionParsing:
    def test_flag_accepts_valid_operation(self):
        assert BaseInst.from_dict({"CMD": "FLAG", "C_OP": "set"}) == FlagInst(
            c_op="set"
        )

    def test_flag_rejects_invalid_operation(self):
        with pytest.raises(ValueError, match="FLAG.C_OP"):
            BaseInst.from_dict({"CMD": "FLAG", "C_OP": "bad"})

    @pytest.mark.parametrize("condition", [None, "NZ"])
    def test_jump_accepts_absent_and_valid_condition(self, condition):
        assert JumpInst.from_dict({"IF": condition}).if_cond == condition

    def test_jump_rejects_invalid_condition(self):
        with pytest.raises(ValueError, match="valid condition code"):
            JumpInst.from_dict({"IF": "bad"})

    def test_regwrite_rejects_invalid_register(self):
        with pytest.raises(ValueError, match="DST"):
            RegWriteInst.from_dict({"DST": "garbage", "SRC": "imm"})

    def test_dispatch_rejects_invalid_test_alu_expression(self):
        with pytest.raises(ValueError, match="Cannot parse ALU expression: 'garbage'"):
            BaseInst.from_dict({"CMD": "TEST", "OP": "garbage"})

    def test_port_write_rejects_invalid_port(self):
        with pytest.raises(ValueError, match="port number"):
            PortWriteInst.from_dict({"DST": "bogus"})

    def test_port_write_rejects_invalid_address(self):
        with pytest.raises(ValueError, match="ADDR"):
            PortWriteInst.from_dict({"DST": "0", "ADDR": "bogus"})

    def test_instruction_str_omits_none_fields(self):
        assert str(TimeInst(c_op="rst")) == "TimeInst(C_OP=rst)"


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_missing_cmd_raises_error(self):
        with pytest.raises(ValueError, match="Unknown instruction format"):
            BaseInst.from_dict({"FIELD": "value"})

    def test_empty_cmd_raises_error(self):
        with pytest.raises(ValueError, match="Unknown instruction format"):
            BaseInst.from_dict({"CMD": ""})

    def test_label_dict_without_kind_raises_error(self):
        """Strict IR: only kind='label' is supported."""
        d = {"LABEL": "test_label"}
        with pytest.raises(ValueError, match="Invalid LabelInst format"):
            LabelInst.from_dict(d)

    def test_mixed_optional_fields(self):
        """Instructions with selective optional fields."""
        d = {"CMD": "TIME", "C_OP": "inc_ref"}
        inst = BaseInst.from_dict(d)
        assert isinstance(inst, TimeInst)
        assert inst.lit is None
        assert inst.r1 is None


class TestStrictParsing:
    """Strict-parsing guarantees: bad input fails at the IR boundary (IR-2/3/4)."""

    def test_div_from_dict_accepts_register_and_immediate_den(self):
        reg_den = BaseInst.from_dict({"CMD": "DIV", "NUM": "r0", "DEN": "r1"})
        assert isinstance(reg_den, DivInst)
        assert reg_den.den == Register("r1")

        imm_den = BaseInst.from_dict({"CMD": "DIV", "NUM": "r0", "DEN": "#5"})
        assert isinstance(imm_den, DivInst)
        assert imm_den.den == Immediate(5)

    def test_div_from_dict_rejects_invalid_den(self):
        # IR-2: a DEN that is neither a register nor an immediate must raise,
        # not be silently wrapped into a bogus Register.
        with pytest.raises(ValueError, match="DIV.DEN"):
            BaseInst.from_dict({"CMD": "DIV", "NUM": "r0", "DEN": "garbage"})

    def test_parse_register_rejects_unknown_descriptive_name(self):
        # IR-3: a descriptive name not in the alias table is not a register.
        assert parse_register("s_typo") is None
        assert parse_register("w_bogus") is None

    def test_parse_register_accepts_known_aliases_and_bare_names(self):
        assert parse_register("s_out_time") == Register("s_out_time")
        assert parse_register("w_freq") == Register("w_freq")
        assert parse_register("r_wave") == Register("r_wave")
        assert parse_register("r5") == Register("r5")
        assert parse_register("s14") == Register("s14")
        assert parse_register("&r1") == Register("r1")

    def test_wait_inst_rejects_non_s15_addr(self):
        # IR-4: WAIT ADDR may only be s15 (assembler enforces the same).
        with pytest.raises(ValueError, match="WaitInst.addr"):
            WaitInst(c_op="time", addr=Register("r0"))

    def test_wait_inst_accepts_s15_addr(self):
        inst = WaitInst(c_op="time", addr=Register("s15"))
        assert inst.addr == Register("s15")

    def test_time_inc_ref_reg_read_includes_s14(self):
        assert TimeInst(c_op="inc_ref").reg_read == frozenset({"s14"})

    def test_from_dict_rejects_unknown_time_string(self):
        with pytest.raises(ValueError, match="WAIT.TIME"):
            BaseInst.from_dict({"CMD": "WAIT", "C_OP": "time", "TIME": "garbage"})

    def test_dport_write_data_rejects_garbage_string(self):
        with pytest.raises(ValueError, match="DPORT_WR.DATA"):
            BaseInst.from_dict(
                {"CMD": "DPORT_WR", "DST": "0", "SRC": "imm", "DATA": "garbage"}
            )

    def test_dport_write_data_rejects_unknown_register_name(self):
        with pytest.raises(ValueError, match="DPORT_WR.DATA"):
            BaseInst.from_dict(
                {"CMD": "DPORT_WR", "DST": "0", "SRC": "reg", "DATA": "w_bogus"}
            )

    def test_dport_write_requires_src(self):
        with pytest.raises(ValueError, match="DPORT_WR.SRC"):
            BaseInst.from_dict({"CMD": "DPORT_WR", "DST": "0", "DATA": "0"})

    def test_dport_write_rejects_invalid_src_keyword(self):
        with pytest.raises(ValueError, match="DPORT_WR.SRC"):
            BaseInst.from_dict(
                {"CMD": "DPORT_WR", "DST": "0", "SRC": "wmem", "DATA": "0"}
            )

    def test_dport_write_rejects_register_src_token(self):
        with pytest.raises(ValueError, match="DPORT_WR.SRC"):
            BaseInst.from_dict(
                {"CMD": "DPORT_WR", "DST": "0", "SRC": "r0", "DATA": "0"}
            )

    @pytest.mark.parametrize(("src", "data"), [("imm", "0"), ("reg", "r0")])
    def test_dport_write_accepts_valid_src_keywords(self, src: str, data: str):
        inst = BaseInst.from_dict(
            {"CMD": "DPORT_WR", "DST": "0", "SRC": src, "DATA": data}
        )
        assert isinstance(inst, DportWriteInst)
        assert inst.to_dict()["SRC"] == src
        assert inst.to_dict()["DATA"] == data

    @pytest.mark.parametrize("data", ["0", "0x10"])
    def test_dport_write_imm_data_accepts_bare_ints(self, data: str):
        inst = BaseInst.from_dict(
            {"CMD": "DPORT_WR", "DST": "0", "SRC": "imm", "DATA": data}
        )
        assert isinstance(inst, DportWriteInst)
        assert isinstance(inst.data, ImmValue)

    @pytest.mark.parametrize("data", ["r0", "w0"])
    def test_dport_write_reg_data_accepts_registers(self, data: str):
        inst = BaseInst.from_dict(
            {"CMD": "DPORT_WR", "DST": "0", "SRC": "reg", "DATA": data}
        )
        assert isinstance(inst, DportWriteInst)
        assert inst.to_dict()["DATA"] == data

    def test_dport_write_imm_data_rejects_prefixed_immediate(self):
        with pytest.raises(ValueError, match="DPORT_WR.DATA"):
            BaseInst.from_dict(
                {"CMD": "DPORT_WR", "DST": "0", "SRC": "imm", "DATA": "#1"}
            )

    def test_dport_write_reg_data_rejects_bare_int(self):
        with pytest.raises(ValueError, match="DPORT_WR.DATA"):
            BaseInst.from_dict(
                {"CMD": "DPORT_WR", "DST": "0", "SRC": "reg", "DATA": "1"}
            )
