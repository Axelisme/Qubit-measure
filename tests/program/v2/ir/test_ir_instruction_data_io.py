"""Register, memory and I/O instruction construction and serialization contracts."""

from __future__ import annotations

import pytest
from zcu_tools.program.v2.ir.instructions import (
    ArithInst,
    BaseInst,
    ClearInst,
    ComInst,
    CustomPeripheralInst,
    FlagInst,
    NetInst,
    RegWriteInst,
)
from zcu_tools.program.v2.ir.io_instructions import (
    DmemReadInst,
    DmemWriteInst,
    DportReadInst,
    PortWriteInst,
    TrigInst,
    WmemWriteInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.operands import (
    AluExpr,
    AluOp,
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    SideWrite,
    SrcKeyword,
)


class TestRegWriteInstruction:
    """Tests for RegWriteInst (REG_WR opcode)."""

    def test_construction_imm_source(self):
        inst = RegWriteInst(dst=Register("s1"), src=SrcKeyword.IMM, lit=Immediate(10))
        assert inst.dst.name == "s1"
        assert inst.src == "imm"
        assert str(inst.lit) == "#10"

    def test_construction_op_source(self):
        inst = RegWriteInst(
            dst=Register("s2"),
            src=SrcKeyword.OP,
            op=AluExpr(Register("s1"), AluOp.ADD, Immediate(1)),
            uf=False,
        )
        assert inst.dst.name == "s2"
        assert inst.src == "op"
        assert inst.op is not None
        assert isinstance(inst.op.lhs, Register)
        assert inst.op.lhs.name == "s1"
        assert inst.uf is False

    def test_dispatch_regwr_imm(self):
        d = {"CMD": "REG_WR", "DST": "s1", "SRC": "imm", "LIT": "#0"}
        inst = BaseInst.from_dict(d)
        assert isinstance(inst, RegWriteInst)
        assert inst.dst.name == "s1"
        assert inst.src == "imm"
        assert str(inst.lit) == "#0"

    def test_dispatch_regwr_op(self):
        d = {"CMD": "REG_WR", "DST": "s2", "SRC": "op", "OP": "s1 + #1", "UF": "1"}
        inst = BaseInst.from_dict(d)
        assert isinstance(inst, RegWriteInst)
        assert inst.src == "op"
        assert inst.op is not None
        assert inst.op.op == AluOp.ADD

    def test_dispatch_regwr_dmem_lowering(self):
        """REG_WR src=dmem is recognized as DmemReadInst."""
        d = {"CMD": "REG_WR", "DST": "r0", "SRC": "dmem", "ADDR": "&123"}
        Label("123")
        inst = BaseInst.from_dict(d)
        assert isinstance(inst, DmemReadInst)
        assert inst.src == "dmem"
        assert inst.dst.name == "r0"

    def test_dispatch_legacy_dmem_rd_raises(self):
        """DMEM_RD opcode is outside the supported tProc v2 instruction set."""
        d = {"CMD": "DMEM_RD", "DST": "r1", "ADDR": "s15"}
        with pytest.raises(ValueError, match="Unknown instruction opcode"):
            BaseInst.from_dict(d)

    def test_roundtrip_regwr(self):
        original = {"CMD": "REG_WR", "DST": "s1", "SRC": "imm", "LIT": "#100"}
        inst = BaseInst.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_regwr_wmem_preserves_wp_and_port(self):
        original = {
            "CMD": "REG_WR",
            "DST": "r_wave",
            "SRC": "wmem",
            "ADDR": "&7",
            "WP": "r_wave",
            "PORT": "3",
        }
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, RegWriteInst)
        assert inst.wp == "r_wave"
        assert inst.port == "3"
        assert inst.to_dict() == original

    def test_regwr_rejects_packed_wp_port_shape(self):
        with pytest.raises(ValueError, match="REG_WR.WP"):
            BaseInst.from_dict(
                {
                    "CMD": "REG_WR",
                    "DST": "r_wave",
                    "SRC": "wmem",
                    "ADDR": "&7",
                    "WP": "r_wave p3",
                }
            )

    def test_regwr_rejects_invalid_src_token(self):
        with pytest.raises(ValueError, match="REG_WR.SRC"):
            BaseInst.from_dict({"CMD": "REG_WR", "DST": "r0", "SRC": "garbage"})

    def test_regwr_rejects_invalid_immediate_field(self):
        with pytest.raises(ValueError, match="REG_WR.LIT"):
            BaseInst.from_dict(
                {"CMD": "REG_WR", "DST": "r0", "SRC": "imm", "LIT": "#bad"}
            )


class TestPortWriteInstruction:
    """Tests for PortWriteInst (WPORT_WR opcode)."""

    def test_construction_with_specific_fields(self):
        inst = PortWriteInst(dst=ImmValue(0), time=Register("r0"), ww="1")
        assert str(inst.dst) == "0"
        assert isinstance(inst.time, Register)
        assert inst.time.name == "r0"
        assert inst.ww == "1"

    def test_dispatch_wport_wr(self):
        d = {"CMD": "WPORT_WR", "DST": "1", "TIME": "r1", "WW": "1"}
        inst = BaseInst.from_dict(d)
        assert isinstance(inst, PortWriteInst)
        assert str(inst.dst) == "1"
        assert isinstance(inst.time, Register)
        assert inst.time.name == "r1"
        assert inst.ww == "1"

    def test_roundtrip_wport_wr(self):
        original = {"CMD": "WPORT_WR", "DST": "0", "TIME": "r0", "WW": "1"}
        inst = BaseInst.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_wport_wr_native_shape(self):
        original = {
            "CMD": "WPORT_WR",
            "DST": "0",
            "SRC": "wmem",
            "ADDR": "&12",
            "TIME": "@10",
        }
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, PortWriteInst)
        assert inst.src == "wmem"
        assert str(inst.addr) == "&12"
        recovered = inst.to_dict()
        assert recovered == original

    def test_wport_wr_preserves_specific_fields(self):
        """WPORT_WR can have various data fields that must be preserved."""
        original = {
            "CMD": "WPORT_WR",
            "DST": "2",
            "TIME": "r2",
            "WW": "1",
        }
        inst = BaseInst.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_wport_wr_reg_read_includes_src_addr_time_and_op_registers(self):
        inst = PortWriteInst(
            dst=Register("r0"),
            src=Register("r1"),
            addr=Register("r2"),
            time=Register("r3"),
            op=AluExpr(Register("r4"), AluOp.ADD, Immediate(1)),
        )
        assert inst.reg_read == frozenset({"s14", "r0", "r1", "r2", "r3", "r4"})

    def test_wport_wr_roundtrip_with_conditional_sidewrite_shape(self):
        original = {
            "CMD": "WPORT_WR",
            "DST": "r0",
            "SRC": "r1",
            "ADDR": "r2",
            "TIME": "r3",
            "WR": "r4 op",
            "OP": "r5 + #1",
            "UF": "1",
            "IF": "NZ",
        }
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, PortWriteInst)
        assert inst.to_dict() == original


class TestDmemWriteInstruction:
    def test_dmem_wr_roundtrip_with_bracketed_memaddr(self):
        original = {"CMD": "DMEM_WR", "DST": "[&7]", "SRC": "imm", "LIT": "#3"}
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, DmemWriteInst)
        assert inst.dst == MemAddr(7)
        assert inst.to_dict() == original

    def test_dmem_wr_roundtrip_op_shape(self):
        original = {
            "CMD": "DMEM_WR",
            "DST": "r0",
            "SRC": "op",
            "WR": "r1 op",
            "OP": "r2 + #1",
            "UF": "1",
            "IF": "S",
        }
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, DmemWriteInst)
        assert inst.to_dict() == original

    def test_dmem_wr_reg_read_includes_dst_register_and_op(self):
        inst = DmemWriteInst(
            dst=Register("r0"),
            src="op",
            op=AluExpr(Register("r1"), AluOp.ADD, Immediate(1)),
        )
        assert inst.reg_read == frozenset({"r0", "r1"})

    def test_dmem_wr_rejects_invalid_src_keyword(self):
        with pytest.raises(ValueError, match="DMEM_WR.SRC"):
            BaseInst.from_dict({"CMD": "DMEM_WR", "DST": "[&7]", "SRC": "bogus"})


class TestWmemWriteInstruction:
    """Tests for WMEM_WR."""

    def test_dispatch_wmem_wr(self):
        d = {"CMD": "WMEM_WR", "DST": "&5", "TIME": "@10", "WP": "r_wave", "PORT": "0"}
        inst = BaseInst.from_dict(d)
        assert isinstance(inst, WmemWriteInst)
        assert str(inst.addr) == "&5"
        assert str(inst.time) == "@10"
        assert inst.wp == "r_wave"
        assert inst.port == "0"

    def test_roundtrip_wmem_wr(self):
        original = {
            "CMD": "WMEM_WR",
            "DST": "&7",
            "TIME": "@12",
            "WR": "r_wave op",
            "OP": "s1 + #1",
        }
        inst = BaseInst.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_wmem_wr_reg_read_includes_wave_bundle_addr_time_and_op(self):
        inst = WmemWriteInst(
            addr=Register("r0"),
            time=Register("r1"),
            op=AluExpr(Register("r2"), AluOp.ADD, Immediate(1)),
            wr=SideWrite(Register("r3"), "op"),
            uf=True,
            if_cond="NZ",
            wp="r_wave",
            port="0",
        )
        reads = inst.reg_read
        assert "s14" in reads
        assert {"r0", "r1", "r2", "w0", "w1", "w2", "w3", "w4", "w5"} <= reads

    def test_wmem_wr_roundtrip_full_shape(self):
        original = {
            "CMD": "WMEM_WR",
            "DST": "r0",
            "TIME": "r1",
            "WR": "r2 op",
            "OP": "r3 + #1",
            "UF": "1",
            "IF": "NZ",
            "WP": "r_wave",
            "PORT": "0",
        }
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, WmemWriteInst)
        assert inst.to_dict() == original

    def test_wmem_wr_rejects_packed_wp_port_shape(self):
        with pytest.raises(ValueError, match="WMEM_WR.WP"):
            BaseInst.from_dict({"CMD": "WMEM_WR", "DST": "&5", "WP": "r_wave p0"})

    def test_wmem_wr_rejects_wp_without_port(self):
        with pytest.raises(ValueError, match="WMEM_WR.PORT"):
            BaseInst.from_dict({"CMD": "WMEM_WR", "DST": "&5", "WP": "r_wave"})

    def test_wmem_wr_rejects_invalid_time_field(self):
        with pytest.raises(ValueError, match="WMEM_WR.TIME"):
            BaseInst.from_dict({"CMD": "WMEM_WR", "DST": "&5", "TIME": "@bad"})


class TestDportReadInstruction:
    def test_dport_rd_roundtrip(self):
        original = {"CMD": "DPORT_RD", "DST": "r0"}
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, DportReadInst)
        assert inst.to_dict() == original

    def test_dport_rd_reg_read_and_write(self):
        inst = DportReadInst(dst=Register("r0"))
        assert inst.reg_read == frozenset({"r0", "s10"})
        assert inst.reg_write == frozenset({"s8", "s9"})


class TestTrigInstruction:
    def test_trig_roundtrip(self):
        original = {"CMD": "TRIG", "DST": "r0", "SRC": "set", "TIME": "r1"}
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, TrigInst)
        assert inst.to_dict() == original

    def test_trig_reg_read(self):
        inst = TrigInst(dst=Register("r0"), src="clr", time=Register("r1"))
        assert inst.reg_read == frozenset({"s14", "r0", "r1"})

    def test_trig_rejects_invalid_src(self):
        with pytest.raises(ValueError, match="TRIG.SRC"):
            BaseInst.from_dict({"CMD": "TRIG", "DST": "0", "SRC": "pulse"})


class TestPeripheralInstructions:
    def test_flag_roundtrip_and_rejects_invalid_cop(self):
        original = {"CMD": "FLAG", "C_OP": "set"}
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, FlagInst)
        assert inst.to_dict() == original
        with pytest.raises(ValueError, match="FLAG.C_OP"):
            BaseInst.from_dict({"CMD": "FLAG", "C_OP": "bad"})

    def test_arith_roundtrip_reg_read_and_reg_write(self):
        original = {
            "CMD": "ARITH",
            "C_OP": "TP",
            "R1": "r0",
            "R2": "r1",
            "R3": "r2",
            "R4": "r3",
        }
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, ArithInst)
        assert inst.reg_read == frozenset({"r0", "r1", "r2", "r3"})
        assert inst.reg_write == frozenset({"s3"})
        assert inst.to_dict() == original

    def test_arith_rejects_invalid_cop(self):
        with pytest.raises(ValueError, match="ARITH.C_OP"):
            BaseInst.from_dict({"CMD": "ARITH", "C_OP": "bad"})

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ({"CMD": "ARITH", "C_OP": "T", "R1": "garbage_reg"}, "ARITH.R1"),
            ({"CMD": "NET", "C_OP": "sync_net", "R2": "w6"}, "NET.R2"),
            ({"CMD": "COM", "C_OP": "sync", "R1": "not_a_reg"}, "COM.R1"),
            ({"CMD": "PA", "C_OP": "7", "R3": "r32"}, "PA.R3"),
            ({"CMD": "PB", "C_OP": "7", "R4": None}, "PB.R4"),
        ],
    )
    def test_optional_register_fields_reject_invalid_values(
        self, payload: dict[str, object], message: str
    ):
        with pytest.raises(ValueError, match=message):
            BaseInst.from_dict(payload)

    def test_net_roundtrip_reg_read_and_invalid_cop(self):
        original = {
            "CMD": "NET",
            "C_OP": "set_flag",
            "R1": "r0",
            "R2": "r1",
            "R3": "r2",
        }
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, NetInst)
        assert inst.reg_read == frozenset({"r0", "r1", "r2"})
        assert inst.to_dict() == original
        with pytest.raises(ValueError, match="NET.C_OP"):
            BaseInst.from_dict({"CMD": "NET", "C_OP": "bad"})

    def test_com_roundtrip_flag_value_mode(self):
        original = {"CMD": "COM", "C_OP": "set_flag", "R1": "1", "LIT": "#2", "IF": "Z"}
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, ComInst)
        assert inst.flag_val == "1"
        assert inst.r1 is None
        assert inst.to_dict() == original

    def test_com_roundtrip_register_mode_and_reg_read(self):
        original = {"CMD": "COM", "C_OP": "sync", "R1": "r0"}
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, ComInst)
        assert inst.reg_read == frozenset({"r0"})
        assert inst.to_dict() == original

    def test_com_rejects_invalid_cop(self):
        with pytest.raises(ValueError, match="COM.C_OP"):
            BaseInst.from_dict({"CMD": "COM", "C_OP": "bad"})

    @pytest.mark.parametrize("cmd", ["PA", "PB"])
    def test_custom_peripheral_roundtrip_and_reg_read(self, cmd: str):
        original = {
            "CMD": cmd,
            "C_OP": "7",
            "R1": "r0",
            "R2": "r1",
            "R3": "r2",
            "R4": "r3",
        }
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, CustomPeripheralInst)
        assert inst.reg_read == frozenset({"r0", "r1", "r2", "r3"})
        assert inst.to_dict() == original

    def test_custom_peripheral_rejects_invalid_cmd(self):
        with pytest.raises(ValueError, match="PA/PB.CMD"):
            CustomPeripheralInst.from_dict({"CMD": "PC", "C_OP": "1"})

    def test_clear_roundtrip_reg_write_and_invalid_cop(self):
        original = {"CMD": "CLEAR", "C_OP": "port"}
        inst = BaseInst.from_dict(original)
        assert isinstance(inst, ClearInst)
        assert inst.reg_write == frozenset({"s2"})
        assert inst.to_dict() == original
        with pytest.raises(ValueError, match="CLEAR.C_OP"):
            BaseInst.from_dict({"CMD": "CLEAR", "C_OP": "bad"})
