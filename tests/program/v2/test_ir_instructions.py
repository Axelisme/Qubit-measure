"""
Comprehensive tests for typed Instruction classes (Phase 2).

Tests include:
- Construction and dispatcher routing for all typed Instruction types
- Round-trip preservation (dict → Inst → dict)
- Immutability verification
- Optional field handling
- Analysis helpers (reads/writes extraction)
- Edge cases and malformed dict handling
"""

from __future__ import annotations

import pytest
from zcu_tools.program.v2.ir.instructions import (
    DmemReadInst,
    DportWriteInst,
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    PortWriteInst,
    RegWriteInst,
    TestInst,
    TimeInst,
    WaitInst,
)
from zcu_tools.program.v2.ir.labels import Label


class TestTimeInstruction:
    """Tests for TimeInst (TIME opcode)."""

    def test_construction_with_all_fields(self):
        inst = TimeInst(c_op="inc_ref", lit="#10", r1="r0")
        assert inst.c_op == "inc_ref"
        assert getattr(inst, "lit") == "#10"
        assert inst.r1 == "r0"

    def test_construction_with_defaults(self):
        inst = TimeInst()
        assert inst.c_op == ""
        assert inst.lit is None
        assert inst.r1 is None

    def test_dispatch_time_to_timeinst(self):
        d = {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#5", "R1": "r1"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, TimeInst)
        assert inst.c_op == "inc_ref"
        assert inst.lit == "#5"
        assert inst.r1 == "r1"

    def test_roundtrip_time_full(self):
        original = {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#10", "R1": "r1"}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_time_minimal(self):
        original = {"CMD": "TIME", "C_OP": "trigger"}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_time_immutable(self):
        inst = TimeInst(c_op="inc_ref")
        with pytest.raises(Exception):
            inst.c_op = "trigger"  # type: ignore


class TestTestInstruction:
    """Tests for TestInst (TEST opcode)."""

    def test_construction(self):
        inst = TestInst(op="r1-r2", uf="1")
        assert inst.op == "r1-r2"
        assert inst.uf == "1"

    def test_dispatch_test_to_testinst(self):
        d = {"CMD": "TEST", "OP": "r1==r2", "UF": "1"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, TestInst)
        assert inst.op == "r1==r2"
        assert inst.uf == "1"

    def test_roundtrip_test_full(self):
        original = {"CMD": "TEST", "OP": "r3 & #255", "UF": "1"}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_test_minimal(self):
        original = {"CMD": "TEST", "OP": "r1"}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_test_immutable(self):
        inst = TestInst(op="r1-r2")
        with pytest.raises(Exception):
            inst.op = "r1+r2"  # type: ignore


class TestJumpInstruction:
    """Tests for JumpInst (JUMP opcode)."""

    def test_construction_unconditional(self):
        inst = JumpInst(label=Label("loop"))
        assert str(inst.label) == "loop"
        assert inst.if_cond is None
        assert inst.addr is None

    def test_construction_conditional(self):
        inst = JumpInst(label=Label("exit"), if_cond="eq")
        assert str(inst.label) == "exit"
        assert inst.if_cond == "eq"

    def test_construction_with_addr(self):
        inst = JumpInst(addr="s15")
        assert inst.label is None
        assert inst.addr == "s15"

    def test_dispatch_jump_unconditional(self):
        d = {"CMD": "JUMP", "LABEL": "loop"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, JumpInst)
        assert str(inst.label) == "loop"
        assert inst.if_cond is None

    def test_dispatch_jump_conditional(self):
        d = {"CMD": "JUMP", "LABEL": "end", "IF": "nz"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, JumpInst)
        assert str(inst.label) == "end"
        assert inst.if_cond == "nz"

    def test_dispatch_jump_with_addr(self):
        d = {"CMD": "JUMP", "ADDR": "s15", "IF": "eq"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, JumpInst)
        assert inst.addr == "s15"
        assert inst.if_cond == "eq"

    def test_roundtrip_jump_unconditional(self):
        original = {"CMD": "JUMP", "LABEL": "loop"}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_jump_conditional(self):
        original = {"CMD": "JUMP", "LABEL": "end", "IF": "eq"}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_jump_with_raw_control_fields(self):
        original = {
            "CMD": "JUMP",
            "LABEL": "loop",
            "IF": "nz",
            "WR": "s1 op",
            "OP": "s1 + #1",
            "UF": "1",
        }
        inst = Instruction.from_dict(original)
        assert isinstance(inst, JumpInst)
        assert inst.wr == "s1 op"
        assert inst.op == "s1 + #1"
        assert inst.uf == "1"
        assert inst.to_dict() == original

    def test_roundtrip_jump_special_labels(self):
        """Test QICK special labels: NEXT, PREV, HERE, SKIP."""
        for label in ["NEXT", "PREV", "HERE", "SKIP"]:
            original = {"CMD": "JUMP", "LABEL": label}
            inst = Instruction.from_dict(original)
            recovered = inst.to_dict()
            assert recovered == original

    def test_jump_immutable(self):
        inst = JumpInst(label=Label("loop"))
        with pytest.raises(Exception):
            inst.label = Label("exit")  # type: ignore

    def test_jump_minimal(self):
        """Empty JUMP should work (no label, no addr)."""
        d = {"CMD": "JUMP"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, JumpInst)
        assert inst.label is None
        assert inst.addr is None


class TestRegWriteInstruction:
    """Tests for RegWriteInst (REG_WR opcode)."""

    def test_construction_imm_source(self):
        inst = RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#10"})
        assert inst.dst == "s1"
        assert inst.src == "imm"
        assert inst.extra_args == {"LIT": "#10"}

    def test_construction_op_source(self):
        inst = RegWriteInst(dst="s2", src="op", extra_args={"OP": "s1+#1", "UF": "0"})
        assert inst.dst == "s2"
        assert inst.src == "op"
        assert inst.op is None
        assert inst.extra_args["OP"] == "s1+#1"

    def test_dispatch_regwr_imm(self):
        d = {"CMD": "REG_WR", "DST": "s1", "SRC": "imm", "LIT": "#0"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, RegWriteInst)
        assert inst.dst == "s1"
        assert inst.src == "imm"
        assert inst.lit == "#0"
        assert inst.extra_args == {}

    def test_dispatch_regwr_op(self):
        d = {"CMD": "REG_WR", "DST": "s2", "SRC": "op", "OP": "s1+#1", "UF": "1"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, RegWriteInst)
        assert inst.src == "op"
        assert inst.op == "s1+#1"

    def test_dispatch_regwr_dmem_lowering(self):
        """REG_WR src=dmem is recognized as DmemReadInst."""
        d = {"CMD": "REG_WR", "DST": "r0", "SRC": "dmem", "ADDR": "&123"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, DmemReadInst)
        assert inst.src == "dmem"
        assert inst.dst == "r0"

    def test_dispatch_legacy_dmem_rd_lowering(self):
        """DMEM_RD opcode is also recognized as DmemReadInst."""
        d = {"CMD": "DMEM_RD", "DST": "r1", "ADDR": "s15"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, DmemReadInst)
        assert inst.dst == "r1"

    def test_roundtrip_regwr(self):
        original = {"CMD": "REG_WR", "DST": "s1", "SRC": "imm", "LIT": "#100"}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_regwr_immutable(self):
        inst = RegWriteInst(dst="s1", src="imm")
        with pytest.raises(Exception):
            inst.dst = "s2"  # type: ignore


class TestPortWriteInstruction:
    """Tests for PortWriteInst (WPORT_WR opcode)."""

    def test_construction_with_extra_args(self):
        inst = PortWriteInst(
            dst="0", time="t0", extra_args={"DATA": "0x1234", "PHASE": "#0"}
        )
        assert inst.dst == "0"
        assert inst.time == "t0"
        assert inst.extra_args["DATA"] == "0x1234"

    def test_dispatch_wport_wr(self):
        d = {"CMD": "WPORT_WR", "DST": "1", "TIME": "t1", "DATA": "0xABCD"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, PortWriteInst)
        assert inst.dst == "1"
        assert inst.time == "t1"
        assert inst.extra_args["DATA"] == "0xABCD"

    def test_roundtrip_wport_wr(self):
        original = {"CMD": "WPORT_WR", "DST": "0", "TIME": "t0", "DATA": "0x5678"}
        inst = Instruction.from_dict(original)
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
        inst = Instruction.from_dict(original)
        assert isinstance(inst, PortWriteInst)
        assert inst.src == "wmem"
        assert inst.addr == "&12"
        assert inst.extra_args == {}
        assert inst.to_dict() == original

    def test_wport_wr_preserves_unknown_fields(self):
        """WPORT_WR can have various data fields that must be preserved."""
        original = {
            "CMD": "WPORT_WR",
            "DST": "2",
            "TIME": "t2",
            "PHASE": "#45",
            "FREQ": "#100",
        }
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_wport_wr_immutable(self):
        inst = PortWriteInst(dst="0", time="t0")
        with pytest.raises(Exception):
            inst.dst = "1"  # type: ignore


class TestUnknownOpcode:
    def test_dispatch_unknown_opcode_raises(self):
        with pytest.raises(ValueError, match="Unknown instruction opcode"):
            Instruction.from_dict({"CMD": "UNKNOWN_OP", "FIELD1": "value1"})

    def test_dispatch_dport_wr_to_specific(self):
        """DPORT_WR is specially handled; should be DportWriteInst."""
        d = {"CMD": "DPORT_WR", "DST": "0", "DATA": "1"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, DportWriteInst)
        assert inst.dst == "0"
        assert inst.data == "1"


class TestWaitInstruction:
    """Tests for WaitInst."""

    def test_wait_default_shape(self):
        inst = WaitInst()
        assert inst.to_dict() == {"CMD": "WAIT", "C_OP": "time"}

    def test_wait_roundtrip(self):
        original = {"CMD": "WAIT", "C_OP": "time", "TIME": "@10", "ADDR": "s15"}
        inst = Instruction.from_dict(original)
        assert isinstance(inst, WaitInst)
        assert inst.time == "@10"
        assert inst.addr == "s15"
        assert inst.to_dict() == original


class TestLabelInstruction:
    """Tests for LabelInst."""

    def test_label_only(self):
        d = {"LABEL": "my_label"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, LabelInst)
        assert str(inst.name) == "my_label"
        assert inst.args == {}

    def test_label_with_extra_args(self):
        d = {"LABEL": "loop_start", "EXTRA": "data"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, LabelInst)
        assert str(inst.name) == "loop_start"
        assert inst.args["EXTRA"] == "data"

    def test_label_roundtrip(self):
        original = {"LABEL": "end_loop"}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original


class TestMetaInstruction:
    """Tests for MetaInst."""

    def test_meta_construction(self):
        inst = MetaInst(type="loop", name="loop_1", info={"n": 10})
        assert inst.type == "loop"
        assert str(inst.name) == "loop_1"
        assert inst.info["n"] == 10


class TestConditionalJumpPattern:
    """
    Tests for the TEST + JUMP pattern (conditional jumps in QICK).

    QICK's CondJump macro expands to:
      1. TestInst: evaluates condition
      2. JumpInst with if_cond: conditional jump based on test result
    """

    def test_test_then_jump_pattern(self):
        """Simulate conditional jump as TEST + JUMP sequence."""
        test_dict = {"CMD": "TEST", "OP": "r1-#10", "UF": "1"}
        jump_dict = {"CMD": "JUMP", "LABEL": "end", "IF": "eq"}

        test_inst = Instruction.from_dict(test_dict)
        jump_inst = Instruction.from_dict(jump_dict)

        assert isinstance(test_inst, TestInst)
        assert isinstance(jump_inst, JumpInst)

        # Should roundtrip correctly
        assert test_inst.to_dict() == test_dict
        assert jump_inst.to_dict() == jump_dict

    def test_conditional_jump_conditions(self):
        """Test various condition codes for conditional jumps."""
        conditions = ["eq", "nz", "z", "f", "s", "ns"]
        for cond in conditions:
            d = {"CMD": "JUMP", "LABEL": "target", "IF": cond}
            inst = Instruction.from_dict(d)
            assert isinstance(inst, JumpInst)
            assert inst.if_cond == cond
            assert inst.to_dict() == d


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_missing_cmd_raises_error(self):
        with pytest.raises(ValueError):
            Instruction.from_dict({"FIELD": "value"})

    def test_empty_cmd_raises_error(self):
        with pytest.raises(ValueError):
            Instruction.from_dict({"CMD": ""})

    def test_label_without_cmd_recognized(self):
        """LABEL-only dicts are recognized even without CMD."""
        d = {"LABEL": "test_label"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, LabelInst)

    def test_mixed_optional_fields(self):
        """Instructions with selective optional fields."""
        d = {"CMD": "TIME", "C_OP": "inc_ref"}
        inst = Instruction.from_dict(d)
        assert getattr(inst, "lit") is None
        assert getattr(inst, "r1") is None
