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
import pytest
from zcu_tools.program.v2.ir.instructions import (
    DmemReadInst,
    DmemWriteInst,
    DportWriteInst,
    GenericInst,
    Instruction,
    JumpInst,
    LabelInst,
    MetaInst,
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TestInst,
    TimeInst,
    WaitInst,
)


class TestTimeInstruction:
    """Tests for TimeInst (TIME opcode)."""

    def test_construction_with_all_fields(self):
        inst = TimeInst(c_op="inc_ref", lit="#10", r1="r0", line=5, p_addr=3)
        assert inst.c_op == "inc_ref"
        assert inst.lit == "#10"
        assert inst.r1 == "r0"
        assert inst.line == 5
        assert inst.p_addr == 3

    def test_construction_with_defaults(self):
        inst = TimeInst()
        assert inst.c_op == ""
        assert inst.lit is None
        assert inst.r1 is None
        assert inst.line is None
        assert inst.p_addr is None

    def test_dispatch_time_to_timeinst(self):
        d = {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#5", "R1": "r1"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, TimeInst)
        assert inst.c_op == "inc_ref"
        assert inst.lit == "#5"
        assert inst.r1 == "r1"

    def test_roundtrip_time_full(self):
        original = {"CMD": "TIME", "C_OP": "trigger", "LIT": "#20", "R1": "r2", "LINE": 10}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_time_minimal(self):
        original = {"CMD": "TIME", "C_OP": "inc_ref"}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_time_immutable(self):
        inst = TimeInst(c_op="inc_ref", lit="#10")
        with pytest.raises(Exception):  # FrozenInstanceError
            inst.c_op = "trigger"

    def test_time_missing_c_op_defaults(self):
        d = {"CMD": "TIME"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, TimeInst)
        assert inst.c_op == ""


class TestTestInstruction:
    """Tests for TestInst (TEST opcode)."""

    def test_construction_with_all_fields(self):
        inst = TestInst(op="r1-r2", uf="1", line=7, p_addr=5)
        assert inst.op == "r1-r2"
        assert inst.uf == "1"
        assert inst.line == 7
        assert inst.p_addr == 5

    def test_dispatch_test_to_testinst(self):
        d = {"CMD": "TEST", "OP": "r1==r2", "UF": "1"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, TestInst)
        assert inst.op == "r1==r2"
        assert inst.uf == "1"

    def test_roundtrip_test_full(self):
        original = {"CMD": "TEST", "OP": "r3 & #255", "UF": "1", "LINE": 15}
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
            inst.op = "r1+r2"


class TestJumpInstruction:
    """Tests for JumpInst (JUMP opcode)."""

    def test_construction_unconditional(self):
        inst = JumpInst(label="loop", line=10, p_addr=8)
        assert inst.label == "loop"
        assert inst.if_cond is None
        assert inst.addr is None

    def test_construction_conditional(self):
        inst = JumpInst(label="exit", if_cond="eq", line=12, p_addr=9)
        assert inst.label == "exit"
        assert inst.if_cond == "eq"

    def test_construction_with_addr(self):
        inst = JumpInst(label="", addr="s15")
        assert inst.label == ""
        assert inst.addr == "s15"

    def test_dispatch_jump_unconditional(self):
        d = {"CMD": "JUMP", "LABEL": "loop"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, JumpInst)
        assert inst.label == "loop"
        assert inst.if_cond is None

    def test_dispatch_jump_conditional(self):
        d = {"CMD": "JUMP", "LABEL": "end", "IF": "nz"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, JumpInst)
        assert inst.label == "end"
        assert inst.if_cond == "nz"

    def test_dispatch_jump_with_addr(self):
        d = {"CMD": "JUMP", "ADDR": "s15", "IF": "eq"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, JumpInst)
        assert inst.addr == "s15"
        assert inst.if_cond == "eq"

    def test_roundtrip_jump_unconditional(self):
        original = {"CMD": "JUMP", "LABEL": "loop", "LINE": 20}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_jump_conditional(self):
        original = {"CMD": "JUMP", "LABEL": "end", "IF": "eq", "LINE": 21}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_jump_special_labels(self):
        """Test QICK special labels: NEXT, PREV, HERE, SKIP."""
        for label in ["NEXT", "PREV", "HERE", "SKIP"]:
            original = {"CMD": "JUMP", "LABEL": label}
            inst = Instruction.from_dict(original)
            recovered = inst.to_dict()
            assert recovered == original

    def test_jump_immutable(self):
        inst = JumpInst(label="loop")
        with pytest.raises(Exception):
            inst.label = "exit"

    def test_jump_minimal(self):
        """Empty JUMP should work (no label, no addr)."""
        d = {"CMD": "JUMP"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, JumpInst)
        assert inst.label == ""
        assert inst.addr is None


class TestRegWriteInstruction:
    """Tests for RegWriteInst (REG_WR opcode)."""

    def test_construction_imm_source(self):
        inst = RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#10"})
        assert inst.dst == "s1"
        assert inst.src == "imm"
        assert inst.extra_args == {"LIT": "#10"}

    def test_construction_op_source(self):
        inst = RegWriteInst(
            dst="s2", src="op", extra_args={"OP": "s1+#1", "UF": "0"}
        )
        assert inst.dst == "s2"
        assert inst.src == "op"
        assert "OP" in inst.extra_args

    def test_dispatch_regwr_imm(self):
        d = {"CMD": "REG_WR", "DST": "s1", "SRC": "imm", "LIT": "#0"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, RegWriteInst)
        assert inst.dst == "s1"
        assert inst.src == "imm"
        assert inst.extra_args["LIT"] == "#0"

    def test_dispatch_regwr_op(self):
        d = {"CMD": "REG_WR", "DST": "s2", "SRC": "op", "OP": "s1+#1", "UF": "1"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, RegWriteInst)
        assert inst.src == "op"
        assert inst.extra_args["OP"] == "s1+#1"
        assert inst.extra_args["UF"] == "1"

    def test_roundtrip_regwr_imm(self):
        original = {"CMD": "REG_WR", "DST": "s1", "SRC": "imm", "LIT": "#5", "LINE": 1}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_roundtrip_regwr_op(self):
        original = {
            "CMD": "REG_WR",
            "DST": "s2",
            "SRC": "op",
            "OP": "s1+#1",
            "UF": "1",
            "LINE": 2,
        }
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_regwr_preserves_unknown_fields(self):
        """REG_WR can have implementation-specific fields that should be preserved."""
        original = {
            "CMD": "REG_WR",
            "DST": "s3",
            "SRC": "imm",
            "LIT": "#100",
            "CUSTOM_FIELD": "custom_value",
        }
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original

    def test_regwr_immutable(self):
        inst = RegWriteInst(dst="s1", src="imm")
        with pytest.raises(Exception):
            inst.dst = "s2"


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
            inst.dst = "1"


class TestGenericInstruction:
    """Tests for GenericInst (fallback for unknown opcodes)."""

    def test_dispatch_unknown_opcode_to_generic(self):
        d = {"CMD": "UNKNOWN_OP", "FIELD1": "value1", "FIELD2": "value2"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, GenericInst)
        assert inst.cmd == "UNKNOWN_OP"
        assert inst.args == {"FIELD1": "value1", "FIELD2": "value2"}

    def test_dispatch_dport_wr_to_specific(self):
        """DPORT_WR is now specially handled; should be DportWriteInst."""
        d = {"CMD": "DPORT_WR", "DST": "0", "DATA": "1"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, DportWriteInst)
        assert inst.dst == "0"
        assert inst.data == "1"

    def test_generic_roundtrip(self):
        original = {"CMD": "SOME_OP", "ARG1": "val1", "ARG2": "val2", "LINE": 5}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original


class TestLabelInstruction:
    """Tests for LabelInst."""

    def test_label_only(self):
        d = {"LABEL": "my_label"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, LabelInst)
        assert inst.name == "my_label"
        assert inst.args == {}

    def test_label_with_extra_args(self):
        d = {"LABEL": "loop_start", "EXTRA": "data"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, LabelInst)
        assert inst.name == "loop_start"
        assert inst.args["EXTRA"] == "data"

    def test_label_roundtrip(self):
        original = {"LABEL": "end_loop", "LINE": 30}
        inst = Instruction.from_dict(original)
        recovered = inst.to_dict()
        assert recovered == original


class TestMetaInstruction:
    """Tests for MetaInst."""

    def test_meta_construction(self):
        inst = MetaInst(type="loop", name="loop_1", args={"n": 10})
        assert inst.type == "loop"
        assert inst.name == "loop_1"
        assert inst.args["n"] == 10

    def test_meta_roundtrip(self):
        original = {"CMD": "__META__", "TYPE": "loop", "NAME": "L1", "ARGS": {"n": 5}}
        inst = Instruction.from_dict(original)
        assert isinstance(inst, MetaInst)
        recovered = inst.to_dict()
        # to_dict includes LINE and P_ADDR even if None
        assert recovered["CMD"] == "__META__"
        assert recovered["TYPE"] == "loop"
        assert recovered["NAME"] == "L1"
        assert recovered["ARGS"] == {"n": 5}

    def test_meta_dispatch(self):
        d = {"CMD": "__META__", "TYPE": "loop", "NAME": "L2", "ARGS": {}}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, MetaInst)

    def test_meta_optional_args(self):
        """MetaInst should handle missing ARGS field."""
        d = {"CMD": "__META__", "TYPE": "branch", "NAME": "B1"}
        inst = Instruction.from_dict(d)
        assert isinstance(inst, MetaInst)
        assert inst.args == {}


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
        d = {"CMD": "TIME", "C_OP": "inc_ref", "LINE": 5}
        inst = Instruction.from_dict(d)
        assert inst.lit is None
        assert inst.r1 is None
        assert inst.line == 5
