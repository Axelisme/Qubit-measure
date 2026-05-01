import pytest
from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.ir.instructions import GenericInst, Instruction
from zcu_tools.program.v2.ir.node import IRBranch, IRBranchCase, IRLoop, BlockNode


def test_instruction_parses_jump_label_as_generic_instruction():
    inst = Instruction.from_dict({"CMD": "JUMP", "LABEL": "target"})

    assert isinstance(inst, GenericInst)
    assert inst.cmd == "JUMP"
    assert inst.args == {"LABEL": "target"}


def test_branch_roundtrip_preserves_cases():
    """Verify IRBranch.emit() includes dispatch + cases in order."""
    dispatch_inst = GenericInst(cmd="JUMP", args={"LABEL": "case_0"})
    case_0_inst = GenericInst(cmd="CASE_0_BODY")
    case_1_inst = GenericInst(cmd="CASE_1_BODY")

    case_0 = IRBranchCase(name="0", insts=[case_0_inst])
    case_1 = IRBranchCase(name="1", insts=[case_1_inst])

    branch = IRBranch(
        name="sel",
        dispatch=BlockNode(insts=[dispatch_inst]),
        cases=[case_0, case_1],
    )

    # Emit to prog_list
    prog_list: list[dict] = []
    branch.emit(prog_list)

    # Should have: dispatch_inst, case_0_inst, case_1_inst
    assert len(prog_list) == 3
    assert prog_list[0]["CMD"] == "JUMP"
    assert prog_list[1]["CMD"] == "CASE_0_BODY"
    assert prog_list[2]["CMD"] == "CASE_1_BODY"
