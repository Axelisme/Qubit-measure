import pytest
from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.ir.instructions import GenericInst, Instruction
from zcu_tools.program.v2.ir.node import IRBranch, IRBranchCase, IRLoop


def test_instruction_parses_jump_label_as_generic_instruction():
    inst = Instruction.from_dict({"CMD": "JUMP", "LABEL": "target"})

    assert isinstance(inst, GenericInst)
    assert inst.cmd == "JUMP"
    assert inst.args == {"LABEL": "target"}
