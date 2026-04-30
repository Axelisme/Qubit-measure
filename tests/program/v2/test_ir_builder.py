from zcu_tools.program.v2.ir.builder import IRBuilder
from zcu_tools.program.v2.ir.instructions import GenericInst, Instruction
from zcu_tools.program.v2.ir.node import IRBranch, IRLoop
import pytest


def test_instruction_parses_jump_label_as_generic_instruction():
    inst = Instruction.from_dict({"CMD": "JUMP", "LABEL": "target"})

    assert isinstance(inst, GenericInst)
    assert inst.cmd == "JUMP"
    assert inst.args == {"LABEL": "target"}


def test_builder_parses_branch_meta_to_irbranch():
    prog_list = [
        {"CMD": "__META__", "TYPE": "BRANCH_START", "NAME": "sel"},
        {"CMD": "__META__", "TYPE": "BRANCH_CASE_START", "NAME": "0"},
        {"CMD": "NOP"},
        {"CMD": "__META__", "TYPE": "BRANCH_CASE_END", "NAME": "0"},
        {"CMD": "__META__", "TYPE": "BRANCH_CASE_START", "NAME": "1"},
        {"CMD": "NOP"},
        {"CMD": "__META__", "TYPE": "BRANCH_CASE_END", "NAME": "1"},
        {"CMD": "__META__", "TYPE": "BRANCH_END", "NAME": "sel"},
    ]

    root = IRBuilder().build(prog_list, labels={})

    assert len(root.insts) == 1
    branch = root.insts[0]
    assert isinstance(branch, IRBranch)
    assert branch.name == "sel"
    assert len(branch.cases) == 2
    assert all(len(case.insts) == 1 for case in branch.cases)
    assert all(isinstance(case.insts[0], GenericInst) for case in branch.cases)
    assert all(case.insts[0].cmd == "NOP" for case in branch.cases)


def test_builder_parses_loop_sections_to_irloop():
    prog_list = [
        {"CMD": "__META__", "TYPE": "LOOP_START", "NAME": "r"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "initial"},
        {"CMD": "INIT"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "stop_check"},
        {"CMD": "CHECK"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "body"},
        {"CMD": "BODY"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "update"},
        {"CMD": "UPDATE"},
        {"CMD": "__META__", "TYPE": "LOOP_END", "NAME": "r"},
    ]

    root = IRBuilder().build(prog_list, labels={})

    assert len(root.insts) == 1
    loop = root.insts[0]
    assert isinstance(loop, IRLoop)
    assert loop.name == "r"
    assert [inst.cmd for inst in loop.initial.insts] == ["INIT"]
    assert [inst.cmd for inst in loop.stop_check.insts] == ["CHECK"]
    assert [inst.cmd for inst in loop.body.insts] == ["BODY"]
    assert [inst.cmd for inst in loop.update.insts] == ["UPDATE"]


def test_unbuild_flattens_irloop_in_section_order():
    prog_list = [
        {"CMD": "__META__", "TYPE": "LOOP_START", "NAME": "r"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "initial"},
        {"CMD": "INIT"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "stop_check"},
        {"CMD": "CHECK"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "body"},
        {"CMD": "BODY"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "update"},
        {"CMD": "UPDATE"},
        {"CMD": "__META__", "TYPE": "LOOP_END", "NAME": "r"},
    ]

    builder = IRBuilder()
    root = builder.build(prog_list, labels={"x": "&1"})
    out, labels = builder.unbuild(root)

    assert [d["CMD"] for d in out] == ["INIT", "CHECK", "BODY", "UPDATE"]
    assert labels == {"x": "&1"}


def test_builder_rejects_unknown_loop_section():
    prog_list = [
        {"CMD": "__META__", "TYPE": "LOOP_START", "NAME": "r"},
        {"CMD": "__META__", "TYPE": "LOOP_SECTION", "NAME": "unknown"},
    ]

    with pytest.raises(ValueError, match="Unknown LOOP_SECTION"):
        IRBuilder().build(prog_list, labels={})


def test_builder_rejects_mismatched_branch_case_end():
    prog_list = [
        {"CMD": "__META__", "TYPE": "BRANCH_START", "NAME": "sel"},
        {"CMD": "__META__", "TYPE": "BRANCH_CASE_START", "NAME": "0"},
        {"CMD": "NOP"},
        {"CMD": "__META__", "TYPE": "BRANCH_CASE_END", "NAME": "1"},
    ]

    with pytest.raises(ValueError, match="Mismatched BRANCH_CASE_END"):
        IRBuilder().build(prog_list, labels={})
