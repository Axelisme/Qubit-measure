import pytest
from zcu_tools.program.v2.ir.analysis import instruction_reads, instruction_writes
from zcu_tools.program.v2.ir.instructions import GenericInst, LabelInst, MetaInst
from zcu_tools.program.v2.ir.node import (
    BlockNode,
    IRBranch,
    IRBranchCase,
    IRLoop,
    RootNode,
)
from zcu_tools.program.v2.ir.passes.branch import BranchCaseNormalizePass
from zcu_tools.program.v2.ir.passes.dce import LabelDCEPass
from zcu_tools.program.v2.ir.passes.loop import ConstantLoopUnrollPass
from zcu_tools.program.v2.ir.passes.optimize import LoopInvariantHoistPass, PeepholePass
from zcu_tools.program.v2.ir.passes.timeline import (
    TimedInstructionMergePass,
    ZeroDelayDCEPass,
)
from zcu_tools.program.v2.ir.passes.timing import TimingSanityPass
from zcu_tools.program.v2.ir.passes.validation import (
    IRStructureValidationPass,
    LabelReferenceValidationPass,
)
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    make_default_pipeline,
)
from zcu_tools.program.v2.ir.traversal import walk_instructions


def test_label_dce_pass():
    """
    Test that the LabelDCEPass removes unreferenced labels from ir.labels.
    """
    insts = [
        GenericInst(cmd="NOP"),
        GenericInst(cmd="JUMP", args={"LABEL": "used_label"}),
        LabelInst(name="unused_definition"),
    ]

    labels = {
        "used_label": "&1",
        "dead_label": "&2"
    }

    ir = RootNode(insts=insts, labels=labels)
    
    dce_pass = LabelDCEPass()
    opt_ir = dce_pass.process(ir, PipeLineContext())
    
    assert "used_label" in opt_ir.labels
    assert "dead_label" not in opt_ir.labels
    assert "unused_definition" not in opt_ir.labels


def test_label_dce_pass_traverses_irloop_sections():
    loop = IRLoop(name="r")
    loop.stop_check.append(GenericInst(cmd="JUMP", args={"LABEL": "used_in_loop"}))
    loop.body.append(GenericInst(cmd="NOP"))

    ir = RootNode(
        insts=[loop],
        labels={"used_in_loop": "&1", "dead_label": "&2"},
    )

    dce_pass = LabelDCEPass()
    opt_ir = dce_pass.process(ir, PipeLineContext())

    assert "used_in_loop" in opt_ir.labels
    assert "dead_label" not in opt_ir.labels


def test_walk_instructions_traverses_irloop_sections_once():
    loop = IRLoop(name="r")
    loop.initial.append(GenericInst(cmd="INIT"))
    loop.stop_check.append(GenericInst(cmd="CHECK"))
    loop.body.append(GenericInst(cmd="BODY"))
    loop.update.append(GenericInst(cmd="UPDATE"))
    ir = RootNode(insts=[loop])

    assert [inst.cmd for inst in walk_instructions(ir)] == [
        "INIT",
        "CHECK",
        "BODY",
        "UPDATE",
    ]


def test_structure_validation_rejects_meta_inst_in_tree():
    ir = RootNode(insts=[MetaInst(type="LOOP_START", name="r")])

    with pytest.raises(ValueError, match="MetaInst"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_structure_validation_rejects_branch_case_not_in_body():
    case = IRBranchCase(name="0")
    branch = IRBranch(name="sel", cases=[case])
    ir = RootNode(insts=[branch])

    with pytest.raises(ValueError, match="case is not present"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_structure_validation_rejects_branch_case_without_identity():
    case = IRBranchCase()
    branch = IRBranch(name="sel", insts=[case], cases=[case])
    ir = RootNode(insts=[branch])

    with pytest.raises(ValueError, match="non-empty name"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_branch_case_normalize_sorts_case_metadata_only():
    case_2 = IRBranchCase(name="2", insts=[GenericInst(cmd="CASE2")])
    case_0 = IRBranchCase(name="0", insts=[GenericInst(cmd="CASE0")])
    dispatch = GenericInst(cmd="JUMP", args={"LABEL": "dispatch"})
    branch = IRBranch(
        name="sel",
        insts=[dispatch, case_2, case_0],
        cases=[case_2, case_0],
    )
    ir = RootNode(insts=[branch])

    out = BranchCaseNormalizePass().process(ir, PipeLineContext())

    assert out is ir
    assert [case.name for case in branch.cases] == ["0", "2"]
    assert branch.insts == [dispatch, case_2, case_0]


def test_label_reference_validation_rejects_missing_label():
    ir = RootNode(
        insts=[GenericInst(cmd="JUMP", args={"LABEL": "missing"})],
        labels={"other": "&1"},
    )

    with pytest.raises(ValueError, match="Undefined label reference"):
        LabelReferenceValidationPass().process(ir, PipeLineContext())


def test_label_reference_validation_ignores_qick_pseudo_labels():
    ir = RootNode(insts=[GenericInst(cmd="JUMP", args={"LABEL": "HERE"})])

    out = LabelReferenceValidationPass().process(ir, PipeLineContext())

    assert out is ir


def test_constant_loop_unroll_unrolls_explicit_trip_count():
    loop = IRLoop(name="r", trip_count=3)
    loop.initial.append(GenericInst(cmd="INIT"))
    loop.body.append(GenericInst(cmd="BODY"))
    loop.update.append(GenericInst(cmd="UPDATE"))
    ir = RootNode(insts=[GenericInst(cmd="PRE"), loop, GenericInst(cmd="POST")])

    out = ConstantLoopUnrollPass(max_trip_count=4).process(ir, PipeLineContext())

    assert out is ir
    assert [inst.cmd for inst in ir.insts] == [
        "PRE",
        "INIT",
        "BODY",
        "UPDATE",
        "BODY",
        "UPDATE",
        "BODY",
        "UPDATE",
        "POST",
    ]


def test_constant_loop_unroll_skips_loop_without_trip_count():
    loop = IRLoop(name="r")
    loop.body.append(GenericInst(cmd="BODY"))
    ir = RootNode(insts=[loop])

    ConstantLoopUnrollPass().process(ir, PipeLineContext())

    assert ir.insts == [loop]


def test_instruction_read_write_analysis_for_generic_inst():
    inst = GenericInst(
        cmd="REG_WR",
        args={"DST": "s1", "SRC": "op", "OP": "s2 + #1"},
    )

    assert instruction_reads(inst) == {"s2"}
    assert instruction_writes(inst) == {"s1"}


def test_loop_invariant_hoist_moves_explicitly_marked_safe_inst():
    loop = IRLoop(name="r")
    loop.update.append(GenericInst(cmd="REG_WR", args={"DST": "s1"}))
    hoistable = GenericInst(
        cmd="REG_WR",
        args={"DST": "s2", "SRC": "imm", "LIT": "#1", "IR_HOISTABLE": True},
    )
    loop.body.append(hoistable)
    loop.body.append(GenericInst(cmd="BODY"))
    ir = RootNode(insts=[loop])

    LoopInvariantHoistPass().process(ir, PipeLineContext())

    assert [inst.cmd for inst in loop.initial.insts] == ["REG_WR"]
    assert loop.initial.insts[0].args == {"DST": "s2", "SRC": "imm", "LIT": "#1"}
    assert [inst.cmd for inst in loop.body.insts] == ["BODY"]


def test_loop_invariant_hoist_keeps_control_register_writes_in_body():
    loop = IRLoop(name="r")
    loop.update.append(GenericInst(cmd="REG_WR", args={"DST": "s1"}))
    blocked = GenericInst(
        cmd="REG_WR",
        args={"DST": "s1", "SRC": "imm", "LIT": "#1", "IR_HOISTABLE": True},
    )
    loop.body.append(blocked)
    ir = RootNode(insts=[loop])

    LoopInvariantHoistPass().process(ir, PipeLineContext())

    assert loop.initial.insts == []
    assert loop.body.insts == [blocked]


def test_peephole_pass_strips_internal_annotations_without_removing_nop():
    ir = RootNode(
        insts=[
            GenericInst(cmd="NOP"),
            GenericInst(cmd="REG_WR", args={"DST": "s1", "IR_NOTE": "x"}),
        ]
    )

    PeepholePass().process(ir, PipeLineContext())

    assert [inst.cmd for inst in ir.insts] == ["NOP", "REG_WR"]
    assert ir.insts[1].args == {"DST": "s1"}


def test_timing_sanity_pass_rejects_negative_time_literal():
    ir = RootNode(insts=[GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#-1"})])

    with pytest.raises(ValueError, match="non-negative"):
        TimingSanityPass().process(ir, PipeLineContext())


def test_timing_sanity_pass_accepts_register_time_increment():
    ir = RootNode(insts=[GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "R1": "s1"})])

    out = TimingSanityPass().process(ir, PipeLineContext())

    assert out is ir


def test_zero_delay_dce_removes_zero_literal_time_increment():
    ir = RootNode(
        insts=[
            GenericInst(cmd="PRE"),
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#0"}),
            GenericInst(cmd="POST"),
        ]
    )

    out = ZeroDelayDCEPass().process(ir, PipeLineContext())

    assert out is ir
    assert [inst.cmd for inst in ir.insts] == ["PRE", "POST"]


def test_zero_delay_dce_traverses_loop_sections():
    loop = IRLoop(name="r")
    loop.initial.append(GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#0"}))
    loop.body.append(GenericInst(cmd="BODY"))
    loop.update.append(GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#00"}))
    ir = RootNode(insts=[loop])

    ZeroDelayDCEPass().process(ir, PipeLineContext())

    assert loop.initial.insts == []
    assert [inst.cmd for inst in loop.body.insts] == ["BODY"]
    assert loop.update.insts == []


def test_zero_delay_dce_preserves_non_noop_time_shapes():
    tagged = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#0", "TAG": "keep"})
    register_driven = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "R1": "s1"})
    set_ref = GenericInst(cmd="TIME", args={"C_OP": "set_ref", "LIT": "#0"})
    nonzero = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#1"})
    ir = RootNode(insts=[tagged, register_driven, set_ref, nonzero])

    ZeroDelayDCEPass().process(ir, PipeLineContext())

    assert ir.insts == [tagged, register_driven, set_ref, nonzero]


def test_timed_instruction_merge_merges_adjacent_time_literals():
    ir = RootNode(
        insts=[
            GenericInst(cmd="PRE"),
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#2"}),
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#3"}),
            GenericInst(cmd="POST"),
        ]
    )

    out = TimedInstructionMergePass().process(ir, PipeLineContext())

    assert out is ir
    assert [inst.to_dict() for inst in ir.insts] == [
        {"CMD": "PRE"},
        {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#5"},
        {"CMD": "POST"},
    ]


def test_timed_instruction_merge_traverses_loop_sections():
    loop = IRLoop(name="r")
    loop.initial.append(GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#1"}))
    loop.initial.append(GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#2"}))
    loop.body.append(GenericInst(cmd="BODY"))
    loop.update.append(GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#3"}))
    loop.update.append(GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#4"}))
    ir = RootNode(insts=[loop])

    TimedInstructionMergePass().process(ir, PipeLineContext())

    assert [inst.to_dict() for inst in loop.initial.insts] == [
        {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#3"}
    ]
    assert [inst.cmd for inst in loop.body.insts] == ["BODY"]
    assert [inst.to_dict() for inst in loop.update.insts] == [
        {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#7"}
    ]


def test_timed_instruction_merge_does_not_cross_barriers():
    label = LabelInst(name="barrier")
    nested = BlockNode(
        insts=[
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#5"}),
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#6"}),
        ]
    )
    ir = RootNode(
        insts=[
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#1"}),
            label,
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#2"}),
            GenericInst(cmd="WPORT_WR", args={"DST": "w0", "TIME": "@10"}),
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#3"}),
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#4"}),
            nested,
        ]
    )

    TimedInstructionMergePass().process(ir, PipeLineContext())

    assert [item.to_dict() for item in ir.insts[:5]] == [
        {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#1"},
        {"LABEL": "barrier"},
        {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#2"},
        {"CMD": "WPORT_WR", "DST": "w0", "TIME": "@10"},
        {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#7"},
    ]
    assert [inst.to_dict() for inst in nested.insts] == [
        {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#11"},
    ]


def test_timed_instruction_merge_preserves_non_mergeable_time_shapes():
    tagged = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#1", "TAG": "keep"})
    register_driven = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "R1": "s1"})
    set_ref = GenericInst(cmd="TIME", args={"C_OP": "set_ref", "LIT": "#1"})
    zero = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#0"})
    negative = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#-1"})
    ir = RootNode(insts=[tagged, register_driven, set_ref, zero, negative])

    TimedInstructionMergePass().process(ir, PipeLineContext())

    assert ir.insts == [tagged, register_driven, set_ref, zero, negative]


def test_default_pipeline_removes_zero_delay_then_merges_time_literals():
    ir = RootNode(
        insts=[
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#0"}),
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#2"}),
            GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#3"}),
        ]
    )

    make_default_pipeline(PipeLineConfig())(ir)

    assert [inst.to_dict() for inst in ir.insts] == [
        {"CMD": "TIME", "C_OP": "inc_ref", "LIT": "#5"}
    ]
