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
from zcu_tools.program.v2.ir.passes.loop import ConstantLoopUnrollPass
from zcu_tools.program.v2.ir.passes.optimize import (
    LoopInvariantHoistPass,
    PeepholePass,
)
from zcu_tools.program.v2.ir.passes.timeline import (
    TimedInstructionMergePass,
    ZeroDelayDCEPass,
)
from zcu_tools.program.v2.ir.passes.validation import (
    IRStructureValidationPass,
    LabelReferenceValidationPass,
)
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    make_default_pipeline,
)


def test_ir_node_analysis():
    inst = GenericInst(cmd="REG_WR", args={"DST": "s1", "SRC": "s2"})
    assert instruction_writes(inst) == {"s1"}
    assert instruction_reads(inst) == {"s2"}


def test_structure_validation_rejects_empty_loop_name():
    loop = IRLoop(name="")
    ir = RootNode(insts=[loop])

    with pytest.raises(ValueError, match="non-empty name"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_structure_validation_rejects_empty_branch_name():
    branch = IRBranch(name="")
    ir = RootNode(insts=[branch])

    with pytest.raises(ValueError, match="non-empty name"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_structure_validation_rejects_branch_without_cases():
    branch = IRBranch(name="sel", cases=[])
    ir = RootNode(insts=[branch])

    with pytest.raises(ValueError, match="at least one case"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_structure_validation_rejects_remaining_metainst():
    meta = MetaInst(type="ANY", name="n")
    ir = RootNode(insts=[meta])

    with pytest.raises(ValueError, match="MetaInst should not remain"):
        IRStructureValidationPass().process(ir, PipeLineContext())


def test_branch_case_normalize_sorts_case_metadata_only():
    case_2 = IRBranchCase(name="2", insts=[GenericInst(cmd="CASE2")])
    case_0 = IRBranchCase(name="0", insts=[GenericInst(cmd="CASE0")])
    dispatch_inst = GenericInst(cmd="JUMP", args={"LABEL": "dispatch"})
    
    branch = IRBranch(
        name="sel",
        dispatch=BlockNode(insts=[dispatch_inst]),
        cases=[case_2, case_0],
    )
    ir = RootNode(insts=[branch])

    BranchCaseNormalizePass().process(ir, PipeLineContext())

    assert branch.cases == [case_0, case_2]
    assert branch.dispatch.insts == [dispatch_inst]


def test_label_reference_validation_allows_defined_labels():
    ir = RootNode(
        insts=[GenericInst(cmd="JUMP", args={"LABEL": "target"})],
        labels={"target": "addr"},
    )

    LabelReferenceValidationPass().process(ir, PipeLineContext())


def test_label_reference_validation_rejects_undefined_labels():
    ir = RootNode(
        insts=[GenericInst(cmd="JUMP", args={"LABEL": "missing"})],
        labels={"target": "addr"},
    )

    with pytest.raises(ValueError, match="Undefined label reference"):
        LabelReferenceValidationPass().process(ir, PipeLineContext())


def test_label_reference_validation_allows_structural_labels():
    loop = IRLoop(name="r", start_label="loop_start")
    ir = RootNode(
        insts=[
            GenericInst(cmd="JUMP", args={"LABEL": "loop_start"}),
            loop,
        ]
    )

    LabelReferenceValidationPass().process(ir, PipeLineContext())


def test_constant_loop_unroll_unrolls_small_count():
    loop = IRLoop(name="r", trip_count=2)
    loop.body.append(GenericInst(cmd="BODY"))
    loop.update.append(GenericInst(cmd="INC"))
    ir = RootNode(insts=[loop])

    ConstantLoopUnrollPass(max_trip_count=5).process(ir, PipeLineContext())

    assert len(ir.insts) == 4
    assert [inst.cmd for inst in ir.insts] == ["BODY", "INC", "BODY", "INC"]


def test_constant_loop_unroll_preserves_large_count():
    loop = IRLoop(name="r", trip_count=100)
    ir = RootNode(insts=[loop])

    ConstantLoopUnrollPass(max_trip_count=10).process(ir, PipeLineContext())

    assert ir.insts == [loop]


def test_loop_invariant_hoist_hoists_marked_instructions():
    loop = IRLoop(name="r")
    hoistable = GenericInst(cmd="VAL", args={"IR_HOISTABLE": True})
    loop.body.append(hoistable)
    loop.body.append(GenericInst(cmd="BODY"))
    ir = RootNode(insts=[loop])

    LoopInvariantHoistPass().process(ir, PipeLineContext())

    assert loop.initial.insts == [GenericInst(cmd="VAL", args={})]
    assert len(loop.body.insts) == 1
    assert loop.body.insts[0].cmd == "BODY"


def test_loop_invariant_hoist_does_not_hoist_if_registers_blocked():
    loop = IRLoop(name="r")
    # initial writes s1
    loop.initial.append(GenericInst(cmd="WR", args={"DST": "s1"}))
    # hoistable reads s1 -> blocked
    hoistable = GenericInst(cmd="RD", args={"SRC": "s1", "IR_HOISTABLE": True})
    loop.body.append(hoistable)
    ir = RootNode(insts=[loop])

    LoopInvariantHoistPass().process(ir, PipeLineContext())

    assert loop.initial.insts == [GenericInst(cmd="WR", args={"DST": "s1"})]
    assert hoistable in loop.body.insts


def test_peephole_removes_internal_annotations():
    ir = RootNode(
        insts=[GenericInst(cmd="OP", args={"IR_HOISTABLE": True, "VAL": 1})]
    )

    PeepholePass().process(ir, PipeLineContext())

    assert ir.insts[0].args == {"VAL": 1}


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
    assert loop.update.insts == []


def test_zero_delay_dce_preserves_non_noop_time_shapes():
    tagged = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#0", "TAG": "keep"})
    register_driven = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "R1": "s1"})
    set_ref = GenericInst(cmd="TIME", args={"C_OP": "set_ref", "LIT": "#0"})
    nonzero = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#1"})
    ir = RootNode(insts=[tagged, register_driven, set_ref, nonzero])

    ZeroDelayDCEPass().process(ir, PipeLineContext())

    assert len(ir.insts) == 4


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

    assert [inst.args["LIT"] for inst in loop.initial.insts] == ["#3"]
    assert [inst.args["LIT"] for inst in loop.update.insts] == ["#7"]


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


def test_timed_instruction_merge_preserves_non_mergeable_time_shapes():
    tagged = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#1", "TAG": "keep"})
    register_driven = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "R1": "s1"})
    set_ref = GenericInst(cmd="TIME", args={"C_OP": "set_ref", "LIT": "#1"})
    zero = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#0"})
    negative = GenericInst(cmd="TIME", args={"C_OP": "inc_ref", "LIT": "#-1"})
    ir = RootNode(insts=[tagged, register_driven, set_ref, zero, negative])

    TimedInstructionMergePass().process(ir, PipeLineContext())

    assert len(ir.insts) == 5


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
