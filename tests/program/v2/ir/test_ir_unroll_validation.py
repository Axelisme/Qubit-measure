from __future__ import annotations

import pytest
from zcu_tools.program.v2.ir.instructions import NopInst
from zcu_tools.program.v2.ir.node import BasicBlockNode, BlockNode, IRLoop, RootNode
from zcu_tools.program.v2.ir.passes.loop.unroll import UnrollLoopPass
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig, PipeLineContext


def test_unroll_rejects_non_general_counter():
    # s11 is a system register, not a general-purpose register.
    root = RootNode(
        insts=[
            IRLoop(
                name="bad_loop",
                counter_reg="s11",
                n=5,
                body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
            )
        ]
    )
    
    pass_ = UnrollLoopPass()
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    
    with pytest.raises(ValueError, match="is not a general-purpose register"):
        pass_.process(root, ctx)

def test_unroll_rejects_n_reg_equal_counter_reg():
    root = RootNode(
        insts=[
            IRLoop(
                name="bad_loop",
                counter_reg="r1",
                n="r1",
                body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
            )
        ]
    )
    
    pass_ = UnrollLoopPass()
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    
    with pytest.raises(ValueError, match="n_reg and counter_reg are the same register"):
        pass_.process(root, ctx)

def test_unroll_rejects_n_reg_conflict_s15():
    # s15 is reserved for absolute jumps in unroll logic.
    root = RootNode(
        insts=[
            IRLoop(
                name="bad_loop",
                counter_reg="r1",
                n="s15",
                body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
            )
        ]
    )
    
    pass_ = UnrollLoopPass()
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    
    with pytest.raises(ValueError, match="conflicts with reserved address register s15"):
        pass_.process(root, ctx)

def test_unroll_accepts_valid_registers():
    root = RootNode(
        insts=[
            IRLoop(
                name="good_loop",
                counter_reg="r1",
                n="r2",
                body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])])
            )
        ]
    )
    
    pass_ = UnrollLoopPass()
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    
    # Should not raise
    pass_.process(root, ctx)
