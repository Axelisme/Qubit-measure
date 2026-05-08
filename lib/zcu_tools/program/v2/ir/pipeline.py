from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field

from .factory import IRLexer, IRParser
from .instructions import Instruction
from .node import BasicBlockNode, RootNode
from .traversal import walk_basic_blocks
from .labels import Label


@dataclass
class PipeLineConfig:
    disable_all_opt: bool = False
    enable_unroll_loop: bool = True
    enable_dead_label: bool = True
    pmem_capacity: int | None = None
    pmem_budget: int | None = None

    # Hard cap on the unroll factor k. For register-driven loops k is also
    # rounded down to the nearest power of 2 (Phase 8D).
    max_unroll_factor: int = 32

    # Maximum number of REG_WR words allowed in the dispatch shift-add
    # sequence. If the body_words multiply cannot fit, register-driven
    # unroll falls back to no-unroll.
    max_dispatch_words: int = 8

    # Unified cycle cost model
    cost_default: int = 1
    cost_wmem: int = 2
    cost_dmem: int = 2
    cost_jump_flush: int = 40


# Global default configuration for quick debugging and toggling optimization options.
DEFAULT_PIPELINE_CONFIG = PipeLineConfig()


@dataclass
class PipeLineContext:
    config: PipeLineConfig = field(default_factory=PipeLineConfig)
    pmem_size: int | None = None


# ---------------------------------------------------------------------------
# Pass interfaces
# ---------------------------------------------------------------------------


class AbsIRPass(ABC):
    """Structural IR pass: transforms a whole RootNode tree."""

    @abstractmethod
    def process(self, ir: RootNode, ctx: PipeLineContext) -> RootNode: ...


class AbsLinearPass(ABC):
    """Straight-line optimization pass that operates on a single BasicBlockNode.

    Implementations receive the whole block and may modify any field of the
    ``BasicBlockNode`` in-place, as long as the result still represents a
    valid basic block:

    - ``block.insts`` must not contain ``LabelInst`` or ``JumpInst``.
    - ``block.branch`` must remain either ``None`` or a terminal ``JumpInst``.
    - Block-local transformations must not depend on neighboring blocks.

    When ``block.fix_addr_size`` is True the total emitted program-memory word
    count of the block (``insts`` plus optional ``branch``) must be preserved.
    Passes that normally delete instructions must replace removed words with
    ``NopInst`` padding instead.
    """

    @abstractmethod
    def process_block(self, block: BasicBlockNode) -> None: ...


# ---------------------------------------------------------------------------
# IRPipeLine: three-stage structural pipeline
# ---------------------------------------------------------------------------


def _block_addr_words(block: BasicBlockNode) -> int:
    total = sum(inst.addr_inc for inst in block.insts)
    if block.branch is not None:
        total += block.branch.addr_inc
    return total


def _validate_linear_pass_result(
    block: BasicBlockNode, *, before_addr_words: int | None
) -> None:
    block.__post_init__()
    if before_addr_words is not None and _block_addr_words(block) != before_addr_words:
        raise ValueError(
            "AbsLinearPass violated fix_addr_size invariant: "
            "block program-memory word count changed."
        )


def _run_linear_passes(passes: list[AbsLinearPass], ir: RootNode) -> None:
    for block in walk_basic_blocks(ir):
        for lp in passes:
            before_addr_words = (
                _block_addr_words(block) if block.fix_addr_size else None
            )
            lp.process_block(block)
            _validate_linear_pass_result(block, before_addr_words=before_addr_words)


class IRPipeLine:
    """Three-stage IR optimization pipeline.

    Stage 1 — Pre-LIR  : ``linear_passes`` applied to every BasicBlockNode
                          before any structural pass.
    Stage 2 — HIR       : sequence of ``AbsIRPass`` instances that may
                          restructure the tree (e.g. loop unrolling).
    Stage 3 — Post-LIR : same ``linear_passes`` re-applied after all
                          structural passes.

    Each AbsLinearPass handles fix_addr_size internally, so the same pass list
    is safe to run both before and after structural changes.
    """

    def __init__(
        self,
        config: PipeLineConfig,
        linear_passes: list[AbsLinearPass],
        ir_passes: list[AbsIRPass],
    ) -> None:
        self.config = config
        self.linear_passes = linear_passes
        self.ir_passes = ir_passes

    def __call__(
        self, insts: list[Instruction]
    ) -> tuple[list[Instruction], PipeLineContext]:
        ctx = PipeLineContext(config=self.config, pmem_size=self.config.pmem_capacity)
        if self.config.disable_all_opt:
            return insts, ctx

        lexer = IRLexer()
        parser = IRParser(pmem_size=self.config.pmem_capacity)

        blocks = lexer.lex(insts)
        ir = parser.parse(blocks)

        # Stage 1: Pre-LIR
        _run_linear_passes(self.linear_passes, ir)

        # Stage 2: HIR structural passes, with linear passes around each
        for _pass in self.ir_passes:
            ir = _pass.process(ir, ctx)
            _run_linear_passes(self.linear_passes, ir)

        opt_blocks = parser.unparse(ir)
        opt_insts = lexer.flatten(opt_blocks)

        return opt_insts, ctx


# ---------------------------------------------------------------------------
# Default pipeline factory
# ---------------------------------------------------------------------------


def make_default_pipeline(pmem_capacity: int) -> IRPipeLine:
    from .passes import (
        BlockMergePass,
        BranchEliminationPass,
        DeadLabelEliminationPass,
        DeadTestEliminationLinear,
        DeadWriteEliminationLinear,
        IncRegMergeLinear,
        LoopConditionMergeLinear,
        TimedMergeLinear,
        UnrollSmallLoopPass,
        ZeroDelayDCELinear,
    )

    config = deepcopy(DEFAULT_PIPELINE_CONFIG)
    if config.pmem_capacity is None:
        config.pmem_capacity = pmem_capacity
    if config.pmem_budget is None:
        config.pmem_budget = int(0.8 * pmem_capacity)

    return IRPipeLine(
        config=config,
        linear_passes=[
            ZeroDelayDCELinear(),
            TimedMergeLinear(),
            IncRegMergeLinear(),
            LoopConditionMergeLinear(),
            DeadWriteEliminationLinear(),
            DeadTestEliminationLinear(),
        ],
        ir_passes=[
            UnrollSmallLoopPass(),
            DeadLabelEliminationPass(),
            BranchEliminationPass(),
            BlockMergePass(),
        ],
    )
