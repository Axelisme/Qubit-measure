"""Pass infrastructure for IR transformation pipeline.

Passes are ordered transformations that run on the IR tree:
  FreshLabels → EstimateDurations → UnrollShortLoops → FuseAdjacentDelays
    → AlignBranchDispatch → ValidateInvariants

Each pass is a stateless visitor that rebuilds the IR tree bottom-up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple

from .nodes import (
    IRNode, IRPulse, IRReadout, IRDelay, IRRegOp, IRReadDmem,
    IRCondJump, IRJump, IRLabel, IRNop, IRSoftDelay,
    IRSeq, IRLoop, IRRegLoop, IRBranch,
    IRParallel, IRLeaf, IRComposite
)


@dataclass
class PassConfig:
    """Configuration for all passes in the pipeline."""
    min_body_us: float = 0.09  # threshold for UnrollShortLoops (µs)
    enable_fusion: bool = True  # enable FuseAdjacentDelays
    enable_align_branches: bool = True  # enable AlignBranchDispatch
    extra: Dict[str, Any] = field(default_factory=dict)  # extensible config


@dataclass
class PassCtx:
    """Context passed through the pass pipeline."""
    config: PassConfig = field(default_factory=PassConfig)
    diagnostics: List[str] = field(default_factory=list)  # collected warnings/errors

    def warn(self, msg: str) -> None:
        """Record a diagnostic warning."""
        self.diagnostics.append(f"warning: {msg}")

    def error(self, msg: str) -> None:
        """Record a diagnostic error."""
        self.diagnostics.append(f"error: {msg}")


class Pass(ABC):
    """Base class for IR transformation passes.

    Subclasses override transform() to implement bottom-up IR rewriting.
    The framework handles recursion and tree rebuilding.
    """

    @abstractmethod
    def transform(self, node: IRNode, ctx: PassCtx) -> IRNode:
        """Transform an IR node (assumed children already transformed).

        Args:
            node: IR node to transform
            ctx: pass context (for config and diagnostics)

        Returns:
            transformed IR node (may be same instance, a replacement, or a structural rebuild)
        """
        pass

    def __call__(self, node: IRNode, ctx: PassCtx | None = None) -> IRNode:
        """Entry point: apply pass to IR tree with bottom-up traversal.

        Args:
            node: root IR node
            ctx: optional pass context (creates default if None)

        Returns:
            transformed tree
        """
        if ctx is None:
            ctx = PassCtx()
        return self._visit(node, ctx)

    def _visit(self, node: IRNode, ctx: PassCtx) -> IRNode:
        """Bottom-up traversal: transform children first, then apply transform()."""
        # Transform children based on node type
        if isinstance(node, IRSeq):
            transformed_body = tuple(self._visit(child, ctx) for child in node.body)
            node_with_new_body = IRSeq(body=transformed_body, meta=node.meta)
            return self.transform(node_with_new_body, ctx)

        elif isinstance(node, IRLoop):
            transformed_body = self._visit(node.body, ctx)
            node_with_new_body = IRLoop(name=node.name, n=node.n, body=transformed_body, meta=node.meta)
            return self.transform(node_with_new_body, ctx)

        elif isinstance(node, IRRegLoop):
            transformed_body = self._visit(node.body, ctx)
            node_with_new_body = IRRegLoop(name=node.name, n_reg=node.n_reg, body=transformed_body, meta=node.meta)
            return self.transform(node_with_new_body, ctx)

        elif isinstance(node, IRBranch):
            transformed_arms = tuple(self._visit(arm, ctx) for arm in node.arms)
            node_with_new_arms = IRBranch(compare_reg=node.compare_reg, arms=transformed_arms, meta=node.meta)
            return self.transform(node_with_new_arms, ctx)

        elif isinstance(node, IRParallel):
            transformed_body = tuple(self._visit(child, ctx) for child in node.body)
            node_with_new_body = IRParallel(
                body=transformed_body,
                end_policy=node.end_policy,
                end_index=node.end_index,
                meta=node.meta,
            )
            return self.transform(node_with_new_body, ctx)

        else:
            # Leaf node: no children to transform
            return self.transform(node, ctx)


class PassPipeline:
    """Chains multiple passes in sequence."""

    def __init__(self, passes: List[Pass], config: PassConfig | None = None):
        """Initialize pipeline.

        Args:
            passes: ordered list of passes to apply
            config: pass configuration (shared by all passes)
        """
        self.passes = passes
        self.config = config or PassConfig()

    def __call__(self, node: IRNode) -> Tuple[IRNode, PassCtx]:
        """Apply all passes in sequence.

        Returns:
            (transformed_ir, final_context_with_diagnostics)
        """
        ctx = PassCtx(config=self.config)
        result = node
        for pass_ in self.passes:
            result = pass_(result, ctx)
        return result, ctx
