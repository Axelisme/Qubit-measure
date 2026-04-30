"""Pass infrastructure for IR transformation pipeline.

Passes are ordered transformations that run on the IR tree:
  FreshLabels → FlattenSeq → UnrollShortLoops → FlattenSeq
    → FuseAdjacentDelays → RemoveZeroDelays → ReorderPulseLikeByTime
    → ValidateInvariants

Each pass is a stateless visitor that rebuilds the IR tree bottom-up.
Per-run mutable state (e.g. the FreshLabels label map) lives on PassCtx.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .nodes import IRBranch, IRLoop, IRNode, IRRegLoop, IRSeq


@dataclass
class PassConfig:
    """Configuration for all passes in the pipeline."""

    pmem_budget: int = 0  # pmem inst budget for UnrollShortLoops (must be set explicitly)
    max_unroll_iters: int = 16  # hard upper bound on n for UnrollShortLoops
    enable_fusion: bool = True  # enable FuseAdjacentDelays
    extra: Dict[str, Any] = field(default_factory=dict)  # extensible config


@dataclass
class PassCtx:
    """Context passed through the pass pipeline.

    ``label_map`` / ``label_counter`` are per-run state owned by FreshLabels;
    a fresh PassCtx is built on every PassPipeline call so they reset cleanly.
    """

    config: PassConfig = field(default_factory=PassConfig)
    diagnostics: List[str] = field(default_factory=list)  # collected warnings/errors
    label_map: Dict[str, str] = field(default_factory=dict)
    label_counter: int = 0
    pmem_used: int = 0  # accumulated inst count consumed by greedy unroll
    unroll_info: Dict[int, Any] = field(default_factory=dict)  # id(IRLoop) -> UnrollInfo, set by MarkUnrollInfo

    def warn(self, msg: str) -> None:
        """Record a diagnostic warning."""
        self.diagnostics.append(f"warning: {msg}")

    def error(self, msg: str) -> None:
        """Record a diagnostic error."""
        self.diagnostics.append(f"error: {msg}")

    def fresh_label(self, original: str) -> str:
        """Return a stable rewritten name for a label/jump target.

        Each distinct ``original`` maps to ``_label_{N}`` with monotonic N.
        Repeated calls with the same ``original`` return the same rewritten name,
        so labels and their jump targets stay consistent within one pipeline run.
        """
        if original not in self.label_map:
            self.label_map[original] = f"_label_{self.label_counter}"
            self.label_counter += 1
        return self.label_map[original]


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
            node_with_new_body = IRLoop(
                name=node.name, n=node.n, body=transformed_body, meta=node.meta
            )
            return self.transform(node_with_new_body, ctx)

        elif isinstance(node, IRRegLoop):
            transformed_body = self._visit(node.body, ctx)
            node_with_new_body = IRRegLoop(
                name=node.name, n_reg=node.n_reg, body=transformed_body, meta=node.meta
            )
            return self.transform(node_with_new_body, ctx)

        elif isinstance(node, IRBranch):
            transformed_arms = tuple(self._visit(arm, ctx) for arm in node.arms)
            node_with_new_arms = IRBranch(
                compare_reg=node.compare_reg, arms=transformed_arms, meta=node.meta
            )
            return self.transform(node_with_new_arms, ctx)

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
