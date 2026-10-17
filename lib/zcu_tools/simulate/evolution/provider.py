"""Developer seam for declarations and unchanged legacy classification.

Use segment(source, batches) to obtain Geometry, classify it, and pass Selection
for one timeline's deferred rows. Dispatch preserves guard products for ordinary
Kernel.apply. This seam does not execute prefixes, schedule work or own a cache.
Engine remains the production execution entry point.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from ._dispatch import Dispatch, Selection, classify
from ._planning import Geometry
from ._planning import plan as segment
from .declarations import KernelDeclaration
from .kernel import Kernel, SegmentTable
from .secular import ModelSource

__all__ = [
    "Dispatch",
    "Geometry",
    "LegacyKernelAdapter",
    "LegacyProvider",
    "Selection",
    "segment",
]


@dataclass(frozen=True)
class LegacyKernelAdapter:
    """Explicitly pair a legacy Kernel with caller-owned numeric declarations.

    kernel needs only the existing Kernel protocol; no reflection fills missing
    metadata. declaration identity must match kernel.name and its build handoff.
    Pairing mismatches raise ValueError before any guard or preparation runs.
    """

    kernel: Kernel
    declaration: KernelDeclaration

    def __post_init__(self) -> None:
        if self.declaration.identity.name != self.kernel.name:
            raise ValueError("declaration identity must match legacy kernel name")
        if self.declaration.guard.builds_model != self.kernel.guard_builds:
            raise ValueError("declaration handoff must match legacy guard_builds")


class LegacyProvider:
    """Discover declared candidates and delegate selection to the existing tree.

    entries replaces the entire registry, with no implicit fallback. None uses
    built-in declarations. eligible excludes by declaration only, never validates
    a candidate. classify deliberately uses legacy guards/order, not eligible or
    declared costs, so old selection/errors and relative scheduling weights stay
    unchanged. Duplicate names and invalid budget raise ValueError.
    """

    def __init__(self, entries: Sequence[LegacyKernelAdapter] | None = None) -> None:
        """Capture explicit adapters, or the registered built-in families."""
        if entries is None:
            from .kernels import REGISTRY

            entries = tuple(LegacyKernelAdapter(k, k.declaration) for k in REGISTRY)
        self._entries = tuple(entries)
        names = [entry.kernel.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("kernel names must be unique")
        self._kernels: Mapping[str, Kernel] = MappingProxyType(
            {entry.kernel.name: entry.kernel for entry in entries}
        )

    def entry(self, name: str) -> LegacyKernelAdapter:
        """Return a declared family for ordinary prepare/bind/apply; unknown names raise KeyError."""
        for entry in self._entries:
            if entry.kernel.name == name:
                return entry
        raise KeyError(name)

    def eligible(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> Mapping[str, NDArray[np.int64]]:
        """Return eligible global rows per family, including empty selections.

        This table-only discovery never builds models, calls guards, reads an
        input certificate or interprets Unknown input as known. It does not
        imply numerical validity or impose an ordering on legacy classification.
        """
        return MappingProxyType(
            {
                entry.kernel.name: rows[entry.declaration.eligibility.mask(table, rows)]
                for entry in self._entries
            }
        )

    def classify(
        self,
        geometry: Geometry,
        source: ModelSource,
        budget: float,
        *,
        selection: Selection | None = None,
    ) -> Dispatch:
        """Delegate shared or deferred classification, retaining opaque handoffs.

        geometry is segment(source, batches), or engine geometry with computed
        prefix certificates. budget is a finite positive per-segment allowance.
        selection=None defers local builders. Selection resolves one timeline,
        using the original Dispatch.shaped and rejected masks. Returned costs
        remain legacy relative weights, not calibrated times. Missing required
        names, terminal rejection and guard exceptions propagate unchanged.
        Prepare/apply are still the ordinary Kernel seam; never retry failures.
        """
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("budget must be finite and positive")
        return classify(geometry, self._kernels, source, budget, selection=selection)
