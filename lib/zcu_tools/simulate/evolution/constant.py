"""Experimental constant planning and inline pinned numerical replay.

This developer seam never changes Engine dispatch or admits production physics.
Only nonempty acquires with one source-input, fixed-flux, constant zero/one-tone
interval per timeline are supported. No prefix is inferred or evolved here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from threading import Event

from .cache import CacheEntry
from .engine import SEGMENT_NUMERIC_VERSION, EvolutionResult, TimelineBatch
from .profiles import OneCoreProfile
from .provider import LegacyKernelAdapter
from .secular import ModelSource
from .shadow import ShadowPlan, ShadowRequest

CONSTANT_PLANNER_REVISION = "experimental-constant-v1"


@dataclass(frozen=True)
class RowCertificate:
    """Explicit guard outcome for one family/row and complete source-input digest.

    row indexes segmented geometry; family names its kernel. accepted is the
    actual numerical guard result, never inferred from cache. input_digest
    identifies source density, models and all controls used for validation.
    """

    row: int
    family: str
    accepted: bool
    input_digest: str


@dataclass(frozen=True)
class PinnedConstantPlan:
    """Experimental numerical assignment and provenance, not an arbitrary graph.

    report retains selected cover, safe incumbent, parameters and profile ID/
    revision. certificates retains actual row acceptance with source-input
    identity. input_digest pins every model, density and segmented control.
    assumptions_digest pins request and immutable cache inventory used in scoring.
    budget pins numerical accuracy. planner_revision and numeric_revision must
    match this implementation on replay. Production admission is always False.
    """

    report: ShadowPlan
    certificates: tuple[RowCertificate, ...]
    input_digest: str
    assumptions_digest: str
    budget: float
    planner_revision: str = CONSTANT_PLANNER_REVISION
    numeric_revision: str = SEGMENT_NUMERIC_VERSION


class ConstantPlanner:
    """Own constant candidate validation and inline pinned-cover execution.

    entries explicitly replaces the constant registry; None uses existing built-in
    declarations. cache_bytes is the replay resident prepare limit, not RSS.
    Logical worker width belongs to ShadowRequest and launches no team.
    Replay is inline only; neither it nor plan changes Engine defaults.
    """

    def __init__(
        self,
        entries: Sequence[LegacyKernelAdapter] | None = None,
        *,
        cache_bytes: int = 0,
    ) -> None:
        """Capture declared constant kernels; reject invalid resident cache size."""
        raise NotImplementedError

    def plan(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        profile: OneCoreProfile,
        request: ShadowRequest,
        *,
        inventory: tuple[CacheEntry, ...] = (),
    ) -> PinnedConstantPlan:
        """Validate guards and score a complete constant acquire using neutral costs.

        request dimension must match source density; its budget controls guards.
        inventory is a serialized immutable cache snapshot for observation only.
        Keys come from kernel.prepare_keys, not caller encoding. t4/t6 must
        accept their entire domains. Unsupported geometry/revisions raise
        ValueError; missing terminals/rejected terminal guards raise RuntimeError.
        Numerical exceptions propagate without reassignment or retry.
        """
        raise NotImplementedError

    def replay(
        self,
        pinned: PinnedConstantPlan,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        cancel: Event,
    ) -> EvolutionResult:
        """Recompute exactly the pinned assignment inline, without cost search.

        Matching source/model/density/controls, selected kernel identities and
        certificates are mandatory. Unsupported planner/numeric/kernel revisions
        or changed inputs raise ValueError before applying anything. Cache capacity
        may differ from planning. Cancellation raises Cancelled, no partial result.
        Build/bind/apply failures propagate; exceedances warn once and keep output.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release replay cache; no worker resources are retained."""
        raise NotImplementedError
