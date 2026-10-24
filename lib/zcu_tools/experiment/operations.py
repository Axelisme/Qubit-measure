"""Shared script operation outcomes and core callable contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import JsonObject
from zcu_tools.experiment.metadata import SoftwareProvenance

if TYPE_CHECKING:
    from .context import RunContext


@dataclass(frozen=True)
class RunIdentity:
    """Stable run_id, native software provenance and optional entry identity.

    entry_id/entry_label are host-supplied workpoint evidence or None. Unknown
    provenance remains None in SoftwareProvenance; it is not inferred here.
    """

    run_id: str
    provenance: SoftwareProvenance
    entry_id: str | None = None
    entry_label: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("RunIdentity.run_id must not be empty")


class MissingCapability(RuntimeError):
    """A required capability is absent; capability identifies the missing port."""

    def __init__(self, capability: str) -> None:
        self.capability = capability
        super().__init__(f"Missing workflow capability: {capability}")


class RunFunction[Cfg: ConfigBase, Result](Protocol):
    """Core experiment callable, including bound Experiment.run methods.

    cfg is an owned, mutable experiment configuration passed positionally; its
    parameter name is unrestricted. context provides borrowed hardware, plots,
    drivers and the cooperative operation cancellation token.
    """

    def __call__(self, cfg: Cfg, /, *, context: RunContext) -> Result: ...


@dataclass(frozen=True)
class Completed[T]:
    """A returned value, partial-delivery flag and facts for one attempt.

    value is fn's payload; env.run uses the existing RunRecord with cfg/data.
    partial_result includes cancellation of the operation or a descendant. It is
    not a scientific completeness check or a persistence guarantee. facts is the
    required OperationFacts for this delivered attempt, also written to journal
    and the enclosing commit. facts.stopped records the operation token's own
    control intent at completion, not descendant cancellation. These flags
    describe different facts and must not be merged.
    """

    value: T
    partial_result: bool
    facts: OperationFacts


@dataclass(frozen=True)
class Failed:
    """A recoverable acquire/analysis failure, without partial data.

    reason describes the cause. Authors decide whether to retry or continue;
    arbitrary framework/build/conversion exceptions instead propagate.
    """

    reason: str


type OperationKind = Literal["run", "call", "setup_device", "wait_until"]
type OperationOutcome = Literal["completed", "partial", "discarded", "failed", "raised"]


@dataclass(frozen=True)
class OperationFacts:
    """One actual executor attempt, retained even when Pause discards its value.

    kind names run/call/setup_device/wait_until. iteration is the nearest
    iteration scope's birth number, or zero outside iterations. attempt counts
    from one within that iteration or the root, including retries. started_at/finished_at are UTC
    worker boundaries. device_snapshot is detached pre-call driver JSON for run,
    otherwise None. outcome is completed, partial, discarded by Pause, failed
    acquire, or raised exception. reason is a diagnostic or None on completion.
    stopped is the operation token's completion-time stop bit, including failed,
    raised and discarded attempts; it is not inferred from later run state.
    """

    kind: OperationKind
    iteration: int
    attempt: int
    started_at: datetime
    finished_at: datetime
    device_snapshot: Mapping[str, JsonObject] | None
    outcome: OperationOutcome
    reason: str | None
    stopped: bool
