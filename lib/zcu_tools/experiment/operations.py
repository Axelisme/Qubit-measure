"""Shared script operation outcomes and core callable contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.metadata import SoftwareProvenance

if TYPE_CHECKING:
    from .context import RunContext


@dataclass(frozen=True)
class RunIdentity:
    """Stable run_id, native software provenance and optional entry identity.

    entry_name/entry_label are host-supplied workpoint evidence or None. Unknown
    provenance remains None in SoftwareProvenance; it is not inferred here.
    """

    run_id: str
    provenance: SoftwareProvenance
    entry_name: str | None = None
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
    """One delivered payload and its immutable operation completion facts.

    value is the callback payload; run_core supplies a RunRecord with cfg/result
    and acquisition metadata. facts is also written to journal and enclosing
    commits. facts.stopped is the operation token's own completion-time control
    bit, not a claim about scientific completeness or descendant cancellation.
    """

    value: T
    facts: OperationFacts


@dataclass(frozen=True)
class Failed:
    """An author-owned recoverable analysis failure.

    reason describes the analysis failure. Authors decide whether to retry or
    continue. Operation exceptions propagate rather than becoming this value.
    """

    reason: str


type OperationKind = Literal["run", "call", "setup_device", "wait_until"]
type OperationOutcome = Literal["completed", "discarded", "raised"]


@dataclass(frozen=True)
class OperationFacts:
    """Facts for one executor invocation; operations are not automatically retried.

    kind names run/call/setup_device/wait_until. iteration is the nearest
    iteration scope's birth number, or zero outside iterations. attempt counts
    from one within that iteration or the root, once per operation. Historical
    records also counted retries from the removed Pause/Resume behavior.
    started_at/finished_at are UTC worker boundaries. Current outcome values are
    completed or raised. discarded is retained for historical Pause records only.
    reason is a diagnostic or None on completion.
    stopped is the operation token's completion-time stop bit, including raised
    attempts and historical discarded attempts; it is not inferred from later run state.
    name is the caller-supplied operation label, or None when unnamed or absent
    in historical records. Readers ignore unknown historical fields.
    """

    kind: OperationKind
    iteration: int
    attempt: int
    started_at: datetime
    finished_at: datetime
    outcome: OperationOutcome
    reason: str | None
    stopped: bool
    name: str | None = None
