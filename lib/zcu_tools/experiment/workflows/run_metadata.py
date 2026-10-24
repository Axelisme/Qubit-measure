"""Assemble explicit workflow acquisition records from delivered attempt facts."""

from dataclasses import replace

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot
from zcu_tools.experiment.records import RunRecord

from .models import Completed, RunIdentity


def record_with_metadata[Cfg: ConfigBase, Result](
    completed: Completed[RunRecord[Cfg, Result]],
    *,
    identity: RunIdentity,
    experiment_tag: str,
    snapshot: RunSnapshot,
) -> RunRecord[Cfg, Result]:
    """Return the delivered record with historical workflow acquisition metadata.

    identity supplies startup provenance and the workflow run_id. experiment_tag
    is the persistence declaration's tag; snapshot is the plan's captured entry
    state. Use completed.facts for UTC times, iteration, attempt and pre-call
    devices. completed.partial_result selects stopped completion, including
    descendant cancellation. Never inspect live workflow state. The derived run_id is
    <workflow run_id>-<six-digit iteration>-<attempt>. Invalid metadata raises
    ValueError. The input record is unchanged and large Result arrays stay shared.
    No device queries, provenance recapture, paths or writes occur.
    """
    facts = completed.facts
    metadata = RunMetadata(
        run_id=f"{identity.run_id}-{facts.iteration:06d}-{facts.attempt}",
        experiment=experiment_tag,
        started_at=facts.started_at.isoformat(),
        finished_at=facts.finished_at.isoformat(),
        completion="stopped" if completed.partial_result else "complete",
        snapshot=snapshot,
        provenance=identity.provenance,
        device_snapshot=facts.device_snapshot,
        acquisitions=facts.acquisitions,
    )
    return replace(completed.value, metadata=metadata)
