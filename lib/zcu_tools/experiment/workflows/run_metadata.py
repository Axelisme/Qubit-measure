"""Assemble explicit workflow acquisition records from delivered attempt facts."""

from dataclasses import replace
from typing import Literal

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
    completion: Literal["complete", "partial", "stopped", "unknown"] = "complete",
) -> RunRecord[Cfg, Result]:
    """Return the delivered record with historical workflow acquisition metadata.

    identity supplies startup provenance and the workflow run_id. experiment_tag
    is the persistence declaration's tag; snapshot is the plan's captured entry
    state. Use completed.facts for UTC times, iteration and attempt. completion
    explicitly records acquisition completeness, not save success; it is never
    inferred from control bits. Preserve the record's device_snapshot and other
    author metadata. Never inspect live workflow state. The derived run_id is
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
        completion=completion,
        snapshot=snapshot,
        provenance=identity.provenance,
    )
    document = dict(completed.value.metadata)
    document.update(
        metadata.model_dump(
            mode="json", exclude={"device_snapshot", "parameter_snapshot"}
        )
    )
    return replace(completed.value, metadata=document)
