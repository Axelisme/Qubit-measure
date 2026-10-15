"""Structured native save projection shared by GUI remote and MCP captures.

The GUI owns output identity, formats, destinations and success. These models
only carry that projection across the wire; they never plan paths or stat files.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from zcu_tools.gui.app.measure.artifacts import (
    ArtifactSectionSnapshot,
    SaveStatus,
)

if TYPE_CHECKING:
    from zcu_tools.experiment.artifacts import SaveReport


ArtifactStatusWire = Literal["Not saved", "Saved", "Partially saved", "Unsaved changes"]
STATUS_WIRE: Mapping[SaveStatus, ArtifactStatusWire] = {
    SaveStatus.NOT_SAVED: "Not saved",
    SaveStatus.SAVED: "Saved",
    SaveStatus.PARTIALLY_SAVED: "Partially saved",
    SaveStatus.UNSAVED_CHANGES: "Unsaved changes",
}


class SaveArtifactRefWire(BaseModel):
    """One selected opaque artifact, never an individual member or output.

    section and name preserve the native ArtifactRef fields. Both must be
    nonblank safe path segments; the native owner performs that validation.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    section: str
    name: str


class SaveOutputWire(BaseModel):
    """One owner destination in the latest attempt.

    format is the owning writer token; path is an absolute local file path.
    status is reserved before an attempt, then saved or failed. error is the
    owner diagnostic for failed outputs and None otherwise.
    """

    model_config = ConfigDict(frozen=True)
    format: Literal["data_h5", "labber", "png", "json"]
    path: str
    status: Literal["reserved", "saved", "failed"]
    error: str | None = None


class SaveArtifactWire(BaseModel):
    """One whole artifact, preserving every member and output.

    status is the GUI aggregate for the latest report or dirty annotation.
    members maps data, figure:<name> or analysis:<pane> to ordered outputs;
    member kinds remain distinct even when their names coincide.
    """

    model_config = ConfigDict(frozen=True)
    status: ArtifactStatusWire
    members: dict[str, list[SaveOutputWire]]


class SaveSnapshotWire(BaseModel):
    """Detached owning save evidence, not a second persistence policy.

    Snapshot reads expose a confirmed run_id only after a persistent output
    succeeds; before that it is None. Save replies expose the attempted Run's
    identity even when every output fails. artifacts maps opaque section names
    to opaque artifact names and their complete members. An empty map means no
    capture/declaration. Output status, not run_id, establishes Save success.
    """

    model_config = ConfigDict(frozen=True)
    run_id: str | None = None
    artifacts: dict[str, dict[str, SaveArtifactWire]] = Field(default_factory=dict)


def project_save_snapshot(
    sections: tuple[ArtifactSectionSnapshot, ...], *, run_id: str | None
) -> SaveSnapshotWire:
    """Encode committed owner sections without mutation, I/O or path planning.

    sections comes from TabSnapshot.artifacts, and run_id from its Run pane.
    Every output retains its format/status/error and producer order. No output
    success is inferred from a reservation or a previously saved destination.
    """
    artifacts: dict[str, dict[str, SaveArtifactWire]] = {}
    for section in sections:
        group: dict[str, SaveArtifactWire] = {}
        for artifact in section.artifacts:
            members: dict[str, list[SaveOutputWire]] = {}
            for member in artifact.members:
                key = member.key
                name = (
                    "data"
                    if key.member == "data"
                    else f"{key.member}:{key.member_name}"
                )
                members[name] = [
                    SaveOutputWire(
                        format=output.format,
                        path=str(output.path),
                        status=output.status,
                        error=output.error,
                    )
                    for output in member.outputs
                ]
            group[artifact.ref.name] = SaveArtifactWire(
                status=STATUS_WIRE[artifact.status], members=members
            )
        artifacts[section.section] = group
    return SaveSnapshotWire(run_id=run_id, artifacts=artifacts)


def project_save_report(
    report: SaveReport,
    sections: tuple[ArtifactSectionSnapshot, ...],
) -> SaveSnapshotWire:
    """Encode the attempted Run identity and artifacts in one synchronous report.

    report comes directly from SaveControl and retains the acquisition's run_id
    even if every output fails. sections is the owner's committed read model
    immediately after that call and supplies aggregate output status. Unselected
    declarations and previous outputs are not part of this response. This
    performs no RPC, path planning or I/O.
    """
    selected = {(item.key.section, item.key.name) for item in report.outputs}
    groups = []
    for section in sections:
        artifacts = tuple(
            artifact
            for artifact in section.artifacts
            if (artifact.ref.section, artifact.ref.name) in selected
        )
        if artifacts:
            groups.append(ArtifactSectionSnapshot(section.section, artifacts))
    return project_save_snapshot(tuple(groups), run_id=report.run_id)
