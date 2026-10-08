"""Structured native save projection shared by GUI remote and MCP captures.

The GUI owns output identity, formats, destinations and success. These models
only carry that projection across the wire; they never plan paths or stat files.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from zcu_tools.gui.app.measure.artifact_tracker import SaveStatus

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.artifact_tracker import ArtifactSectionSnapshot
    from zcu_tools.gui.app.measure.services.ports import SaveDestination


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

    run_id is confirmed only after a persistent output succeeds and is None
    beforehand. artifacts maps opaque section names to opaque artifact names
    and their complete members. An empty map means no capture/declaration.
    A later failed resave can retain run_id but replaces selected outputs.
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


def project_save_reservations(
    destinations: tuple[SaveDestination, ...], *, run_id: str | None
) -> SaveSnapshotWire:
    """Project this submission's complete reservations, never earlier successes.

    destinations comes from SaveDataSubmission or SaveArtifactsSubmission;
    run_id is the currently confirmed identity, not the reserved internal ID.
    This does no path planning or I/O. Native terminal reads replace these
    outputs with the report's actual paths/status/error.
    """
    artifacts: dict[str, dict[str, SaveArtifactWire]] = {}
    for destination in destinations:
        key = destination.key
        section = artifacts.setdefault(key.section, {})
        artifact = section.setdefault(
            key.name, SaveArtifactWire(status="Not saved", members={})
        )
        member = "data" if key.member == "data" else f"{key.member}:{key.member_name}"
        artifact.members.setdefault(member, []).append(
            SaveOutputWire(
                format=destination.output.format,
                path=str(destination.output.path),
                status="reserved",
            )
        )
    return SaveSnapshotWire(run_id=run_id, artifacts=artifacts)
