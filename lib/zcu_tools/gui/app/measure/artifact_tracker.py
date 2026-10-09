"""Owner-thread section declarations and latest multi-output save projection."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from zcu_tools.experiment.run_saving import ArtifactRef, SaveReport
from zcu_tools.resources.entry import ArtifactKey, OutputFormat, SaveLayout


class SaveStatus(str, Enum):
    """Current artifact state, derived from its latest outputs and annotation."""

    NOT_SAVED = "not_saved"
    SAVED = "saved"
    PARTIALLY_SAVED = "partially_saved"
    UNSAVED_CHANGES = "unsaved_changes"


@dataclass(frozen=True)
class ArtifactDeclaration:
    """One opaque artifact ref and ordered, nonempty, unique member identities.

    Every key must belong to ref. No payload or live pane resource is retained.
    Invalid membership or duplicate keys raises ValueError at construction.
    """

    ref: ArtifactRef
    members: tuple[ArtifactKey, ...]

    def __post_init__(self) -> None:
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("artifact requires nonempty unique member keys")
        if any(
            (key.section, key.name) != (self.ref.section, self.ref.name)
            for key in self.members
        ):
            raise ValueError("member section/artifact must match ref")


@dataclass(frozen=True)
class OutputSnapshot:
    """One planned or attempted destination.

    format is the resource owner's format token; path is its absolute path.
    status is reserved before saving, then saved or failed from the latest report.
    error is None except for failed outputs, whose diagnostic comes from the report.
    """

    format: OutputFormat
    path: Path
    status: Literal["reserved", "saved", "failed"]
    error: str | None


@dataclass(frozen=True)
class MemberSnapshot:
    """A complete member identity and its ordered destination projections."""

    key: ArtifactKey
    outputs: tuple[OutputSnapshot, ...]


@dataclass(frozen=True)
class ArtifactSnapshot:
    """An opaque artifact identity, ordered members, and aggregate SaveStatus."""

    ref: ArtifactRef
    members: tuple[MemberSnapshot, ...]
    status: SaveStatus


@dataclass(frozen=True)
class ArtifactSectionSnapshot:
    """One declared section name and its artifacts in producer order."""

    section: str
    artifacts: tuple[ArtifactSnapshot, ...]


class ArtifactTracker:
    """Track fixed-pane declarations and latest reports without retaining payloads.

    Call on the State owner thread. Sections retain first-declaration order;
    artifacts and members retain producer order. No method writes files, loads
    ledger history or inspects acquisition completion. Snapshots are immutable.
    """

    def __init__(self) -> None:
        self._sections: dict[str, tuple[ArtifactSnapshot, ...]] = {}

    def replace_section(
        self,
        section: str,
        artifacts: tuple[ArtifactDeclaration, ...],
        *,
        layout: SaveLayout,
    ) -> None:
        """Replace the entire section with Not saved declarations and planned paths.

        Empty artifacts removes the section. Duplicate refs, mismatched section
        or layout errors fail before mutation, preserving the previous section.
        Path planning happens only here; reading a snapshot never reserves paths.
        """
        if any(artifact.ref.section != section for artifact in artifacts):
            raise ValueError("artifact section must match replaced section")
        if len({artifact.ref for artifact in artifacts}) != len(artifacts):
            raise ValueError("section artifact refs must be unique")
        snapshots = tuple(
            ArtifactSnapshot(
                artifact.ref,
                tuple(
                    MemberSnapshot(
                        key,
                        tuple(
                            OutputSnapshot(output.format, output.path, "reserved", None)
                            for output in layout.outputs(key)
                        ),
                    )
                    for key in artifact.members
                ),
                SaveStatus.NOT_SAVED,
            )
            for artifact in artifacts
        )
        if snapshots:
            self._sections[section] = snapshots
        else:
            self._sections.pop(section, None)

    def clear(self) -> None:
        """Discard all declarations, planned paths, dirty state and report projections."""
        self._sections.clear()

    def select(
        self, refs: tuple[ArtifactRef, ...] | None = None
    ) -> tuple[ArtifactDeclaration, ...]:
        """Return declarations, including Saved artifacts, in requested order.

        None selects all in section/producer order. Empty, repeated or unknown
        explicit refs (or an empty all-selection) raise ValueError, without mutation.
        """
        available = {
            artifact.ref: ArtifactDeclaration(
                artifact.ref, tuple(member.key for member in artifact.members)
            )
            for artifacts in self._sections.values()
            for artifact in artifacts
        }
        selected = tuple(available) if refs is None else refs
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Save requires a nonempty unique artifact set")
        if any(ref not in available for ref in selected):
            raise ValueError("Save requires currently declared artifact refs")
        return tuple(available[ref] for ref in selected)

    def apply_report(self, report: SaveReport) -> None:
        """Replace selected artifacts' outputs with the complete latest Save report.

        Report membership follows the shared saver contract; unselected artifacts
        stay unchanged. Actual paths replace reservations. Prior successful outputs
        and dirty state are not carried into the new selected-artifact projection.
        """
        selected = {
            ArtifactRef(item.key.section, item.key.name) for item in report.outputs
        }
        for section, artifacts in self._sections.items():
            updated = []
            for artifact in artifacts:
                if artifact.ref not in selected:
                    updated.append(artifact)
                    continue
                members = tuple(
                    MemberSnapshot(
                        member.key,
                        tuple(
                            OutputSnapshot(
                                item.output.format,
                                item.output.path,
                                item.status,
                                item.error,
                            )
                            for item in report.outputs
                            if item.key == member.key
                        ),
                    )
                    for member in artifact.members
                )
                outcomes = [
                    output.status for member in members for output in member.outputs
                ]
                if outcomes and all(status == "saved" for status in outcomes):
                    status = SaveStatus.SAVED
                elif "saved" in outcomes:
                    status = SaveStatus.PARTIALLY_SAVED
                else:
                    status = SaveStatus.NOT_SAVED
                updated.append(ArtifactSnapshot(artifact.ref, members, status))
            self._sections[section] = tuple(updated)

    def mark_dirty(self, ref: ArtifactRef) -> None:
        """Mark a declared artifact with any successful output Unsaved changes.

        No successful output leaves its status unchanged. Paths and diagnostics
        stay unchanged. Unknown refs raise ValueError; no declaration is created.
        """
        artifacts = self._sections.get(ref.section, ())
        if not any(artifact.ref == ref for artifact in artifacts):
            raise ValueError("Dirty ref must identify a declared artifact")
        self._sections[ref.section] = tuple(
            ArtifactSnapshot(artifact.ref, artifact.members, SaveStatus.UNSAVED_CHANGES)
            if artifact.ref == ref
            and any(
                output.status == "saved"
                for member in artifact.members
                for output in member.outputs
            )
            else artifact
            for artifact in artifacts
        )

    def snapshot(self) -> tuple[ArtifactSectionSnapshot, ...]:
        """Read immutable sections without invoking adapters, planning paths or I/O."""
        return tuple(
            ArtifactSectionSnapshot(section, artifacts)
            for section, artifacts in self._sections.items()
        )
