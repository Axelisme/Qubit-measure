"""Validated caller registrations in selector order, without native run state."""

from __future__ import annotations

from dataclasses import dataclass

from zcu_tools.gui.expected_error import InvalidInputError

from .attachment import Attachment


@dataclass(frozen=True)
class WorkflowCatalog[C]:
    """Current coherent attachments in selector order, without run/draft state.

    attachments pairs native science with independent current forms. The first
    attachment is the default for Start without name (caller-declared order).
    Names must be unique and at least one attachment must exist. The app selector
    and optional Start name consume this same pairing; Registry has no lookup.
    New Starts use current callables; Controller's existing run binding stays
    fixed across catalog replacement. No live resources are owned here.
    """

    attachments: tuple[Attachment[C], ...]

    def __post_init__(self) -> None:
        """Reject empty catalogs or duplicate declared names before publication."""
        if not self.attachments:
            raise ValueError("Workflow catalog must contain an attachment")
        names = self.names
        if len(names) != len(set(names)):
            raise ValueError("Workflow catalog names must be unique")

    @property
    def names(self) -> tuple[str, ...]:
        """Return selector names in app-declared order, without Registry lookup."""
        return tuple(attachment.name for attachment in self.attachments)

    def select(self, name: str | None = None) -> Attachment[C]:
        """Return a current pairing; None selects default, unknown name raises InvalidInputError."""
        if name is None:
            return self.attachments[0]
        for attachment in self.attachments:
            if attachment.name == name:
                return attachment
        raise InvalidInputError(
            f"Unknown workflow: {name}", reason_code="unknown_workflow"
        )
