"""Point-file reload wire contract and optional optimistic writer admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.context_types import PointFileRevision

from .point import PointFile

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


@dataclass(frozen=True)
class ContextReloadParams:
    """Validated wire request for one selected-point owner file.

    path is the caller's file path, checked by the session owner for absolute
    selected-point membership. expected_revision is the installed base SHA-256,
    not the edited disk hash; None is valid only for a missing aliases base.
    """

    path: Path
    expected_revision: str | None

    @classmethod
    def from_wire(cls, params: Mapping[str, object]) -> ContextReloadParams:
        """Require path and explicit string/null base, or raise INVALID_PARAMS."""
        path = params.get("path")
        if not isinstance(path, str):
            raise RemoteError(ErrorCode.INVALID_PARAMS, "'path' must be a string")
        if "expected_revision" not in params:
            raise RemoteError(ErrorCode.INVALID_PARAMS, "missing 'expected_revision'")
        revision = params["expected_revision"]
        if revision is not None and not isinstance(revision, str):
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "'expected_revision' must be a string or null"
            )
        return cls(Path(path), revision)


def h_context_reload(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> PointFile:
    """Validate/install one edited disk file and return its installed revision.

    params requires path and expected_revision, which names the installed base,
    not the candidate disk hash. Only a missing aliases base accepts null.
    Owner path/validation errors include the path; stale bases propagate
    revision_mismatch. Failures preserve disk and installed state. Holder
    admission belongs to dispatch; no retry, repair or selection change occurs.
    """
    request = ContextReloadParams.from_wire(params)
    file = adapter.context_control.reload_file(request.path, request.expected_revision)
    return {
        "document": file.document,
        "path": str(file.path),
        "revision": file.revision,
    }


def check_writer_revision(
    adapter: RemoteControlAdapter,
    document: Literal["parameters", "library", "aliases"],
    params: Mapping[str, object],
) -> None:
    """Check an optional wire base against the selected document's disk bytes.

    Omission preserves legacy writer admission without a disk check. Explicit
    null means missing aliases, never omission; other documents reject null.
    Malformed values raise INVALID_PARAMS. No point raises no_context. The
    session owner checks disk and raises revision_mismatch without installing,
    publishing or writing. The caller must invoke this before its mutation.
    """
    if "expected_revision" not in params:
        return
    revision = params["expected_revision"]
    if not isinstance(revision, str) and not (
        revision is None and document == "aliases"
    ):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            "'expected_revision' must be a string; only aliases accepts null",
        )
    for file in adapter.context_control.get_file_revisions():
        if file.document == document:
            adapter.context_control.check_file_revision(
                PointFileRevision(document, file.path, revision)
            )
            return
    raise FailedPreconditionError(
        "Select or create a working point first", reason_code="no_context"
    )
