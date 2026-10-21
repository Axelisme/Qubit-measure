"""Project measure Form precondition failures into structured wire errors."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypedDict

from zcu_tools.gui.form import FormInputRejected, FormPathError
from zcu_tools.gui.form_adapter import LogicalEditError, LogicalSourceError
from zcu_tools.gui.remote.errors import (
    ErrorCode,
    RemoteError,
    remote_error_from_expected,
)

from ..services.form_contracts import CfgPreconditionError

if TYPE_CHECKING:
    from .form_observation import EditPathHints


class FormError(TypedDict):
    """One validation error; path contains model fields and sequence indexes."""

    path: list[str | int]
    message: str


def cfg_error_to_remote(exc: CfgPreconditionError) -> RemoteError:
    """Keep cfg detail at its wire boundary, separate from other resource guards."""
    mapped = remote_error_from_expected(exc)
    data: dict[str, object] = {}
    if exc.path is not None:
        data["path"] = list(exc.path)
    if exc.edit_index is not None:
        data["edit_index"] = exc.edit_index
    return RemoteError(
        mapped.code, mapped.message, reason=mapped.reason, data=data or None
    )


def input_rejected_to_remote(
    exc: FormInputRejected, *, path_hints: EditPathHints | None = None
) -> RemoteError:
    """Preserve refusal details and optionally include active cfg path hints.

    Keep the existing form_input_rejected taxonomy. Readonly current is raw JSON,
    including null; ordinary owner refusals retain only form_path in data.
    path_hints comes from edit_path_hints for the rejected tab's current Form;
    when supplied, add its canonical editable_paths and selection_paths to data.
    Omitting hints preserves the error data of commands and recipe operations.
    """
    data: dict[str, object] = {"form_path": list(exc.path)}
    if exc.reason == "readonly":
        data.update(reason="readonly", current=exc.current)
    if path_hints is not None:
        data.update(path_hints)
    return RemoteError(
        ErrorCode.INVALID_PARAMS,
        str(exc),
        reason="form_input_rejected",
        data=data,
    )


@contextmanager
def recipe_form_errors() -> Generator[None]:
    """Translate declared logical/cfg failures without reclassifying author bugs.

    Encloses one owner turn. Paths and indexes remain logical, expected source
    failures keep library_source, and unexpected provider exceptions propagate.
    """
    try:
        yield
    except CfgPreconditionError as exc:
        raise cfg_error_to_remote(exc) from exc
    except FormInputRejected as exc:
        raise input_rejected_to_remote(exc) from exc
    except FormPathError as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            str(exc),
            reason="form_path",
            data={"form_path": list(exc.path)},
        ) from exc
    except LogicalEditError as exc:
        data: dict[str, object] = {"path": list(exc.path)}
        if exc.reason_code == "form_path":
            data["form_path"] = list(exc.path)
        if exc.edit_index is not None:
            data["edit_index"] = exc.edit_index
        if exc.selector_path is not None:
            data["selector_path"] = list(exc.selector_path)
        if exc.available_keys is not None:
            data["available_keys"] = list(exc.available_keys)
        raise RemoteError(
            ErrorCode.INVALID_PARAMS,
            str(exc),
            reason=exc.reason_code,
            data=data,
        ) from exc
    except LogicalSourceError as exc:
        mapped = remote_error_from_expected(exc)
        raise RemoteError(
            mapped.code,
            mapped.message,
            reason=mapped.reason,
            data={
                "path": list(exc.path),
                "selector_path": list(exc.selector_path),
                "key": exc.key,
                "source_kind": exc.source_kind,
            },
        ) from exc
