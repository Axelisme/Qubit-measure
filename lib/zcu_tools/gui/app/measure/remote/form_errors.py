"""Project measure Form precondition failures into structured wire errors."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import TypedDict

from zcu_tools.gui.form import FormInputRejected, FormPathError
from zcu_tools.gui.form_adapter import LogicalEditError, LogicalSourceError
from zcu_tools.gui.remote.errors import (
    ErrorCode,
    RemoteError,
    remote_error_from_expected,
)

from ..services.form_contracts import CfgPreconditionError


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


def input_rejected_to_remote(exc: FormInputRejected) -> RemoteError:
    """Preserve a declaration refusal and its original typed mutation path."""
    return RemoteError(
        ErrorCode.INVALID_PARAMS,
        str(exc),
        reason="form_input_rejected",
        data={"form_path": list(exc.path)},
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
