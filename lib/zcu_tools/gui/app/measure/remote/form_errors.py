"""Project measure Form precondition failures into structured wire errors."""

from zcu_tools.gui.remote.errors import RemoteError, remote_error_from_expected

from ..services.form_contracts import CfgPreconditionError


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
