"""context.ml_list_roles / context.ml_create_from_role dispatch handlers (role templates)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from ._helpers import dispatch_handler as _dispatch  # noqa: E402


def test_list_roles_returns_catalog_meta():
    ctrl = MagicMock()
    ctrl.get_role_catalog.return_value.list_meta.return_value = [
        {"role_id": "res_probe", "label": "Resonator probe", "item_kind": "module"},
    ]
    res = _dispatch(ctrl, "context.ml_list_roles", {})
    assert res["roles"] == [
        {"role_id": "res_probe", "label": "Resonator probe", "item_kind": "module"}
    ]


def test_list_roles_no_catalog_precondition():
    ctrl = MagicMock()
    ctrl.get_role_catalog.side_effect = RuntimeError("No role catalog is wired up.")
    with pytest.raises(RemoteError) as exc:
        _dispatch(ctrl, "context.ml_list_roles", {})
    assert exc.value.code is ErrorCode.PRECONDITION_FAILED


def test_create_from_role_drives_controller():
    # item_kind is now DERIVED from role_id (the agent no longer passes it): the
    # handler reads get_role_catalog().get(role_id).item_kind, so the catalog mock
    # must report the role's kind.
    ctrl = MagicMock()
    ctrl.get_role_catalog.return_value.get.return_value.item_kind = "module"
    _dispatch(
        ctrl,
        "context.ml_create_from_role",
        {"role_id": "res_probe", "name": "my_ro"},
    )
    ctrl.create_from_role.assert_called_once_with("module", "res_probe", "my_ro")


def test_create_from_role_unknown_role_invalid_params():
    ctrl = MagicMock()
    ctrl.create_from_role.side_effect = KeyError("Role 'x' not found")
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "context.ml_create_from_role",
            {"item_kind": "module", "role_id": "x", "name": "n"},
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_create_from_role_no_context_precondition():
    ctrl = MagicMock()
    ctrl.create_from_role.side_effect = RuntimeError("No experiment context.")
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "context.ml_create_from_role",
            {"item_kind": "module", "role_id": "res_probe", "name": "n"},
        )
    assert exc.value.code is ErrorCode.PRECONDITION_FAILED
