"""ml.list_roles / ml.create_from_role dispatch handlers (role templates)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.services.remote.errors import ErrorCode, RemoteError


def _dispatch(ctrl, method, params):
    return METHOD_REGISTRY[method].handler(ctrl, params)


def test_list_roles_returns_catalog_meta():
    ctrl = MagicMock()
    ctrl.get_role_catalog.return_value.list_meta.return_value = [
        {"role_id": "res_probe", "label": "Resonator probe", "item_kind": "module"},
    ]
    res = _dispatch(ctrl, "ml.list_roles", {})
    assert res["roles"] == [
        {"role_id": "res_probe", "label": "Resonator probe", "item_kind": "module"}
    ]


def test_list_roles_no_catalog_precondition():
    ctrl = MagicMock()
    ctrl.get_role_catalog.side_effect = RuntimeError("No role catalog is wired up.")
    with pytest.raises(RemoteError) as exc:
        _dispatch(ctrl, "ml.list_roles", {})
    assert exc.value.code is ErrorCode.PRECONDITION_FAILED


def test_create_from_role_drives_controller():
    ctrl = MagicMock()
    _dispatch(
        ctrl,
        "ml.create_from_role",
        {"item_kind": "module", "role_id": "res_probe", "name": "my_ro"},
    )
    ctrl.create_from_role.assert_called_once_with("module", "res_probe", "my_ro")


def test_create_from_role_unknown_role_invalid_params():
    ctrl = MagicMock()
    ctrl.create_from_role.side_effect = KeyError("Role 'x' not found")
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "ml.create_from_role",
            {"item_kind": "module", "role_id": "x", "name": "n"},
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMS


def test_create_from_role_no_context_precondition():
    ctrl = MagicMock()
    ctrl.create_from_role.side_effect = RuntimeError("No experiment context.")
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "ml.create_from_role",
            {"item_kind": "module", "role_id": "res_probe", "name": "n"},
        )
    assert exc.value.code is ErrorCode.PRECONDITION_FAILED
