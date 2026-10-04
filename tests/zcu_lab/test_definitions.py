"""Validate explicit registration through the framework catalog seam."""

import pytest
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.role_catalog import RoleCatalog

from zcu_lab.definitions import register_all


@pytest.mark.parametrize("startup", [False, True], ids=["reload", "startup"])
def test_registration_produces_valid_catalog(startup: bool) -> None:
    registry = Registry()
    roles = RoleCatalog() if startup else None

    register_all(registry, roles=roles)

    registry.validate()
