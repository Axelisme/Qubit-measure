"""Explicit registration entry for the user-owned measure catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.gui.app.measure.role_catalog import RoleCatalog


def register_all(registry: Registry, *, roles: RoleCatalog | None = None) -> None:
    """Register this package's measure adapters and optional startup roles.

    ``registry`` is the caller-owned measure adapter Registry. Pass ``roles``
    only at startup to register program/module roles; omit it on source reload.
    Importing this module performs no registration. The skeleton has no catalog
    declarations yet, so this call leaves both caller-owned catalogs unchanged.
    """
    # Declarations arrive with the experiment migration; keep the explicit entry.
    del registry, roles
