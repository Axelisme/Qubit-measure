"""Single source of truth mapping a ``role_id`` to its default factories.

A role has a *blank* factory (md-linked defaults, never a library lookup, never
``None``) and optionally a *ref* factory (prefers a named library entry, falls
back to the blank, and may return ``None`` when ``optional`` and nothing matches).
Both the GUI ``TemplateCatalog`` registration (``registry.py``) and fresh measure cfg
materialization consume this table, so the role vocabulary lives in exactly one
place.

The factory pair for each role is generated from the declarative ``ROLE_TABLE``
(``role_table.py``): ``role_blank`` / ``role_ref`` close over the role's
``RoleDef``.

The context-free measure builder records a module initialization mode. During fresh
cfg materialization, ``ModuleInit.SMART`` calls the *ref* factory (library-aware),
``ModuleInit.INLINE`` forces the *blank* factory, and ``ModuleInit.DISABLED``
materializes ``None`` for an optional ref. ``TemplateCatalog`` always uses the *blank*
factory (creating from a role seeds a fresh entry, it never references an existing
library entry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zcu_tools.gui.app.measure.role_factories import RoleFactorySpec

from zcu_lab.v2._support.measure.defaults.role_table import (
    ROLE_TABLE,
    RoleDef,
    role_blank,
    role_ref,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.cfg import ReferenceValue


def _from_role(role: RoleDef) -> RoleFactorySpec:
    """Build the blank/ref factory pair from a declarative ``RoleDef``."""

    kind: Literal["module", "waveform"] = "waveform" if role.is_waveform else "module"

    def blank(ctx: SessionEnv, _role: RoleDef = role) -> ReferenceValue:
        return role_blank(_role, ctx)

    if role.lib is None:
        return RoleFactorySpec(kind=kind, shape=role.shape, blank=blank)

    def ref(
        ctx: SessionEnv, *, optional: bool = False, _role: RoleDef = role
    ) -> ReferenceValue | None:
        return role_ref(_role, ctx, optional=optional)

    return RoleFactorySpec(kind=kind, shape=role.shape, blank=blank, ref=ref)


# role_id -> factory pair, generated from ROLE_TABLE. Order is informational only
# (TemplateCatalog defines its own dropdown order in registry.py).
ROLE_FACTORIES: dict[str, RoleFactorySpec] = {
    role_id: _from_role(role) for role_id, role in ROLE_TABLE.items()
}
