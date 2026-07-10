"""Single source of truth mapping a ``role_id`` to its default factories.

A role has a *blank* factory (md-linked defaults, never a library lookup, never
``None``) and optionally a *ref* factory (prefers a named library entry, falls
back to the blank, and may return ``None`` when ``optional`` and nothing matches).
Both the GUI ``RoleCatalog`` registration (``registry.py``) and the value-assembly
``CfgBuilder`` consume this table, so the role vocabulary lives in exactly one
place.

The factory pair for each role is generated from the declarative ``ROLE_TABLE``
(``role_table.py``): ``role_blank`` / ``role_ref`` close over the role's
``RoleDef``.

The ``CfgBuilder.role()`` verb selects an initialization mode: ``RoleInit.ADOPT``
calls the *ref* factory (library-aware, the common case), ``RoleInit.INLINE`` forces
the *blank* factory (e.g. a twotone readout that must stay inline, never adopting
a library ``readout_dpm``), and ``RoleInit.DISABLED`` takes the *ref* factory's
optional path (library miss → ``None``). ``RoleCatalog`` always uses the *blank*
factory (creating from a role seeds a fresh entry, it never references an
existing library entry).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .role_table import ROLE_TABLE, RoleDef, role_blank, role_ref

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import (
        ExpContext,
        ReferenceValue,
    )

# A blank factory always produces a value (never None); a ref factory's optional
# path may return None (the disabled-optional ref, ADR-0010).
BlankFactory = Callable[["ExpContext"], "ReferenceValue"]
RefFactory = Callable[..., "ReferenceValue | None"]


@dataclass(frozen=True)
class RoleFactorySpec:
    """The factory pair for one role.

    ``ref`` is ``None`` for roles that have no library-aware variant (a concrete
    shape that is always built inline — e.g. ``direct_readout``, ``bath_reset``).
    Asking such a role for a ref/optional mount is a Fast-Fail at the call site.
    """

    kind: Literal["module", "waveform"]
    blank: BlankFactory
    ref: RefFactory | None = None


def _from_role(role: RoleDef) -> RoleFactorySpec:
    """Build the blank/ref factory pair from a declarative ``RoleDef``."""

    kind: Literal["module", "waveform"] = "waveform" if role.is_waveform else "module"

    def blank(ctx: ExpContext, _role: RoleDef = role) -> ReferenceValue:
        return role_blank(_role, ctx)

    if role.lib is None:
        return RoleFactorySpec(kind=kind, blank=blank)

    def ref(
        ctx: ExpContext, *, optional: bool = False, _role: RoleDef = role
    ) -> ReferenceValue | None:
        return role_ref(_role, ctx, optional=optional)

    return RoleFactorySpec(kind=kind, blank=blank, ref=ref)


# role_id -> factory pair, generated from ROLE_TABLE. Order is informational only
# (RoleCatalog defines its own dropdown order in registry.py).
ROLE_FACTORIES: dict[str, RoleFactorySpec] = {
    role_id: _from_role(role) for role_id, role in ROLE_TABLE.items()
}
