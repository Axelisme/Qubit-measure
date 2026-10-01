"""Headless ownership of the cfg resource associated with each measure tab.

Composition owns this registry. Frontends receive resource-bound CfgEditing;
execution receives CfgAcceptance. No editable field tree is exposed or mirrored.
Tab operation guards remain with the application lifecycle owner.
"""

from collections.abc import Callable
from typing import Protocol

from zcu_tools.gui.cfg.lowering import RangeFactory
from zcu_tools.gui.cfg.model import CfgSchema
from zcu_tools.gui.cfg.resource import (
    CfgAcceptance,
    CfgEditing,
    CfgObservation,
    CfgPreconditionError,
    CfgPreconditionReason,
    CfgResolution,
    CfgResource,
)


class TabCfgLookup(Protocol):
    def lookup(self, tab_id: str) -> CfgEditing: ...


class TabCfgResources:
    def __init__(
        self,
        *,
        resolution: Callable[[], CfgResolution],
        make_range: RangeFactory,
        mutation_allowed: Callable[[str], bool],
    ) -> None:
        self._resolution = resolution
        self._make_range = make_range
        self._mutation_allowed = mutation_allowed
        self._resources: dict[str, CfgResource] = {}

    def create(
        self,
        tab_id: str,
        defaults: Callable[[], CfgSchema],
        *,
        initial: CfgSchema | None = None,
    ) -> CfgResource:
        """Prepare before registering; restore does not execute defaults.

        The concrete result belongs to the creating application flow, not the
        frontend. A failed preparation leaves no tab association behind.
        """
        self._require_not_notifying()
        if not tab_id:
            raise ValueError("tab identity must not be empty")
        if tab_id in self._resources:
            raise RuntimeError(f"Tab {tab_id!r} already owns a cfg resource")
        resource = CfgResource(
            defaults,
            initial=initial,
            resolution=self._resolution,
            make_range=self._make_range,
            mutation_allowed=lambda: self._mutation_allowed(tab_id),
            notification_active=lambda: CfgResource.notifications_active(
                tuple(self._resources.values())
            ),
        )
        self._resources[tab_id] = resource
        return resource

    def lookup(self, tab_id: str) -> CfgEditing:
        """Resolve app identity without forwarding editing commands."""
        return self._require(tab_id)

    def acceptance(self, tab_id: str) -> CfgAcceptance:
        return self._require(tab_id)

    def retire(self, tab_id: str) -> None:
        """Revoke retained handles before removing the app association.

        A notification-reentrant call is rejected by the resource and leaves the
        association intact. The app checks its busy/close policy before calling.
        """
        resource = self._require(tab_id)
        resource.revoke()
        del self._resources[tab_id]

    def refresh_all(self) -> tuple[CfgObservation, ...]:
        return CfgResource.refresh_group(
            tuple(self._resources.values()), resolution=self._resolution
        )

    def _require_not_notifying(self) -> None:
        if CfgResource.notifications_active(tuple(self._resources.values())):
            raise CfgPreconditionError(
                CfgPreconditionReason.REENTRANT_MUTATION,
                "Tab lifetime mutation is forbidden during cfg notification",
            )

    def _require(self, tab_id: str) -> CfgResource:
        try:
            return self._resources[tab_id]
        except KeyError as exc:
            raise CfgPreconditionError(
                CfgPreconditionReason.RESOURCE_GONE,
                f"Tab {tab_id!r} has no cfg resource",
            ) from exc
