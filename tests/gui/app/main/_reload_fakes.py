"""Shared catalog-reload fakes for service and UI tests."""

from __future__ import annotations

from collections.abc import Callable

from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.app.main.catalog import CatalogReloadError, PreparedCatalogReload
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)

from tests.gui._adapter_fakes import DummyAdapter


class OldAdapter(DummyAdapter):
    def make_default_cfg(self, ctx: ExpContext) -> CfgSchema:
        del ctx
        return CfgSchema(
            spec=CfgSectionSpec(fields={"knob": ScalarSpec(label="Knob", type=int)}),
            value=CfgSectionValue(fields={"knob": DirectValue(7)}),
        )


class NewAdapter(OldAdapter):
    pass


class Loader:
    def __init__(self) -> None:
        self.candidate = Registry()
        self.candidate.register("demo", NewAdapter)
        self.failure: CatalogReloadError | None = None
        self.prepare_failure: Exception | None = None
        self.during_load: Callable[[], None] = lambda: None
        self.loads = 0

    def prepare(self) -> PreparedCatalogReload:
        if self.prepare_failure is not None:
            raise self.prepare_failure
        return PreparedCatalogReload()

    def load(self, plan: PreparedCatalogReload) -> Registry:
        del plan
        self.loads += 1
        self.during_load()
        if self.failure is not None:
            raise self.failure
        return self.candidate
