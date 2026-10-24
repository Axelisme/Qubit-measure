"""Shared catalog-reload fakes for service and UI tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
)
from zcu_tools.gui.app.measure.catalog import CatalogReloadError, PreparedCatalogReload
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.gui._adapter_fakes import DummyAdapter


class ReloadCfg(ConfigBase):
    """A knob is retained across catalog reload without scientific dependencies."""

    knob: int = 7


def OldAdapter_make_form(ctx: FormContext) -> Form[ConfigBase]:
    del ctx
    return Form(ReloadCfg, {})


OldAdapter = Adapter(
    spec=replace(DummyAdapter.spec, cfg_type=ReloadCfg),
    make_form=OldAdapter_make_form,
    run=DummyAdapter.run,
    guide=None,
    analysis=DummyAdapter.analysis,
    post_analysis=DummyAdapter.post_analysis,
)


NewAdapter = Adapter(
    spec=OldAdapter.spec,
    make_form=OldAdapter.make_form,
    run=OldAdapter.run,
    guide=None,
    analysis=OldAdapter.analysis,
    post_analysis=OldAdapter.post_analysis,
)


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
