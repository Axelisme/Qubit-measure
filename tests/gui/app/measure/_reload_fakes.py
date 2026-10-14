"""Shared catalog-reload fakes for service and UI tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Literal

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import AnalysisDocument, RunMetadata
from zcu_tools.experiment.native_persistence import PreparedRunData
from zcu_tools.gui.app.measure.adapter import LoadDataRequest, SessionEnv
from zcu_tools.gui.app.measure.catalog import CatalogReloadError, PreparedCatalogReload
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.gui._adapter_fakes import DummyAdapter


class ReloadCfg(ConfigBase):
    """A knob is retained across catalog reload without scientific dependencies."""

    knob: int = 7


class OldAdapter(DummyAdapter):
    """Context-free cfg adapter for reload tests, without native persistence."""

    experiment_tag: ClassVar[str | None] = None

    @staticmethod
    def make_form(ctx: FormContext) -> Form[ConfigBase]:
        del ctx
        return Form(ReloadCfg, {})

    def load(self, req: LoadDataRequest) -> tuple[object, RunMetadata]:
        """Reject native Load requests; this reload-only fake has no loader."""
        del req
        raise NotImplementedError("Reload fake does not load native data")

    def capture_analysis_document(
        self,
        result: object,
        parameters: object | None,
        *,
        pane: Literal["analysis", "post_analysis"],
    ) -> AnalysisDocument:
        """Reject result/parameters for either pane; no analysis is persisted."""
        del result, parameters, pane
        raise TypeError("Reload fake does not capture analysis documents")

    def prepare_save_data(self, result: object, *, comment: str) -> PreparedRunData:
        """Reject any result/comment; this fake has no persistence declaration."""
        del result, comment
        raise NotImplementedError("Reload fake does not prepare native data")

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
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
