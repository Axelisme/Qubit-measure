"""Snapshot contract adapter with the unchanged production BaseAdapter.run."""

from typing import ClassVar

from zcu_tools.experiment import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.experiment._snapshot_support import SnapshotRecordCore
from tests.gui.app.measure.adapter._cfg_fakes import StaticCfgDefinition


class SnapshotAdapter(BaseAdapter[SavingCfg, RunRecord[SavingCfg, SavingResult]]):
    """Use actual assembly and core acquisition with no mocked run method."""

    exp_cls = SnapshotRecordCore

    @staticmethod
    def make_form(ctx: FormContext) -> Form[SavingCfg]:
        del ctx
        return Form(SavingCfg, {})

    ExpCfg_cls = SavingCfg
    experiment_tag: ClassVar[str | None] = SavingCore().persistence_spec().tag
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=False, requires_soc=False
    )

    @classmethod
    def cfg_definition(cls) -> StaticCfgDefinition:
        return StaticCfgDefinition()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return "snapshot"
