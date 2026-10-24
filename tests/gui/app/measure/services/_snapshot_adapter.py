"""Snapshot contract adapter with a record from the connected-device core."""

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.experiment._snapshot_support import SnapshotCore


def SnapshotAdapter_make_form(ctx: FormContext) -> Form[SavingCfg]:
    del ctx
    return Form(SavingCfg, {})


def SnapshotCore_run(
    cfg: SavingCfg, *, context: RunContext
) -> RunRecord[SavingCfg, SavingResult]:
    return RunRecord(cfg=cfg, result=SnapshotCore().run(cfg, context=context))


SnapshotAdapter = Adapter(
    spec=SavingCore.AXES_SPEC,
    make_form=SnapshotAdapter_make_form,
    run=SnapshotCore_run,
    guide=None,
    analysis=None,
    post_analysis=None,
)
