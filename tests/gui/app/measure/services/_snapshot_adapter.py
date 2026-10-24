"""Snapshot contract adapter with the unchanged production BaseAdapter.run."""

from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.experiment._saving_support import SavingCfg, SavingCore
from tests.experiment._snapshot_support import SnapshotCore


def SnapshotAdapter_make_form(ctx: FormContext) -> Form[SavingCfg]:
    del ctx
    return Form(SavingCfg, {})


SnapshotAdapter = Adapter(
    spec=SavingCore.AXES_SPEC,
    make_form=SnapshotAdapter_make_form,
    run=SnapshotCore().run,
    guide=None,
    analysis=None,
    post_analysis=None,
)
