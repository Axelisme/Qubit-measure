"""Role-declared adapter shared by native acquisition and diagnostic save tests."""

from zcu_tools.gui.app.measure.adapter import Adapter

from tests.gui.app.measure._native_save_support import (
    SavingAdapter,
)

RoleSavingAdapter = Adapter(
    spec=SavingAdapter.spec,
    make_form=SavingAdapter.make_form,
    run=SavingAdapter.run,
    guide=None,
    analysis=SavingAdapter.analysis,
    post_analysis=SavingAdapter.post_analysis,
)
