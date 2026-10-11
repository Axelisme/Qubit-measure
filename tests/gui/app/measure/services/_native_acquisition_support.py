"""Role-declared adapter shared by native acquisition and diagnostic save tests."""

from tests.gui.app.measure._native_save_support import (
    SavingAdapter,
)


class RoleSavingAdapter(SavingAdapter):
    """Use the existing offline saving backend with a declared driver alias."""
