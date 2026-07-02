"""ProgressControlFacet delegation contract."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from zcu_tools.gui.session.progress_control import ProgressControlFacet


def test_progress_control_facet_delegates_owner_progress_calls() -> None:
    progress = MagicMock()
    facet = ProgressControlFacet(cast(Any, progress))
    listener = MagicMock()

    progress.attach_by_owner.return_value = "dispose"
    progress.bars_for_owner.return_value = ((1, "bar"),)

    assert facet.attach_progress("tab-1", listener) == "dispose"
    assert facet.progress_bars("tab-1") == ((1, "bar"),)

    progress.attach_by_owner.assert_called_once_with("tab-1", listener)
    progress.bars_for_owner.assert_called_once_with("tab-1")
