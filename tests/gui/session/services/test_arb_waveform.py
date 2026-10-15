from __future__ import annotations

import pytest
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.services.arb_waveform import resolve_arb_waveform_root
from zcu_tools.gui.session.types import SessionEnv


def test_resolve_arb_waveform_root_without_project_is_failed_precondition() -> None:
    ctx = SessionEnv(parameters=None, soc=None, soccfg=None)

    with pytest.raises(
        FailedPreconditionError,
        match="No project database_path is configured",
    ) as exc_info:
        resolve_arb_waveform_root(ctx)

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == "no_project"
