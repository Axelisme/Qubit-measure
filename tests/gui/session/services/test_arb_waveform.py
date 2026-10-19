from __future__ import annotations

from pathlib import Path

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.services.arb_waveform import (
    ArbWaveformService,
    resolve_arb_waveform_root,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


def test_entry_switch_routes_waveforms_to_each_entry_database(tmp_path: Path) -> None:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    context = ContextService(
        state, BaseEventBus(), bind_assets=ArbWaveformDatabase.info_at
    )
    service = ArbWaveformService(state)
    database_root = tmp_path / "unknown_chip" / "unknown_qubit" / "Database"
    first = ResultEntry.create(
        "first", result_root=tmp_path / "result", database_root=database_root
    )
    second = ResultEntry.create(
        "second", result_root=tmp_path / "result", database_root=database_root
    )
    recipe: dict[str, object] = {
        "segments": [{"duration": 0.002, "formula": "0.5"}],
        "normalize": "none",
    }

    context.install_entry(first)
    assert service.root_path() == first.database_path / "arb_waveforms"
    assert service.set_formula("first_pulse", recipe, overwrite=False) == "created"
    assert service.list_data_keys() == ["first_pulse"]

    context.install_entry(second)
    assert service.root_path() == second.database_path / "arb_waveforms"
    assert service.list_data_keys() == []
    assert service.set_formula("second_pulse", recipe, overwrite=False) == "created"
    assert service.list_data_keys() == ["second_pulse"]

    context.install_entry(first)
    assert service.list_data_keys() == ["first_pulse"]
    assert service.load_data("first_pulse").idata.tolist() == [0.5, 0.5, 0.5]


def test_resolve_arb_waveform_root_without_project_is_failed_precondition() -> None:
    ctx = SessionEnv(parameters=None, soc=None, soccfg=None)

    with pytest.raises(
        FailedPreconditionError,
        match="No project database_path is configured",
    ) as exc_info:
        resolve_arb_waveform_root(ctx)

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == "no_project"
