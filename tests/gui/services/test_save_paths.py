from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.gui.adapter import ContextReadiness, ExpContext
from zcu_tools.gui.registry import Registry
from zcu_tools.gui.services.tab import TabService
from zcu_tools.gui.state import State


def _make_context(tmp_path: Path) -> ExpContext:
    return ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        result_dir=str(tmp_path / "result"),
        database_path=str(tmp_path / "database"),
        active_label="ctx001",
        readiness=ContextReadiness.ACTIVE,
    )


def test_tab_save_path_query_is_pure_and_does_not_create_directories(
    tmp_path: Path,
) -> None:
    state = State(_make_context(tmp_path))
    registry = Registry()
    registry.register("fake", FakeAdapter)
    svc = TabService(state, registry)
    tab_id = svc.new_tab("fake")

    paths = svc.get_tab_save_paths(tab_id)

    assert paths is not None
    assert state.get_effective_save_paths(tab_id) is None
    assert not (tmp_path / "database").exists()
    assert not (tmp_path / "result").exists()
