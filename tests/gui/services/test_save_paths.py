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


def test_save_data_path_joins_database_path_directly_without_appending_date(
    tmp_path: Path,
) -> None:
    """make_default_save_paths trusts ctx.database_path as the dated data folder
    (derive_project_paths owns the date) — it must NOT re-append YYYY/MM/Data_MMDD.
    Here the injected database_path has no date, so none appears in data_path."""
    ctx = _make_context(tmp_path)
    paths = FakeAdapter().make_default_save_paths(ctx)

    data_dir = str(Path(paths.data_path).parent)
    assert data_dir == str(tmp_path / "database")  # exactly ctx.database_path
