from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.guard import SavePermit
from zcu_tools.gui.services.save import SaveService
from zcu_tools.gui.state import State, TabState


def _make_service() -> tuple[SaveService, State, MagicMock]:
    state = State(MagicMock())
    adapter = MagicMock()
    state.add_tab(
        "tab",
        TabState(adapter_name="fake", adapter=adapter, cfg_schema=MagicMock()),
    )
    state.update_tab_result("tab", object())
    runner = MagicMock()
    svc = SaveService(state, runner, EventBus())
    return svc, state, runner


def test_start_save_data_creates_parent_at_command_boundary(
    qapp,
    tmp_path: Path,  # noqa: ARG001
) -> None:
    svc, _, runner = _make_service()
    data_path = tmp_path / "data" / "measurement"

    svc.start_save_data(SavePermit(tab_id="tab"), str(data_path))

    assert data_path.parent.is_dir()
    runner.start_save.assert_called_once()


def test_save_image_creates_parent_at_command_boundary(
    qapp,
    tmp_path: Path,  # noqa: ARG001
) -> None:
    svc, state, _ = _make_service()
    figure = MagicMock()
    state.get_tab("tab").figure = figure
    image_path = tmp_path / "images" / "plot.png"

    svc.save_image_sync(SavePermit(tab_id="tab"), str(image_path))

    assert image_path.parent.is_dir()
    figure.savefig.assert_called_once_with(str(image_path))
