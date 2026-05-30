from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.guard import SavePermit
from zcu_tools.gui.services.save import SaveBothOutcome, SaveService
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


# ---------------------------------------------------------------------------
# start_save_both
# ---------------------------------------------------------------------------


def test_start_save_both_saves_image_and_starts_data_runner(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, runner = _make_service()
    figure = MagicMock()
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    svc.start_save_both(SavePermit(tab_id="tab"), str(data_path), str(image_path))

    figure.savefig.assert_called_once_with(str(image_path))
    runner.start_save.assert_called_once()


def test_start_save_both_captures_image_error_and_continues_data(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, runner = _make_service()
    figure = MagicMock()
    figure.savefig.side_effect = OSError("disk full")
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    # Should not raise — image error is captured, data save continues
    svc.start_save_both(SavePermit(tab_id="tab"), str(data_path), str(image_path))

    runner.start_save.assert_called_once()


def test_start_save_both_raises_if_no_figure(qapp) -> None:  # noqa: ARG001
    svc, _, _ = _make_service()
    with pytest.raises(RuntimeError, match="No figure"):
        svc.start_save_both(SavePermit(tab_id="tab"), "/data", "/img")


# ---------------------------------------------------------------------------
# _on_save_finished without pending_image
# ---------------------------------------------------------------------------


def test_on_save_finished_emits_save_finished(qapp) -> None:  # noqa: ARG001
    svc, _, runner = _make_service()
    permit = SavePermit(tab_id="tab")

    finished: list = []
    svc.save_finished.connect(lambda tid, path: finished.append((tid, path)))

    # Stage the active path as start_save_data would
    svc.start_save_data(permit, "/tmp/data")
    svc._on_save_finished("tab")

    assert len(finished) == 1
    assert finished[0][0] == "tab"


# ---------------------------------------------------------------------------
# _on_save_finished with pending_image (save_both flow)
# ---------------------------------------------------------------------------


def test_on_save_finished_with_pending_image_emits_save_both_finished(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, runner = _make_service()
    figure = MagicMock()
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    outcomes: list[SaveBothOutcome] = []
    svc.save_both_finished.connect(lambda tid, o: outcomes.append(o))

    svc.start_save_both(SavePermit(tab_id="tab"), str(data_path), str(image_path))
    svc._on_save_finished("tab")

    assert len(outcomes) == 1
    assert outcomes[0].data_error is None
    assert outcomes[0].image_error is None


def test_on_save_finished_with_pending_image_error_propagates(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, runner = _make_service()
    figure = MagicMock()
    figure.savefig.side_effect = OSError("disk full")
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    outcomes: list[SaveBothOutcome] = []
    svc.save_both_finished.connect(lambda tid, o: outcomes.append(o))

    svc.start_save_both(SavePermit(tab_id="tab"), str(data_path), str(image_path))
    svc._on_save_finished("tab")

    assert len(outcomes) == 1
    assert outcomes[0].image_error == "disk full"
    assert outcomes[0].data_error is None


# ---------------------------------------------------------------------------
# _on_save_failed
# ---------------------------------------------------------------------------


def test_on_save_failed_emits_save_failed(qapp) -> None:  # noqa: ARG001
    svc, _, runner = _make_service()
    permit = SavePermit(tab_id="tab")

    failed: list = []
    svc.save_failed.connect(lambda tid, path, err: failed.append((tid, err)))

    svc.start_save_data(permit, "/tmp/data")
    error = OSError("write error")
    svc._on_save_failed("tab", error)

    assert len(failed) == 1
    assert failed[0][0] == "tab"
    assert failed[0][1] is error


def test_on_save_failed_with_pending_image_emits_save_both_finished(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, runner = _make_service()
    figure = MagicMock()
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    outcomes: list[SaveBothOutcome] = []
    svc.save_both_finished.connect(lambda tid, o: outcomes.append(o))

    svc.start_save_both(SavePermit(tab_id="tab"), str(data_path), str(image_path))
    error = OSError("data write failed")
    svc._on_save_failed("tab", error)

    assert len(outcomes) == 1
    assert outcomes[0].data_error == str(error)
    assert outcomes[0].image_error is None
