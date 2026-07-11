from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.figure_export import SAVE_DPI, SAVE_FIGSIZE
from zcu_tools.gui.app.main.services.guard import SavePermit
from zcu_tools.gui.app.main.services.save import SaveResultOutcome, SaveService
from zcu_tools.gui.app.main.state import Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)


def _make_figure() -> MagicMock:
    """Mock figure whose get_size_inches returns a real tuple, so the fixed-size
    export helper (set/savefig/restore) can run against it."""
    figure = MagicMock()
    figure.get_size_inches.return_value = (6.0, 4.0)
    return figure


def _assert_saved_fixed_size(figure: MagicMock, image_path: str) -> None:
    """save_figure_to_path pins the fixed export size, savefig(path, dpi), restores."""
    figure.savefig.assert_called_once_with(image_path, dpi=SAVE_DPI)
    figure.set_size_inches.assert_any_call(*SAVE_FIGSIZE)
    figure.set_size_inches.assert_called_with(6.0, 4.0)  # restored last


def _make_service() -> tuple[SaveService, State, MagicMock]:
    state = State(MagicMock())
    adapter = MagicMock()
    state.add_tab(
        "tab",
        Session(adapter_name="fake", adapter=adapter, cfg_schema=MagicMock()),
    )
    state.update_tab_result("tab", object())
    bg = MagicMock()  # BackgroundRunner stand-in; submit() is inspected per-test
    svc = SaveService(state, bg, EventBus())
    return svc, state, bg


def test_start_save_data_creates_parent_at_command_boundary(
    qapp,
    tmp_path: Path,  # noqa: ARG001
) -> None:
    svc, _, bg = _make_service()
    data_path = tmp_path / "data" / "measurement"

    svc.start_save_data(SavePermit(tab_id="tab"), str(data_path))

    assert data_path.parent.is_dir()
    bg.submit.assert_called_once()


def test_start_save_data_resolves_path_to_actual_hdf5(
    qapp,
    tmp_path: Path,  # noqa: ARG001
) -> None:
    # The path handed to the saver (and reported to the agent) is normalised up
    # front to what actually lands on disk: .hdf5 extension + uniqueness suffix —
    # not the caller's raw stem (Phase 130 follow-up: display matches reality).
    svc, _, _ = _make_service()
    data_path = tmp_path / "data" / "meas"  # no extension

    returned = svc.start_save_data(SavePermit(tab_id="tab"), str(data_path))

    # The path the saver actually writes (.hdf5 + uniqueness suffix) is resolved
    # up front. The returned path and the reported _active_paths both reflect it
    # (start_save_data returns it synchronously, so the RPC/agent gets it back
    # immediately, not via a later diagnostic).
    assert returned.endswith("meas_1.hdf5")
    assert svc._active_paths["tab"] == returned


def test_save_image_creates_parent_at_command_boundary(
    qapp,
    tmp_path: Path,  # noqa: ARG001
) -> None:
    svc, state, _ = _make_service()
    figure = _make_figure()
    state.get_tab("tab").figure = figure
    image_path = tmp_path / "images" / "plot.png"

    svc.save_image_sync(SavePermit(tab_id="tab"), str(image_path))

    assert image_path.parent.is_dir()
    _assert_saved_fixed_size(figure, str(image_path))


# ---------------------------------------------------------------------------
# start_save_result
# ---------------------------------------------------------------------------


def test_start_save_result_saves_image_and_starts_data_save(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, bg = _make_service()
    figure = _make_figure()
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    svc.start_save_result(SavePermit(tab_id="tab"), str(data_path), str(image_path))

    _assert_saved_fixed_size(figure, str(image_path))
    bg.submit.assert_called_once()


def test_start_save_result_captures_image_error_and_continues_data(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, bg = _make_service()
    figure = _make_figure()
    figure.savefig.side_effect = OSError("disk full")
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    # Should not raise — image error is captured, data save continues
    svc.start_save_result(SavePermit(tab_id="tab"), str(data_path), str(image_path))

    bg.submit.assert_called_once()


def test_start_save_result_raises_if_no_figure(qapp) -> None:  # noqa: ARG001
    svc, _, _ = _make_service()
    with pytest.raises(FailedPreconditionError, match="No figure") as exc_info:
        svc.start_save_result(SavePermit(tab_id="tab"), "/data", "/img")

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""


# ---------------------------------------------------------------------------
# _on_save_finished without pending_image
# ---------------------------------------------------------------------------


def test_on_save_finished_emits_save_finished(qapp) -> None:  # noqa: ARG001
    svc, _, _ = _make_service()
    permit = SavePermit(tab_id="tab")

    finished: list = []
    svc.save_finished.connect(lambda tid, path: finished.append((tid, path)))

    # Stage the active path as start_save_data would
    svc.start_save_data(permit, "/tmp/data")
    svc._on_save_finished("tab")

    assert len(finished) == 1
    assert finished[0][0] == "tab"


# ---------------------------------------------------------------------------
# _on_save_finished with pending_image (save_result flow)
# ---------------------------------------------------------------------------


def test_on_save_finished_with_pending_image_emits_save_result_finished(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, _ = _make_service()
    figure = _make_figure()
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    outcomes: list[SaveResultOutcome] = []
    svc.save_result_finished.connect(lambda tid, o: outcomes.append(o))

    svc.start_save_result(SavePermit(tab_id="tab"), str(data_path), str(image_path))
    svc._on_save_finished("tab")

    assert len(outcomes) == 1
    assert outcomes[0].data_error is None
    assert outcomes[0].image_error is None


def test_on_save_finished_with_pending_image_error_propagates(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, _ = _make_service()
    figure = _make_figure()
    figure.savefig.side_effect = OSError("disk full")
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    outcomes: list[SaveResultOutcome] = []
    svc.save_result_finished.connect(lambda tid, o: outcomes.append(o))

    svc.start_save_result(SavePermit(tab_id="tab"), str(data_path), str(image_path))
    svc._on_save_finished("tab")

    assert len(outcomes) == 1
    assert outcomes[0].image_error == "disk full"
    assert outcomes[0].data_error is None


# ---------------------------------------------------------------------------
# _on_save_failed
# ---------------------------------------------------------------------------


def test_on_save_failed_emits_save_failed(qapp) -> None:  # noqa: ARG001
    svc, _, _ = _make_service()
    permit = SavePermit(tab_id="tab")

    failed: list = []
    svc.save_failed.connect(lambda tid, path, err: failed.append((tid, err)))

    svc.start_save_data(permit, "/tmp/data")
    error = OSError("write error")
    svc._on_save_failed("tab", error)

    assert len(failed) == 1
    assert failed[0][0] == "tab"
    assert failed[0][1] is error


def test_on_save_failed_with_pending_image_emits_save_result_finished(
    qapp,
    tmp_path: Path,
) -> None:
    svc, state, _ = _make_service()
    figure = _make_figure()
    state.get_tab("tab").figure = figure
    data_path = tmp_path / "data" / "meas"
    image_path = tmp_path / "img" / "plot.png"

    outcomes: list[SaveResultOutcome] = []
    svc.save_result_finished.connect(lambda tid, o: outcomes.append(o))

    svc.start_save_result(SavePermit(tab_id="tab"), str(data_path), str(image_path))
    error = OSError("data write failed")
    svc._on_save_failed("tab", error)

    assert len(outcomes) == 1
    assert outcomes[0].data_error == str(error)
    assert outcomes[0].image_error is None
