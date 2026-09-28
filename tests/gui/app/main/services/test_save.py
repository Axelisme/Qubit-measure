from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.artifact_tracker import ArtifactKind, SaveStatus
from zcu_tools.gui.app.main.events.completion import SaveDataFinishedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.main.figure_export import SAVE_DPI, SAVE_FIGSIZE
from zcu_tools.gui.app.main.services.guard import SavePermit
from zcu_tools.gui.app.main.services.save import SaveService
from zcu_tools.gui.app.main.state import Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import EventMeta, EventOrigin
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner


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


def _make_service(
    *,
    handles: OperationHandles | None = None,
    gate: MagicMock | None = None,
    bus: EventBus | None = None,
) -> tuple[SaveService, State, MagicMock]:
    state = State(MagicMock())
    adapter = MagicMock()
    state.add_tab(
        "tab",
        Session(adapter_name="fake", adapter=adapter, cfg_schema=MagicMock()),
    )
    state.update_tab_result("tab", object())
    bg = MagicMock()  # BackgroundRunner stand-in; submit() is inspected per-test
    bus = bus if bus is not None else EventBus()
    runner = OperationRunner(
        gate if gate is not None else MagicMock(),
        handles if handles is not None else OperationHandles(),
        MagicMock(),
        bg,
        bus,
    )
    svc = SaveService(state, runner, bus)
    return svc, state, bg


def _record_facts(svc: SaveService) -> list[TabInteractionFact]:
    facts: list[TabInteractionFact] = []
    svc._bus.subscribe(  # type: ignore[attr-defined]
        TabInteractionChangedPayload,
        lambda payload: facts.append(payload.fact),
    )
    return facts


def _record_outcomes(svc: SaveService) -> list[SaveDataFinishedPayload]:
    outcomes: list[SaveDataFinishedPayload] = []
    svc._bus.subscribe(  # type: ignore[attr-defined]
        SaveDataFinishedPayload, outcomes.append
    )
    return outcomes


def test_start_save_data_creates_parent_at_command_boundary(
    qapp,
    tmp_path: Path,  # noqa: ARG001
) -> None:
    svc, _, bg = _make_service()
    facts = _record_facts(svc)
    data_path = tmp_path / "data" / "measurement"

    svc.start_save_data(SavePermit(tab_id="tab"), str(data_path))

    assert data_path.parent.is_dir()
    bg.submit.assert_called_once()
    assert facts == [TabInteractionFact.SAVE_STARTED]


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

    assert returned.data_path.endswith("meas_1.hdf5")
    assert returned.operation_id > 0


def test_save_terminal_restores_agent_origin_with_operation_id(
    qapp, tmp_path: Path
) -> None:  # noqa: ARG001
    svc, _state, bg = _make_service()
    observed: list[EventMeta] = []
    svc._bus.subscribe_with_meta(  # type: ignore[attr-defined]
        TabInteractionChangedPayload,
        lambda payload, meta: (
            observed.append(meta)
            if payload.fact is TabInteractionFact.SAVE_SUCCEEDED
            else None
        ),
    )

    with svc._bus.origin(EventOrigin(kind="agent", client_id="client-a")):  # type: ignore[attr-defined]
        started = svc.start_save_data(SavePermit(tab_id="tab"), str(tmp_path / "save"))
    on_done = bg.submit.call_args.kwargs["on_done"]
    on_done(None)

    assert observed == [
        EventMeta(
            seq=observed[0].seq,
            origin=EventOrigin(
                kind="agent",
                client_id="client-a",
                operation_id=str(started.operation_id),
            ),
        )
    ]


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("origin_kind", ["user", "agent"])
def test_save_terminal_subscribers_observe_settled_handle(
    tmp_path: Path, failed: bool, origin_kind: Literal["user", "agent"]
) -> None:
    handles = OperationHandles()
    bus = EventBus()
    svc, state, bg = _make_service(handles=handles, bus=bus)
    observed = []

    def observe(payload, meta):
        if (
            isinstance(payload, TabInteractionChangedPayload)
            and payload.fact is TabInteractionFact.SAVE_STARTED
        ):
            return
        token = int(meta.origin.operation_id)
        observed.append(
            (
                type(payload),
                handles.known_outcome(token),
                state.is_tab_busy("tab"),
                svc.active_save_operations(),
                state.get_artifact_snapshots("tab")[0].last_saved_path,
            )
        )

    bus.subscribe_with_meta(TabInteractionChangedPayload, observe)
    bus.subscribe_with_meta(SaveDataFinishedPayload, observe)
    with bus.origin(EventOrigin(kind=origin_kind)):
        started = svc.start_save_data(SavePermit(tab_id="tab"), str(tmp_path / "save"))
    if failed:
        bg.submit.call_args.kwargs["on_error"](OSError("disk full"))
    else:
        bg.submit.call_args.kwargs["on_done"](None)

    assert [entry[0] for entry in observed] == [
        TabInteractionChangedPayload,
        SaveDataFinishedPayload,
    ]
    for _, outcome, busy, active, last_path in observed:
        assert outcome is not None
        assert outcome.status == ("failed" if failed else "finished")
        assert outcome.error == ("disk full" if failed else None)
        assert not busy
        assert active == ()
        assert last_path == (None if failed else started.data_path)


def test_save_submit_failure_leaves_no_busy_tab_or_live_handle(tmp_path: Path) -> None:
    handles = OperationHandles()
    gate = MagicMock()
    svc, state, bg = _make_service(handles=handles, gate=gate)
    bg.submit.side_effect = RuntimeError("executor unavailable")
    with pytest.raises(RuntimeError, match="executor unavailable"):
        svc.start_save_data(SavePermit(tab_id="tab"), str(tmp_path / "save"))
    assert handles.live_count() == 0
    assert svc.active_save_operations() == ()
    assert not state.is_tab_busy("tab")
    assert state.get_artifact_snapshots("tab")[0].last_saved_path is None
    gate.ensure_can_start.assert_not_called()
    gate.register.assert_not_called()


def test_inline_save_terminal_does_not_leave_live_operation(tmp_path: Path) -> None:
    handles = OperationHandles()
    gate = MagicMock()
    svc, state, bg = _make_service(handles=handles, gate=gate)

    def inline(work, *, run_in_pool, on_done, on_error):
        on_done(work())

    bg.submit.side_effect = inline
    started = svc.start_save_data(SavePermit(tab_id="tab"), str(tmp_path / "save"))
    outcome = handles.known_outcome(started.operation_id)
    assert outcome is not None and outcome.status == "finished"
    assert handles.live_count() == 0
    assert not handles.has_cancel_hook(started.operation_id)
    assert svc.active_save_operations() == ()
    assert not state.is_tab_busy("tab")
    assert state.get_artifact_snapshots("tab")[0].last_saved_path == started.data_path
    gate.ensure_can_start.assert_not_called()
    gate.register.assert_not_called()


def test_save_image_creates_parent_at_command_boundary(
    qapp,
    tmp_path: Path,  # noqa: ARG001
) -> None:
    svc, state, _ = _make_service()
    figure = _make_figure()
    state.get_tab("tab").analysis.figure = figure
    image_path = tmp_path / "images" / "plot.png"

    svc.save_image_sync(SavePermit(tab_id="tab"), str(image_path))

    assert image_path.parent.is_dir()
    _assert_saved_fixed_size(figure, str(image_path))


@pytest.mark.parametrize(
    "entrypoint",
    ("start_save_data", "save_image_sync", "save_post_image_sync"),
)
def test_save_entrypoints_reject_busy_tab_before_side_effects(
    qapp,
    tmp_path: Path,
    entrypoint: str,
) -> None:
    svc, state, bg = _make_service()
    figure = _make_figure()
    tab = state.get_tab("tab")
    tab.analysis.figure = figure
    tab.post_analysis.figure = figure
    state.set_tab_analyzing("tab", True)
    permit = SavePermit(tab_id="tab")
    data_path = str(tmp_path / "data" / "measurement")
    image_path = str(tmp_path / "images" / "plot.png")

    with pytest.raises(FailedPreconditionError, match="busy"):
        if entrypoint == "start_save_data":
            svc.start_save_data(permit, data_path)
        elif entrypoint == "save_image_sync":
            svc.save_image_sync(permit, image_path)
        else:
            svc.save_post_image_sync(permit, image_path)

    bg.submit.assert_not_called()
    figure.savefig.assert_not_called()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "images").exists()


# ---------------------------------------------------------------------------
# Save Data completion
# ---------------------------------------------------------------------------


def test_save_success_emits_completion(tmp_path: Path) -> None:
    svc, _, bg = _make_service()
    facts = _record_facts(svc)
    permit = SavePermit(tab_id="tab")

    finished = _record_outcomes(svc)

    svc.start_save_data(permit, str(tmp_path / "data"))
    bg.submit.call_args.kwargs["on_done"](None)

    assert len(finished) == 1
    assert finished[0].tab_id == "tab"
    assert facts == [
        TabInteractionFact.SAVE_STARTED,
        TabInteractionFact.SAVE_SUCCEEDED,
    ]


def test_save_failure_emits_save_failed(tmp_path: Path) -> None:
    svc, _, bg = _make_service()
    facts = _record_facts(svc)
    permit = SavePermit(tab_id="tab")

    failed = _record_outcomes(svc)

    svc.start_save_data(permit, str(tmp_path / "data"))
    error = OSError("write error")
    bg.submit.call_args.kwargs["on_error"](error)

    assert len(failed) == 1
    assert failed[0].tab_id == "tab"
    assert failed[0].error == str(error)
    assert facts == [
        TabInteractionFact.SAVE_STARTED,
        TabInteractionFact.SAVE_FAILED,
    ]


def test_data_save_terminal_reports_actual_path_without_rewriting_draft(
    qapp, tmp_path: Path
) -> None:
    svc, state, bg = _make_service()
    draft_path = str(tmp_path / "data" / "measurement")
    state.update_tab_data_path_override("tab", draft_path)
    state.update_tab_comment("tab", "first")

    actual_path = svc.start_save_data(
        SavePermit(tab_id="tab"), draft_path, comment="first"
    ).data_path
    assert state.get_artifact_snapshots("tab")[0].status is SaveStatus.NOT_SAVED
    bg.submit.call_args.kwargs["on_done"](None)

    saved = state.get_artifact_snapshots("tab")[0]
    assert saved.kind is ArtifactKind.DATA
    assert saved.status is SaveStatus.SAVED
    assert saved.default_path == draft_path
    assert saved.last_saved_path == actual_path
    assert actual_path != draft_path
    state.update_tab_comment("tab", "edited later")
    changed = state.get_artifact_snapshots("tab")[0]
    assert changed.status is SaveStatus.UNSAVED_CHANGES
    assert changed.last_saved_path == actual_path


def test_failed_data_save_does_not_erase_prior_success(qapp, tmp_path: Path) -> None:
    svc, state, bg = _make_service()
    first = str(tmp_path / "first")
    state.update_tab_data_path_override("tab", first)
    previous_path = svc.start_save_data(SavePermit(tab_id="tab"), first).data_path
    bg.submit.call_args.kwargs["on_done"](None)
    assert state.get_artifact_snapshots("tab")[0].status is SaveStatus.SAVED

    second = str(tmp_path / "second")
    state.update_tab_data_path_override("tab", second)
    svc.start_save_data(SavePermit(tab_id="tab"), second)
    bg.submit.call_args.kwargs["on_error"](OSError("disk full"))

    artifact = state.get_artifact_snapshots("tab")[0]
    assert artifact.status is SaveStatus.UNSAVED_CHANGES
    assert artifact.default_path == second
    assert artifact.last_saved_path == previous_path


def test_image_save_success_and_failure_share_state_without_undoing_data(
    qapp, tmp_path: Path
) -> None:
    svc, state, _ = _make_service()
    figure = _make_figure()
    state.update_tab_analyze("tab", object(), figure)
    image_path = str(tmp_path / "analysis.png")
    state.update_tab_analysis_image_path_override("tab", image_path)

    svc.save_image_sync(SavePermit(tab_id="tab"), image_path)

    snapshots = {item.kind: item for item in state.get_artifact_snapshots("tab")}
    assert snapshots[ArtifactKind.DATA].status is SaveStatus.NOT_SAVED
    assert snapshots[ArtifactKind.ANALYSIS].status is SaveStatus.SAVED
    assert snapshots[ArtifactKind.ANALYSIS].last_saved_path == image_path

    next_path = str(tmp_path / "next.png")
    state.update_tab_analysis_image_path_override("tab", next_path)
    figure.savefig.side_effect = OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        svc.save_image_sync(SavePermit(tab_id="tab"), next_path)

    after_failure = {item.kind: item for item in state.get_artifact_snapshots("tab")}
    assert after_failure[ArtifactKind.ANALYSIS].status is SaveStatus.UNSAVED_CHANGES
    assert after_failure[ArtifactKind.ANALYSIS].last_saved_path == image_path
    assert after_failure[ArtifactKind.DATA].status is SaveStatus.NOT_SAVED
