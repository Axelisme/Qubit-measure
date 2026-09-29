from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
from matplotlib import rc_context
from matplotlib.figure import Figure
from qtpy.QtCore import QEventLoop, QTimer
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKind, SaveStatus
from zcu_tools.gui.app.measure.events.completion import (
    SaveArtifactsFinishedPayload,
    SaveDataFinishedPayload,
)
from zcu_tools.gui.app.measure.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.measure.figure_export import SAVE_DPI, SAVE_FIGSIZE
from zcu_tools.gui.app.measure.services.guard import SavePermit
from zcu_tools.gui.app.measure.services.ports import SaveDestination
from zcu_tools.gui.app.measure.services.save import SaveService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import EventMeta, EventOrigin
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
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
    svc = SaveService(state, runner, bus, owner_scheduler=MagicMock())
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


def _await_artifact_completion(bus, handles, token):
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    observed = []

    def completed(_payload) -> None:
        observed.append(handles.known_outcome(token))
        loop.quit()

    subscription = bus.subscribe(SaveArtifactsFinishedPayload, completed)
    try:
        timer.start(5000)
        loop.exec()
    finally:
        timer.stop()
        subscription.unsubscribe()
    assert len(observed) == 1 and observed[0] is not None
    return observed[0]


@pytest.fixture
def batch_save_service(qapp):
    state = State(MagicMock())
    adapter = MagicMock()
    state.add_tab(
        "tab", Session(adapter_name="fake", adapter=adapter, cfg_schema=MagicMock())
    )
    state.update_tab_result("tab", object())
    primary, post = _make_figure(), _make_figure()
    state.update_tab_analyze("tab", object(), primary)
    state.update_tab_post_analyze("tab", object(), post)
    handles, bus, gate = OperationHandles(), EventBus(), MagicMock()
    background = BackgroundRunner()
    service = SaveService(
        state,
        OperationRunner(gate, handles, MagicMock(), background, bus),
        bus,
        owner_scheduler=QtOwnerScheduler(),
    )
    try:
        yield service, state, adapter, primary, post, handles, bus, gate
    finally:
        background.quiesce()
        background.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    "requested",
    [
        (ArtifactKind.POST_ANALYSIS, ArtifactKind.ANALYSIS),
        (ArtifactKind.DATA, ArtifactKind.POST_ANALYSIS, ArtifactKind.ANALYSIS),
    ],
)
@pytest.mark.parametrize("fails", [False, True])
def test_save_artifacts_runs_in_order_and_preserves_partial_success(
    batch_save_service, tmp_path: Path, requested: tuple[ArtifactKind, ...], fails: bool
) -> None:
    service, state, adapter, primary, post, handles, bus, gate = batch_save_service
    owner_id = threading.get_ident()
    calls: list[ArtifactKind] = []

    def image_export(kind: ArtifactKind, path: str) -> None:
        assert threading.get_ident() == owner_id
        calls.append(kind)
        if fails and kind is ArtifactKind.POST_ANALYSIS:
            raise OSError("post export failed")
        Path(path).write_bytes(b"image")

    primary.savefig.side_effect = lambda path, **kw: image_export(
        ArtifactKind.ANALYSIS, path
    )
    post.savefig.side_effect = lambda path, **kw: image_export(
        ArtifactKind.POST_ANALYSIS, path
    )

    def data_export(req) -> None:
        assert threading.get_ident() != owner_id
        calls.append(ArtifactKind.DATA)
        Path(req.data_path).write_bytes(b"data")

    adapter.save.side_effect = data_export
    destinations = tuple(
        SaveDestination(kind, str(tmp_path / f"{kind.value}.dat")) for kind in requested
    )
    observed = []
    submission = service.start_save_artifacts(SavePermit("tab"), destinations)

    def observe(payload) -> None:
        if payload.fact in (
            TabInteractionFact.SAVE_SUCCEEDED,
            TabInteractionFact.SAVE_FAILED,
        ):
            observed.append(handles.known_outcome(submission.operation_id))

    bus.subscribe(TabInteractionChangedPayload, observe)
    assert state.is_tab_busy("tab")
    with pytest.raises(FailedPreconditionError, match="busy"):
        service.start_save_artifacts(SavePermit("tab"), destinations)
    outcome = _await_artifact_completion(bus, handles, submission.operation_id)
    assert outcome.status == ("failed" if fails else "finished")
    assert observed == [outcome]
    assert not state.is_tab_busy("tab")
    assert service.active_save_operations() == ()
    assert not gate.register.called
    expected = [ArtifactKind.ANALYSIS, ArtifactKind.POST_ANALYSIS]
    if ArtifactKind.DATA in requested and not fails:
        expected.append(ArtifactKind.DATA)
    assert calls == expected
    artifacts = {a.kind: a for a in state.get_artifact_snapshots("tab")}
    assert artifacts[ArtifactKind.ANALYSIS].last_saved_path == str(
        tmp_path / "analysis.dat"
    )
    assert artifacts[ArtifactKind.POST_ANALYSIS].last_saved_path == (
        None if fails else str(tmp_path / "post_analysis.dat")
    )
    assert (artifacts[ArtifactKind.DATA].last_saved_path is not None) == (
        ArtifactKind.DATA in requested and not fails
    )


@pytest.mark.parametrize(
    "data_collision,extensionless", [(False, False), (True, False), (False, True)]
)
def test_batch_rejects_colliding_actual_paths_before_writing(
    batch_save_service, tmp_path: Path, data_collision: bool, extensionless: bool
) -> None:
    service, state, adapter, primary, post, handles, _bus, _gate = batch_save_service
    if data_collision:
        target = tmp_path / "shared_1.hdf5"
        other = SaveDestination(ArtifactKind.DATA, str(tmp_path / "shared.hdf5"))
    else:
        target = tmp_path / "shared.png"
        target.write_bytes(b"existing image")
        other = SaveDestination(
            ArtifactKind.POST_ANALYSIS, str(tmp_path / "sub" / ".." / "shared.png")
        )
    before = state.get_artifact_snapshots("tab")
    with pytest.raises(FailedPreconditionError, match="distinct"):
        service.start_save_artifacts(
            SavePermit("tab"),
            (
                SaveDestination(
                    ArtifactKind.ANALYSIS,
                    str(target.with_suffix("") if extensionless else target),
                ),
                other,
            ),
        )
    assert state.get_artifact_snapshots("tab") == before
    assert not state.is_tab_busy("tab")
    assert handles.live_count() == 0
    primary.savefig.assert_not_called()
    post.savefig.assert_not_called()
    adapter.save.assert_not_called()
    if data_collision:
        assert not target.exists()
    else:
        assert target.read_bytes() == b"existing image"


@pytest.mark.parametrize("image_format", ["png", "svg"])
def test_batch_extensionless_image_reports_existing_output(
    batch_save_service, tmp_path: Path, image_format: str
) -> None:
    service, state, _adapter, _primary, _post, handles, bus, _gate = batch_save_service
    figure = Figure()
    figure.subplots().plot([0, 1], [1, 0])
    state.update_tab_analyze("tab", object(), figure)
    draft_path = str(tmp_path / "figure")
    state.update_tab_analysis_image_path_override("tab", draft_path)
    with rc_context({"savefig.format": image_format}):
        submission = service.start_save_artifacts(
            SavePermit("tab"), (SaveDestination(ArtifactKind.ANALYSIS, draft_path),)
        )
        outcome = _await_artifact_completion(bus, handles, submission.operation_id)
    assert outcome.status == "finished"
    expected = tmp_path / f"figure.{image_format}"
    assert submission.destinations[0].path == str(expected)
    artifact = next(
        a
        for a in state.get_artifact_snapshots("tab")
        if a.kind is ArtifactKind.ANALYSIS
    )
    assert artifact.last_saved_path == str(expected)
    assert artifact.status is SaveStatus.SAVED
    assert expected.stat().st_size > 0
    assert not Path(draft_path).exists()


@pytest.mark.parametrize("kind", [ArtifactKind.ANALYSIS, ArtifactKind.POST_ANALYSIS])
def test_sync_image_service_records_actual_extensionless_path(
    batch_save_service, tmp_path: Path, kind: ArtifactKind
) -> None:
    service, state, _adapter, _primary, _post, _handles, _bus, _gate = (
        batch_save_service
    )
    figure = Figure()
    figure.subplots().plot([0, 1], [1, 0])
    if kind is ArtifactKind.ANALYSIS:
        state.update_tab_analyze("tab", object(), figure)
        save = service.save_image_sync
    else:
        state.update_tab_post_analyze("tab", object(), figure)
        save = service.save_post_image_sync
    with rc_context({"savefig.format": "png"}):
        save(SavePermit("tab"), str(tmp_path / "figure"))
    artifact = next(a for a in state.get_artifact_snapshots("tab") if a.kind is kind)
    expected = tmp_path / "figure.png"
    assert artifact.last_saved_path == str(expected)
    assert artifact.status is SaveStatus.SAVED
    assert expected.stat().st_size > 0


def test_batch_save_keeps_submission_signature_when_later_drafts_change(
    batch_save_service, tmp_path: Path
) -> None:
    service, state, adapter, primary, post, handles, bus, _gate = batch_save_service
    data_path = str(tmp_path / "data.hdf5")
    post_path = str(tmp_path / "post.png")
    state.update_tab_data_path_override("tab", data_path)
    state.update_tab_post_analysis_image_path_override("tab", post_path)
    state.update_tab_comment("tab", "submitted")

    def export_primary(path, **_kwargs) -> None:
        Path(path).write_bytes(b"primary")
        state.update_tab_data_path_override("tab", str(tmp_path / "new-data.hdf5"))
        state.update_tab_post_analysis_image_path_override(
            "tab", str(tmp_path / "new-post.png")
        )
        state.update_tab_comment("tab", "edited during export")
        state.get_artifact_snapshots("tab")

    primary.savefig.side_effect = export_primary
    post.savefig.side_effect = lambda path, **kw: Path(path).write_bytes(b"post")
    adapter.save.side_effect = lambda req: Path(req.data_path).write_text(req.comment)
    submission = service.start_save_artifacts(
        SavePermit("tab"),
        (
            SaveDestination(ArtifactKind.ANALYSIS, str(tmp_path / "primary.png")),
            SaveDestination(ArtifactKind.POST_ANALYSIS, post_path),
            SaveDestination(ArtifactKind.DATA, data_path),
        ),
        comment="submitted",
    )
    outcome = _await_artifact_completion(bus, handles, submission.operation_id)
    assert outcome.status == "finished"
    artifacts = {a.kind: a for a in state.get_artifact_snapshots("tab")}
    assert artifacts[ArtifactKind.DATA].status is SaveStatus.UNSAVED_CHANGES
    assert artifacts[ArtifactKind.POST_ANALYSIS].status is SaveStatus.SAVED
    actual_data = artifacts[ArtifactKind.DATA].last_saved_path
    assert actual_data is not None
    assert Path(actual_data).read_text() == "submitted"
    assert artifacts[ArtifactKind.POST_ANALYSIS].last_saved_path == post_path
    assert Path(post_path).read_bytes() == b"post"
    assert not (tmp_path / "new-post.png").exists()
    assert not (tmp_path / "new-data.hdf5").exists()


def test_batch_exports_captured_post_figure_when_replaced_before_export(
    batch_save_service, tmp_path: Path
) -> None:
    service, state, _adapter, primary, post, handles, bus, _gate = batch_save_service
    replacement = _make_figure()
    primary_path = str(tmp_path / "primary.png")
    post_path = str(tmp_path / "post.png")

    def export_primary(path, **_kwargs) -> None:
        Path(path).write_bytes(b"primary")
        state.update_tab_post_analyze("tab", object(), replacement)
        state.get_artifact_snapshots("tab")

    primary.savefig.side_effect = export_primary
    post.savefig.side_effect = lambda path, **kw: Path(path).write_bytes(b"old post")
    replacement.savefig.side_effect = lambda path, **kw: Path(path).write_bytes(
        b"new post"
    )
    submission = service.start_save_artifacts(
        SavePermit("tab"),
        (
            SaveDestination(ArtifactKind.ANALYSIS, primary_path),
            SaveDestination(ArtifactKind.POST_ANALYSIS, post_path),
        ),
    )
    outcome = _await_artifact_completion(bus, handles, submission.operation_id)
    assert outcome.status == "finished"
    assert Path(post_path).read_bytes() == b"old post"
    replacement.savefig.assert_not_called()
    snapshots = {a.kind: a for a in state.get_artifact_snapshots("tab")}
    assert snapshots[ArtifactKind.ANALYSIS].status is SaveStatus.SAVED
    assert snapshots[ArtifactKind.POST_ANALYSIS].status is SaveStatus.NOT_SAVED
    assert snapshots[ArtifactKind.POST_ANALYSIS].last_saved_path is None


def test_batch_later_parent_failure_preserves_earlier_saved_image(
    batch_save_service, tmp_path: Path
) -> None:
    service, state, adapter, primary, _post, handles, bus, _gate = batch_save_service
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("keep")
    image_path = str(tmp_path / "images" / "primary.png")
    primary.savefig.side_effect = lambda path, **kw: Path(path).write_bytes(b"primary")
    submission = service.start_save_artifacts(
        SavePermit("tab"),
        (
            SaveDestination(ArtifactKind.ANALYSIS, image_path),
            SaveDestination(ArtifactKind.DATA, str(parent_file / "data.hdf5")),
        ),
    )
    outcome = _await_artifact_completion(bus, handles, submission.operation_id)
    assert outcome.status == "failed"
    artifacts = {a.kind: a for a in state.get_artifact_snapshots("tab")}
    assert artifacts[ArtifactKind.ANALYSIS].status is SaveStatus.SAVED
    assert artifacts[ArtifactKind.ANALYSIS].last_saved_path == image_path
    assert Path(image_path).read_bytes() == b"primary"
    assert artifacts[ArtifactKind.DATA].last_saved_path is None
    assert not state.is_tab_busy("tab")
    adapter.save.assert_not_called()
    assert parent_file.read_text() == "keep"


def test_artifact_save_submit_failure_settles_before_completion(tmp_path: Path) -> None:
    handles, bus = OperationHandles(), EventBus()
    service, state, background = _make_service(handles=handles, bus=bus)
    background.submit.side_effect = RuntimeError("cannot submit")
    observed = []
    bus.subscribe(
        SaveArtifactsFinishedPayload,
        lambda payload: observed.append(
            (payload.error, handles.live_count(), state.is_tab_busy("tab"))
        ),
    )
    with pytest.raises(RuntimeError, match="cannot submit"):
        service.start_save_artifacts(
            SavePermit("tab"),
            (SaveDestination(ArtifactKind.DATA, str(tmp_path / "data")),),
        )
    assert observed == [("cannot submit", 0, False)]
    assert service.active_save_operations() == ()
    assert state.get_artifact_snapshots("tab")[0].last_saved_path is None


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
    assert after_failure[ArtifactKind.ANALYSIS].status is SaveStatus.SAVED
    assert after_failure[ArtifactKind.ANALYSIS].last_saved_path == image_path
    assert after_failure[ArtifactKind.DATA].status is SaveStatus.NOT_SAVED


@pytest.mark.parametrize("kind", [ArtifactKind.ANALYSIS, ArtifactKind.POST_ANALYSIS])
@pytest.mark.parametrize("replacement", ["result", "figure"])
def test_image_save_history_survives_edits_but_not_replacement(
    batch_save_service, tmp_path: Path, kind: ArtifactKind, replacement: str
) -> None:
    service, state, *_ = batch_save_service
    if kind is ArtifactKind.ANALYSIS:
        publish = state.update_tab_analyze
        set_path = state.update_tab_analysis_image_path_override
        save = service.save_image_sync
    else:
        publish = state.update_tab_post_analyze
        set_path = state.update_tab_post_analysis_image_path_override
        save = service.save_post_image_sync
    result = object()
    figure = Figure()
    ax = figure.subplots()
    ax.plot([0, 1], [1, 0])
    publish("tab", result, figure)
    path = str(tmp_path / "saved.png")
    save(SavePermit("tab"), path)
    assert Path(path).stat().st_size > 0

    ax.set_title("edited after saving")
    ax.set_xlim(0, 2)
    set_path("tab", str(tmp_path / "another.png"))
    saved = next(a for a in state.get_artifact_snapshots("tab") if a.kind is kind)
    assert saved.status is SaveStatus.SAVED
    assert saved.last_saved_path == path

    publish(
        "tab",
        object() if replacement == "result" else result,
        Figure() if replacement == "figure" else figure,
    )
    fresh = next(a for a in state.get_artifact_snapshots("tab") if a.kind is kind)
    assert fresh.status is SaveStatus.NOT_SAVED
    assert fresh.last_saved_path is None
    assert fresh.is_saveable


@pytest.mark.parametrize("kind", [ArtifactKind.ANALYSIS, ArtifactKind.POST_ANALYSIS])
def test_first_failed_image_save_remains_unsaved(
    batch_save_service, tmp_path: Path, kind: ArtifactKind
) -> None:
    service, state, _adapter, primary, post, *_ = batch_save_service
    figure = primary if kind is ArtifactKind.ANALYSIS else post
    save = (
        service.save_image_sync
        if kind is ArtifactKind.ANALYSIS
        else service.save_post_image_sync
    )
    figure.savefig.side_effect = OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        save(SavePermit("tab"), str(tmp_path / "failed.png"))
    snapshot = next(a for a in state.get_artifact_snapshots("tab") if a.kind is kind)
    assert snapshot.status is SaveStatus.NOT_SAVED
    assert snapshot.last_saved_path is None


@pytest.mark.parametrize("kind", [ArtifactKind.ANALYSIS, ArtifactKind.POST_ANALYSIS])
def test_image_save_completion_does_not_mark_replacement_saved(
    batch_save_service, tmp_path: Path, kind: ArtifactKind
) -> None:
    service, state, _adapter, primary, post, *_ = batch_save_service
    if kind is ArtifactKind.ANALYSIS:
        figure, publish, save = (
            primary,
            state.update_tab_analyze,
            service.save_image_sync,
        )
    else:
        figure, publish, save = (
            post,
            state.update_tab_post_analyze,
            service.save_post_image_sync,
        )

    def export(path, **_kwargs) -> None:
        Path(path).write_bytes(b"old image")
        publish("tab", object(), Figure())
        state.get_artifact_snapshots("tab")

    figure.savefig.side_effect = export
    path = str(tmp_path / "old.png")
    save(SavePermit("tab"), path)
    assert Path(path).read_bytes() == b"old image"
    snapshot = next(a for a in state.get_artifact_snapshots("tab") if a.kind is kind)
    assert snapshot.status is SaveStatus.NOT_SAVED
    assert snapshot.last_saved_path is None
