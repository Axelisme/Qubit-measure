from __future__ import annotations

from collections.abc import Generator
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.artifacts import ArtifactRef, SaveReport
from zcu_tools.gui.app.measure.adapter import (
    ContextReadiness,
    SessionEnv,
)
from zcu_tools.gui.app.measure.artifacts import (
    SaveStatus,
)
from zcu_tools.gui.app.measure.services.ports import TabSnapshot
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
    RunPaneState,
    SavePaneState,
    Session,
    State,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._cfg_fakes import (
    parameter_capture,
)
from tests.gui.app.measure._form_fakes import form_resources, form_sources
from tests.gui.app.measure._native_save_support import (
    GUI_ORIGIN,
    NativeSaveSession,
    native_save_session,
)


def test_tab_snapshot_is_single_pure_render_model() -> None:
    state = State(
        SessionEnv(
            parameters=parameter_capture(),
            soc=MagicMock(),
            soccfg=MagicMock(),
            readiness=ContextReadiness.ACTIVE,
        )
    )
    analyze_params = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=MagicMock(), params=analyze_params),
            save=SavePaneState(comment="draft note"),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.return_value = []
    # TabService's render model depends only on State + a writeback query port;
    # readiness / save paths come off State's aggregates, not sibling
    # app-services. The registry is unused by get_snapshot.
    service = TabService(
        state,
        MagicMock(),
        writeback,
        form_resources(state),
        form_sources=lambda aliases: form_sources(state, aliases),
        captured_parameters=lambda _aliases: state.session_env.parameters,
    )

    snapshot = service.get_snapshot("tab")

    assert snapshot.tab_id == "tab"
    assert snapshot.interaction is not None  # render path fills every live field
    assert snapshot.interaction.has_run_result is True
    assert snapshot.interaction.has_active_context is True  # ctx.readiness=ACTIVE
    assert snapshot.analysis is not None
    assert snapshot.analysis.params is analyze_params
    assert snapshot.save is not None
    assert asdict(snapshot.save)["comment"] == "draft note"
    assert state.get_tab("tab").analysis.params is analyze_params


def test_snapshot_projects_empty_writeback_draft_existence() -> None:
    state = _active_state()
    draft = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg=MagicMock(),
            analysis=AnalysisPaneState(writeback_draft=draft),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.return_value = []

    snapshot = TabService(
        state,
        MagicMock(),
        writeback,
        form_resources(state),
        form_sources=lambda aliases: form_sources(state, aliases),
        captured_parameters=lambda _aliases: state.session_env.parameters,
    ).get_snapshot("tab")

    assert snapshot.analysis is not None
    assert snapshot.analysis.has_writeback_draft is True
    assert snapshot.analysis.writeback_items == ()


def test_snapshot_propagates_writeback_preview_failure() -> None:
    state = _active_state()
    draft = object()
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=MagicMock(),
            cfg=MagicMock(),
            analysis=AnalysisPaneState(writeback_draft=draft),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.side_effect = RuntimeError("broken draft")

    with pytest.raises(RuntimeError, match="broken draft"):
        TabService(
            state,
            MagicMock(),
            writeback,
            form_resources(state),
            form_sources=lambda aliases: form_sources(state, aliases),
            captured_parameters=lambda _aliases: state.session_env.parameters,
        ).get_snapshot("tab")


def test_snapshot_projects_running_owner_without_session_run_flag() -> None:
    state = _active_state()
    for tab_id in ("running", "idle"):
        state.add_tab(
            tab_id,
            Session(
                adapter_name="fake",
                adapter=MagicMock(),
                cfg=MagicMock(),
            ),
        )
    state.set_tab_running("running", running=True)
    writeback = MagicMock()
    writeback.preview_draft.return_value = []
    service = TabService(
        state,
        MagicMock(),
        writeback,
        form_resources(state),
        form_sources=lambda aliases: form_sources(state, aliases),
        captured_parameters=lambda _aliases: state.session_env.parameters,
    )

    running = service.get_snapshot("running").interaction
    idle = service.get_snapshot("idle").interaction

    assert running is not None
    assert running.is_running is True
    assert running.global_run_active is False
    assert idle is not None
    assert idle.is_running is False
    assert idle.global_run_active is True


def _active_state() -> State:
    return State(
        SessionEnv(
            parameters=parameter_capture(),
            soc=MagicMock(),
            soccfg=MagicMock(),
            readiness=ContextReadiness.ACTIVE,
        )
    )


def test_snapshot_carries_post_analyze_fields() -> None:
    state = _active_state()
    post_params = object()
    post_fig = Figure()
    post_plots = Plots(NonPresentingHost())
    post_plots.adopt("fit", post_fig)
    post_plots.finish()
    state.add_tab(
        "tab",
        Session(
            adapter_name="ge",
            adapter=MagicMock(),
            cfg=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=MagicMock()),
            post_analysis=PostAnalysisPaneState(
                result=MagicMock(),
                params=post_params,
                plots=post_plots,
            ),
        ),
    )
    writeback = MagicMock()
    writeback.preview_draft.return_value = []
    service = TabService(
        state,
        MagicMock(),
        writeback,
        form_resources(state),
        form_sources=lambda aliases: form_sources(state, aliases),
        captured_parameters=lambda _aliases: state.session_env.parameters,
    )

    snapshot = service.get_snapshot("tab")

    assert snapshot.post_analysis is not None
    assert snapshot.post_analysis.params is post_params
    assert snapshot.post_analysis.figures is not None
    assert snapshot.post_analysis.figures is post_plots
    assert snapshot.post_analysis.figures["fit"] is post_fig
    assert snapshot.interaction is not None
    assert snapshot.interaction.has_post_analyze_result is True


def test_initialize_post_analyze_params_seeds_from_primary_result() -> None:
    state = _active_state()
    adapter = MagicMock()
    built = object()
    adapter.get_post_analyze_params.return_value = built
    state.add_tab(
        "tab",
        Session(
            adapter_name="ge",
            adapter=adapter,
            cfg=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=MagicMock()),  # primary result present
        ),
    )
    service = TabService(
        state,
        MagicMock(),
        MagicMock(),
        form_resources(state),
        form_sources=lambda aliases: form_sources(state, aliases),
        captured_parameters=lambda _aliases: state.session_env.parameters,
    )

    out = service.initialize_tab_post_analyze_params("tab")

    assert out is built
    assert state.get_tab("tab").post_analysis.params is built


def test_initialize_post_analyze_params_fast_fails_without_primary_result() -> None:
    import pytest

    state = _active_state()
    state.add_tab(
        "tab",
        Session(
            adapter_name="ge",
            adapter=MagicMock(),
            cfg=MagicMock(),
            run=RunPaneState(result=object()),
            analysis=AnalysisPaneState(result=None),  # no primary analyze result
        ),
    )
    service = TabService(
        state,
        MagicMock(),
        MagicMock(),
        form_resources(state),
        form_sources=lambda aliases: form_sources(state, aliases),
        captured_parameters=lambda _aliases: state.session_env.parameters,
    )

    with pytest.raises(RuntimeError, match="primary analyze result"):
        service.initialize_tab_post_analyze_params("tab")


def _native_snapshot(session: NativeSaveSession) -> TabSnapshot:
    return TabService(
        session.state,
        registry=MagicMock(),
        writeback=MagicMock(),
        cfg_resources=MagicMock(),
        form_sources=MagicMock(),
        captured_parameters=MagicMock(),
    ).get_snapshot("tab")


@pytest.fixture
def native_session(tmp_path: Path, qapp: QApplication) -> Generator[NativeSaveSession]:
    with native_save_session(tmp_path, qapp) as session:
        yield session


def test_tab_snapshot_loaded_artifacts_have_no_success_record(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    record = session.state.get_tab("tab").run.result
    assert record is not None
    run = session.state.get_tab("tab").run
    session.state.update_tab_loaded_result(
        "tab",
        record,
        "existing.hdf5",
        metadata=session.metadata,
        parameters=run.parameters,
        storage=run.storage,
    )
    snapshot = _native_snapshot(session)

    assert snapshot.run is not None and snapshot.run.source_path == "existing.hdf5"
    assert [
        (section.section, artifact.ref, artifact.status)
        for section in snapshot.artifacts
        for artifact in section.artifacts
    ] == [
        ("run", ArtifactRef("run", "measurement_data"), SaveStatus.NOT_SAVED),
    ]
    data = snapshot.artifacts[0].artifacts[0]
    assert [output.format for output in data.members[0].outputs] == [
        "data_h5",
        "labber",
    ]
    assert all(output.status == "reserved" for output in data.members[0].outputs)
    assert data.members[0].outputs[0].path == (session.native_data_path)


def test_tab_snapshot_tracks_actual_success_and_draft_or_result_drift(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    assert (
        _native_snapshot(session).artifacts[0].artifacts[0].status
        is SaveStatus.NOT_SAVED
    )
    first = session.control.save_data("tab", comment="initial", origin=GUI_ORIGIN)
    data = _native_snapshot(session).artifacts[0].artifacts[0]
    assert data.status is SaveStatus.SAVED
    assert [output.path for output in data.members[0].outputs] == [
        item.output.path for item in first.outputs
    ]

    session.state.update_tab_comment("tab", "edited")
    assert (
        _native_snapshot(session).artifacts[0].artifacts[0].status
        is SaveStatus.UNSAVED_CHANGES
    )
    session.state.update_tab_comment("tab", "initial")
    assert (
        _native_snapshot(session).artifacts[0].artifacts[0].status
        is SaveStatus.UNSAVED_CHANGES
    )
    session.control.save_data("tab", origin=GUI_ORIGIN)
    assert (
        _native_snapshot(session).artifacts[0].artifacts[0].status is SaveStatus.SAVED
    )

    record = session.state.get_tab("tab").run.result
    assert record is not None
    run = session.state.get_tab("tab").run
    session.state.update_tab_result(
        "tab", record, metadata=session.metadata, parameters=run.parameters
    )
    assert (
        _native_snapshot(session).artifacts[0].artifacts[0].status
        is SaveStatus.NOT_SAVED
    )
    session.state.update_tab_loaded_result(
        "tab",
        record,
        "imported.hdf5",
        metadata=session.metadata,
        parameters=run.parameters,
        storage=run.storage,
    )
    loaded = _native_snapshot(session)
    assert loaded.run is not None and loaded.run.source_path == "imported.hdf5"
    assert loaded.artifacts[0].artifacts[0].status is SaveStatus.NOT_SAVED
    assert all(
        output.status == "reserved"
        for output in loaded.artifacts[0].artifacts[0].members[0].outputs
    )


def test_artifact_failure_replaces_previous_output_evidence_without_marking_draft_saved(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    first = session.control.save_data("tab", origin=GUI_ORIGIN)
    assert (
        _native_snapshot(session).artifacts[0].artifacts[0].status is SaveStatus.SAVED
    )
    session.state.update_tab_comment("tab", "next")
    assert (
        _native_snapshot(session).artifacts[0].artifacts[0].status
        is SaveStatus.UNSAVED_CHANGES
    )
    failed = SaveReport(
        first.run_id,
        tuple(
            replace(item, status="failed", error="disk full") for item in first.outputs
        ),
    )
    session.state.get_tab("tab").artifacts.apply_report(failed)
    after_failure = _native_snapshot(session).artifacts[0].artifacts[0]
    assert after_failure.status is SaveStatus.NOT_SAVED
    assert [
        (output.format, output.path, output.status, output.error)
        for output in after_failure.members[0].outputs
    ] == [
        (item.output.format, item.output.path, item.status, item.error)
        for item in failed.outputs
    ]
