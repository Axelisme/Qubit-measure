"""Captured native parameter environments and deferred historical device evidence."""

from dataclasses import replace

import pytest
from zcu_tools.datafile import load
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.load import LoadService
from zcu_tools.gui.app.measure.services.pane_commit import commit_run
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.state import RunPaneState
from zcu_tools.gui.app.measure.writeback import WritebackService
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.parameter_capture import capture_parameters
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._native_save_support import (
    AnalysisSavingAdapter,
    NativeSaveSession,
    PaneParams,
    PaneSummary,
    PostSummary,
)
from tests.gui.app.measure._record_support import (
    require_metadata,
    require_record,
    require_target,
)


@pytest.mark.parametrize("absent_library", [False, True])
def test_native_save_load_uses_captured_environment_after_source_removal(
    role_load_session: NativeSaveSession,
    absent_library: bool,
) -> None:
    import shutil

    from zcu_tools.cfg_model import PulseSpec

    session = role_load_session
    control = session.context
    target = require_target(control.read_context())
    assert target.point is not None, "Fixture must select a point before reloading it"
    control.set_module(
        target,
        "Q1.pulses.pi",
        PulseSpec(
            ch=0,
            nqz=1,
            freq=100.0,
            gain=0.25,
            waveform="const",
        ),
    )
    library_file = session.context.get_file_revisions()[1].path
    if absent_library:
        library_file.unlink()
    parameter_file = session.context.get_file_revisions()[0].path
    parameter_file.write_text(
        parameter_file.read_text().replace(
            "general:\n",
            "general:\n  entry_id: ignored-legacy-identity\n"
            "  description: captured environment\n"
            "  ext: {entry_id: user-owned-extension, note: preserved}\n",
        )
    )
    source_bytes = parameter_file.read_bytes()
    control.use_point(PointUseRequest(target.point))
    captured = control.read_context()
    tab = session.state.get_tab("tab")
    record = require_record(tab.run)
    metadata = require_metadata(tab.run).model_copy(
        update={
            "parameter_snapshot": capture_parameters(captured),
        }
    )
    commit_run(
        session.state,
        "tab",
        RunPaneState(
            result=replace(record, metadata=metadata.model_dump(mode="json")),
            parameters=captured,
            storage=captured.storage,
        ),
    )
    session.save()
    saved_snapshot = metadata_from_container(
        load(session.native_data_path)
    ).parameter_snapshot
    assert saved_snapshot is not None
    serialized_general = saved_snapshot.point.section("general")
    assert isinstance(serialized_general, dict)
    assert "entry_id" not in serialized_general
    assert serialized_general == {
        "created_at": captured.value("general.created_at"),
        "description": "captured environment",
        "ext": {"entry_id": "user-owned-extension", "note": "preserved"},
    }
    assert parameter_file.read_bytes() == source_bytes
    assert library_file.exists() is not absent_library
    control.set_parameter(target, "Q1.rate", 99.0)
    session.save()
    portable = session.entry.database_path.parent.parent / "portable.h5"
    shutil.copyfile(session.native_data_path, portable)
    session.context.clear_context()
    shutil.rmtree(session.entry.result_path)
    LoadService(session.state).load_result("tab", str(portable))
    restored = tab.run.parameters
    assert restored is not None
    assert restored.target is None
    assert restored.value("$source_qubit.rate") == 7.0
    assert restored.value("$custom.rate") == 2.0
    if absent_library:
        assert restored.library.module_keys() == ()
        assert restored.library.waveform_keys() == ()
    else:
        pulse = restored.library.module_spec("Q1.pulses.pi")
        assert isinstance(pulse, PulseSpec)
        assert pulse.gain == 0.25
    assert restored.aliases == captured.aliases
    assert require_metadata(tab.run).model_dump(mode="json") == metadata.model_dump(
        mode="json"
    )


def test_loaded_analysis_and_post_defer_unused_device_evidence(
    role_load_session: NativeSaveSession,
) -> None:
    from zcu_tools.datafile import save

    session = role_load_session
    path = session.native_data_path
    stored = load(path)
    stored.docs["metadata"]["device_snapshot"] = "malformed-unused"
    save(path, stored, replace=True)
    before = path.read_bytes()
    LoadService(session.state).load_result("tab", str(path))
    tab = session.state.get_tab("tab")
    tab.adapter = AnalysisSavingAdapter
    writeback = WritebackService(session.context, arb_names=lambda: ())
    runner = OperationRunner(
        None, session.handles, None, session.background, session.bus
    )
    analysis = AnalyzeService(
        session.state, runner, session.bus, writeback, session.handles
    )
    post = PostAnalyzeService(
        session.state, runner, session.bus, session.handles, writeback
    )

    analysis.start_analyze("tab", PaneParams(), plots=Plots(NonPresentingHost()))
    assert session.background.quiesce()
    assert tab.analysis.result == PaneSummary()
    primary_draft = tab.analysis.writeback_draft
    assert primary_draft is not None
    assert primary_draft.items == ()
    with pytest.raises(ValueError, match="/docs/metadata/device_snapshot"):
        _ = primary_draft.device_snapshot

    post.start_post_analyze("tab", PaneParams(), plots=Plots(NonPresentingHost()))
    assert session.background.quiesce()
    assert tab.post_analysis.result == PostSummary()
    post_draft = tab.post_analysis.writeback_draft
    assert post_draft is not None
    with pytest.raises(ValueError, match="/docs/metadata/device_snapshot"):
        _ = post_draft.device_snapshot
    assert path.read_bytes() == before
