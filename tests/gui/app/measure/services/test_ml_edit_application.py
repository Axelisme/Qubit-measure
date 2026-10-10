"""Library edit application commits a successful prefix through ContextService."""

from pathlib import Path
from time import monotonic

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.gui.app.measure.services.ports import CfgEdit

from tests.gui.app.measure.remote._helpers import Fixture
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@pytest.fixture()
def library_app(qapp, tmp_path: Path):
    before = registry_state()
    fx = None
    try:
        register_fakes()
        installed = registry_state()
        fx = Fixture(active_label="ctx001", project_root=str(tmp_path))
        control = fx.ctrl.context_control
        control.add_component("Q1", "fake/drive/a", {})
        parameters = control.read_context()
        control.set_waveform(
            parameters.target, "Q1.waveforms.seed", ConstWaveformCfg(length=0.1)
        )
        yield fx.ctrl
        assert registry_state() == installed
    finally:
        try:
            if fx is not None:
                fx.ctrl.set_cfg_editor_change_listener(None)
                closed = False

                def on_closed() -> None:
                    nonlocal closed
                    closed = True

                fx.ctrl.begin_shutdown(on_closed)
                deadline = monotonic() + 1.0
                while not closed and monotonic() < deadline:
                    qapp.processEvents()
                if not closed:
                    raise RuntimeError("Synchronous library fixture shutdown timed out")
                fx.stop()
                qapp.processEvents()
                # edit_library is synchronous; this fixture never submits work.
                # Shutdown is not a general BackgroundRunner quiesce replacement.
        finally:
            restore_registry(before)


@pytest.mark.parametrize("save_as", [None, "Q1.waveforms.copy"], ids=["None", "copy"])
def test_library_edit_commits_successful_prefix_and_stops(library_app, save_as):
    ctrl = library_app
    result = ctrl.edit_library(
        "waveform",
        "Q1.waveforms.seed",
        [CfgEdit("length", 0.25), CfgEdit("unknown_field", 1), CfgEdit("length", 0.9)],
        save_as=save_as,
    )
    assert result.valid is False
    assert result.applied == 1
    assert result.errors and result.errors[0]["path"] == "unknown_field"
    library = ctrl.read_context().library
    assert (
        library.entry_value(save_as or "Q1.waveforms.seed", kind="waveform")["length"]
        == 0.25
    )
    if save_as:
        assert (
            library.entry_value("Q1.waveforms.seed", kind="waveform")["length"] == 0.1
        )


@pytest.mark.parametrize(
    "bad_edit", [CfgEdit("unknown_field", 1), CfgEdit("length", "bad")]
)
def test_library_edit_first_failure_does_not_create_destination(library_app, bad_edit):
    ctrl = library_app
    result = ctrl.edit_library(
        "waveform", "Q1.waveforms.seed", [bad_edit], save_as="Q1.waveforms.copy"
    )
    assert result.valid is False
    assert result.applied == 0
    assert result.errors and result.errors[0]["path"] == bad_edit.path
    library = ctrl.read_context().library
    assert "Q1.waveforms.copy" not in library.waveform_keys()
    assert library.entry_value("Q1.waveforms.seed", kind="waveform")["length"] == 0.1


def test_library_edit_success_can_be_followed_by_another_edit(library_app):
    ctrl = library_app
    first = ctrl.edit_library(
        "waveform", "Q1.waveforms.seed", [CfgEdit("length", 0.25)]
    )
    second = ctrl.edit_library(
        "waveform", "Q1.waveforms.seed", [CfgEdit("length", 0.5)]
    )
    assert first.valid and second.valid
    assert first.applied == second.applied == 1
    assert (
        ctrl.read_context().library.entry_value("Q1.waveforms.seed", kind="waveform")[
            "length"
        ]
        == 0.5
    )


@pytest.mark.parametrize("path", ["length", "unknown_field"])
def test_library_edit_releases_internal_draft_on_success_and_failure(library_app, path):
    from zcu_tools.gui.expected_error import ExpectedError

    ctrl = library_app
    closed = []
    ctrl.set_cfg_editor_change_listener(
        lambda editor_id, event, payload: (
            closed.append(editor_id) if event == "editor_closed" else None
        )
    )
    result = ctrl.edit_library("waveform", "Q1.waveforms.seed", [CfgEdit(path, 0.25)])
    assert result.valid is (path == "length")
    assert len(closed) == 1
    with pytest.raises(ExpectedError):
        ctrl.get_cfg_editor_draft(closed[0])


def test_library_edit_save_as_collision_rejects_before_mutation(library_app):
    from zcu_tools.gui.expected_error import ExpectedError

    ctrl = library_app
    parameters = ctrl.read_context()
    ctrl.context_control.set_waveform(
        parameters.target, "Q1.waveforms.copy", ConstWaveformCfg(length=0.8)
    )
    with pytest.raises(ExpectedError):
        ctrl.edit_library(
            "waveform",
            "Q1.waveforms.seed",
            [CfgEdit("length", 0.25)],
            save_as="Q1.waveforms.copy",
        )
    library = ctrl.read_context().library
    assert library.entry_value("Q1.waveforms.seed", kind="waveform")["length"] == 0.1
    assert library.entry_value("Q1.waveforms.copy", kind="waveform")["length"] == 0.8
