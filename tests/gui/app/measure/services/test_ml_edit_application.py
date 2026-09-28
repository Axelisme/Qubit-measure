"""Library edit application commits a successful prefix through ContextService."""

from dataclasses import replace

import pytest
from zcu_tools.gui.app.measure.services.ports import CfgEdit
from zcu_tools.program.v2 import WaveformCfgFactory
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui.app.measure.remote._helpers import Fixture


@pytest.fixture()
def library_app(qapp):
    library = ModuleLibrary()
    library.waveforms["seed"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.1}
    )
    fx = Fixture(active_label="ctx001")
    fx.state.set_context(replace(fx.state.session_env, md=MetaDict(), ml=library))
    return fx.ctrl, library


@pytest.mark.parametrize("save_as", [None, "copy"])
def test_library_edit_commits_successful_prefix_and_stops(library_app, save_as):
    ctrl, library = library_app
    result = ctrl.edit_library(
        "waveform",
        "seed",
        [CfgEdit("length", 0.25), CfgEdit("unknown_field", 1), CfgEdit("length", 0.9)],
        save_as=save_as,
    )
    assert result.valid is False
    assert result.applied == 1
    assert result.errors and result.errors[0]["path"] == "unknown_field"
    assert library.waveforms[save_as or "seed"].to_dict()["length"] == 0.25
    if save_as:
        assert library.waveforms["seed"].to_dict()["length"] == 0.1


@pytest.mark.parametrize(
    "bad_edit", [CfgEdit("unknown_field", 1), CfgEdit("length", "bad")]
)
def test_library_edit_first_failure_does_not_create_destination(library_app, bad_edit):
    ctrl, library = library_app
    result = ctrl.edit_library("waveform", "seed", [bad_edit], save_as="copy")
    assert result.valid is False
    assert result.applied == 0
    assert result.errors and result.errors[0]["path"] == bad_edit.path
    assert "copy" not in library.waveforms
    assert library.waveforms["seed"].to_dict()["length"] == 0.1


def test_library_edit_success_can_be_followed_by_another_edit(library_app):
    ctrl, library = library_app
    first = ctrl.edit_library("waveform", "seed", [CfgEdit("length", 0.25)])
    second = ctrl.edit_library("waveform", "seed", [CfgEdit("length", 0.5)])
    assert first.valid and second.valid
    assert first.applied == second.applied == 1
    assert library.waveforms["seed"].to_dict()["length"] == 0.5


@pytest.mark.parametrize("path", ["length", "unknown_field"])
def test_library_edit_releases_internal_draft_on_success_and_failure(library_app, path):
    from zcu_tools.gui.expected_error import ExpectedError

    ctrl, _ = library_app
    closed = []
    ctrl.set_cfg_editor_change_listener(
        lambda editor_id, event, payload: (
            closed.append(editor_id) if event == "editor_closed" else None
        )
    )
    result = ctrl.edit_library("waveform", "seed", [CfgEdit(path, 0.25)])
    assert result.valid is (path == "length")
    assert len(closed) == 1
    with pytest.raises(ExpectedError):
        ctrl.get_cfg_editor_draft(closed[0])


def test_library_edit_save_as_collision_rejects_before_mutation(library_app):
    from zcu_tools.gui.expected_error import ExpectedError

    ctrl, library = library_app
    library.waveforms["copy"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.8}
    )
    with pytest.raises(ExpectedError):
        ctrl.edit_library("waveform", "seed", [CfgEdit("length", 0.25)], save_as="copy")
    assert library.waveforms["seed"].to_dict()["length"] == 0.1
    assert library.waveforms["copy"].to_dict()["length"] == 0.8
