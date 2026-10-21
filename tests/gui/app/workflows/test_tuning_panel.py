"""The registered Qt caller submits the same current input as external JSON edits."""

import pytest
from qtpy.QtWidgets import QLineEdit
from zcu_tools.experiment.workflows import (
    Actor,
    Engine,
)
from zcu_tools.gui.app.workflows.tuning_panel import WorkflowTuningPanel
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel
from zcu_tools.resources.entry import ResultEntry

from tests._filesystem import DIRECTORY_OPEN_ERROR
from tests.gui.app.workflows._start_fakes import StartHarness
from tests.gui.app.workflows._synthetic import TunablesForm
from tests.gui.widgets.form._text_input import commit_text

from ._sample_input import sample_input
from ._synthetic import Context
from ._window_support import show_sample_section


@pytest.fixture
def tuning(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
) -> tuple[WorkflowTuningPanel[FormModel, Context], Engine[Context]]:
    del paused_tuning
    window, _harness, _entry = attachment
    widget, engine = window.tuning_panel, window.controller.engine
    assert widget is not None and engine is not None
    show_sample_section(window, "Tunables (live)")
    return widget, engine


def test_gui_apply_and_json_edits_share_native_revision(
    tuning: tuple[WorkflowTuningPanel[FormModel, Context], Engine[Context]],
) -> None:
    widget, engine = tuning
    widget.panel.state.write({("sample", "value"): {"expr": "10 * 3"}}, FormEnv())
    applied = widget.apply()
    assert not isinstance(applied, FormErrors)
    assert applied.revision == 1
    native_values = engine.tunables("tuning").values
    assert isinstance(native_values, dict)
    assert sample_input(native_values)["value"] == 30.0
    assert sample_input(widget.host.accepted_input.data)["value"] == {"expr": "10 * 3"}
    remote = widget.apply_external(
        {("sample", "count"): 200}, actor=Actor("agent", "rpc")
    )
    assert not isinstance(remote, FormErrors)
    assert remote.revision == 2
    assert sample_input(widget.panel.state.input.data)["count"] == 200
    assert sample_input(widget.host.accepted_input.data)["value"] == {"expr": "10 * 3"}


def test_incomplete_gui_text_blocks_apply_and_discard_restores_control(
    tuning: tuple[WorkflowTuningPanel[FormModel, Context], Engine[Context]],
) -> None:
    widget, engine = tuning
    control = widget.panel.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(control, QLineEdit)
    commit_text(control, "-")
    blocked = widget.apply()
    assert isinstance(blocked, FormErrors)
    assert ("sample", "factor") in blocked.messages
    assert engine.tunables("tuning").revision == 0
    widget.discard()
    assert widget.panel.editor.local_parse_errors() is None
    restored = widget.panel.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(restored, QLineEdit)
    result = widget.panel.state.result(widget.host.env)
    assert isinstance(result, TunablesForm)
    assert float(restored.text()) == result.sample.factor


def test_external_edit_submits_pending_parsed_gui_values_in_the_same_batch(
    tuning: tuple[WorkflowTuningPanel[FormModel, Context], Engine[Context]],
) -> None:
    widget, engine = tuning
    widget.panel.state.write({("sample", "value"): 31.0}, FormEnv())
    applied = widget.apply_external(
        {("sample", "count"): 200}, actor=Actor("agent", "rpc")
    )
    assert not isinstance(applied, FormErrors)
    values = engine.tunables("tuning").values
    assert isinstance(values, dict)
    assert (sample_input(values)["value"], sample_input(values)["count"]) == (31.0, 200)
    assert widget.panel.state.input.data == widget.panel.state.base.data
    widget.panel.state.write({("sample", "value"): 50.0}, FormEnv())
    widget.discard()
    assert sample_input(widget.panel.state.input.data)["value"] == 31.0
    assert engine.tunables("tuning").revision == 1


def test_successful_external_edit_replaces_unparseable_view_text(
    tuning: tuple[WorkflowTuningPanel[FormModel, Context], Engine[Context]],
) -> None:
    widget, engine = tuning
    control = widget.panel.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(control, QLineEdit)
    commit_text(control, "-")
    applied = widget.apply_external(
        {("sample", "factor"): 8.0}, actor=Actor("agent", "rpc")
    )
    assert not isinstance(applied, FormErrors)
    assert engine.tunables("tuning").revision == 1
    assert widget.panel.editor.local_parse_errors() is None
    restored = widget.panel.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(restored, QLineEdit)
    assert float(restored.text()) == 8.0


def test_journal_failure_propagates_without_accepting_input_or_clearing_local_text(
    tuning: tuple[WorkflowTuningPanel[FormModel, Context], Engine[Context]],
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
) -> None:
    widget, engine = tuning
    before = engine.tunables("tuning")
    accepted = widget.host.accepted_input.data
    control = widget.panel.editor.findChild(QLineEdit, "form.input.sample.factor")
    assert isinstance(control, QLineEdit)
    commit_text(control, "-")
    _window, _harness, entry = attachment
    journal = entry.result_path / "workflows/tuning-paused_tuning/journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        with pytest.raises(DIRECTORY_OPEN_ERROR) as caught:
            widget.apply_external(
                {("sample", "value"): 45.0}, actor=Actor("agent", "rpc")
            )
    finally:
        journal.rmdir()
        journal.write_bytes(original)
    assert caught.value.filename == str(journal)
    assert engine.tunables("tuning") == before
    status = engine.status()
    assert status is not None and status.lifecycle == "failed"
    assert widget.host.accepted_input.data == widget.panel.state.base.data == accepted
    assert sample_input(widget.panel.state.input.data)["value"] == 45.0
    assert (
        widget.panel.editor.local_parse_errors() is not None and control.text() == "-"
    )


def test_invalid_native_batch_retains_input_without_partial_publication(
    tuning: tuple[WorkflowTuningPanel[FormModel, Context], Engine[Context]],
) -> None:
    widget, engine = tuning
    before = engine.tunables("tuning")
    blocked = widget.apply_external(
        {("sample", "count"): 200, ("sample", "value"): -1.0},
        actor=Actor("agent", "rpc"),
    )
    assert isinstance(blocked, FormErrors)
    assert ("sample", "value") in blocked.messages
    assert engine.tunables("tuning") == before
    assert sample_input(widget.panel.state.input.data)["count"] == 200
    assert sample_input(widget.panel.state.input.data)["value"] == -1.0
    assert widget.panel.state.base.data == widget.host.accepted_input.data
    widget.discard()
    assert widget.panel.state.input.data == widget.host.accepted_input.data
