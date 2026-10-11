"""Fresh typed declarations through Window/FormState and bounded loader seams."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.catalog_loader import CatalogReloadError
from zcu_tools.gui.form import FormErrors, FormInput
from zcu_tools.resources.entry import ResultEntry

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.workflows.gui.app import WorkflowsGuiBehavior
from zcu_lab.workflows.gui.window import T1WorkflowWindow

from ._plan_inputs import inline_input
from ._reload_source import ReloadSource


@pytest.fixture
def source(tmp_path: Path, qapp: QApplication) -> Iterator[ReloadSource]:
    fixture = ReloadSource(tmp_path)
    try:
        yield fixture
    finally:
        fixture.close(qapp)


def test_reload_replaces_declarations_and_preserves_compatible_drafts(
    source: ReloadSource,
) -> None:
    assert callable(getattr(T1WorkflowWindow, "reload_source", None)), (
        "Window must reload typed source declarations without replacing session owners"
    )
    harness = StartHarness()
    window = source.open(harness)
    controller = window.controller
    old_plan = window.plan_panel
    old_knobs = window.tunables_panel
    window.plan_panel.state.write(
        {("pi_length_us",): {"expr": "1 / 4"}, ("legacy",): 6},
        window.plan_panel.env,
    )
    window.tunables_panel.state.write(
        {("guess_us",): {"expr": "8 * 4"}, ("legacy_knob",): 7},
        window.tunables_panel.env,
    )
    source.write(2)
    report = window.reload_source()
    assert report.plan_dropped == (("legacy",),)
    assert report.knobs_dropped == (("legacy_knob",),)
    assert window.controller is controller
    assert window.plan_panel is not old_plan and window.tunables_panel is not old_knobs
    saved = window.capture_persisted_state()
    assert saved.plan["pi_length_us"] == {"expr": "1 / 4"}
    assert saved.knobs["guess_us"] == {"expr": "8 * 4"}
    assert saved.plan["new_field"] == 9 and saved.knobs["new_knob"] == 10
    assert window.plan_panel.state.base.data == saved.plan
    assert window.tunables_panel.state.base.data == saved.knobs
    inspected = window.inspect_form("plan")
    assert not isinstance(inspected, FormErrors)
    assert "new_field" in inspected["schema"]["properties"]
    assert any(
        item == {"path": ["pi_length_us"], "value": 0.25}
        for item in inspected["values"]
    )
    knobs = window.inspect_form("tunables")
    assert not isinstance(knobs, FormErrors)
    assert any(
        item == {"path": ["guess_us"], "value": 32.0} for item in knobs["values"]
    )


def test_failed_import_retains_draft_blocks_start_and_recovers_after_retry(
    source: ReloadSource,
) -> None:
    harness = StartHarness()
    window = source.open(harness)
    window.plan_panel.state.write(
        {("pi_length_us",): {"expr": "1 / 4"}}, window.plan_panel.env
    )
    saved = window.capture_persisted_state()
    plan, knobs = window.plan_panel, window.tunables_panel
    source.write(2, failure='raise ValueError("broken import")')
    with pytest.raises(CatalogReloadError, match="broken import"):
        window.reload_source()
    assert window.plan_panel is plan and window.tunables_panel is knobs
    assert window.capture_persisted_state() == saved
    with pytest.raises(CatalogReloadError, match="unavailable"):
        window.start("blocked")
    source.write(2)
    report = window.reload_source()
    assert report.plan_dropped == (("legacy",),)
    assert window.capture_persisted_state().plan["pi_length_us"] == {"expr": "1 / 4"}
    assert window.plan_panel is not plan
    # Successful retry clears reload admission; ordinary EMPTY-source admission
    # is now the only reason Start cannot run this source-free view.
    with pytest.raises(RuntimeError, match="parameter"):
        window.start("empty")


def test_current_start_and_tuning_use_new_catalog_but_old_export_stays_fixed(
    source: ReloadSource,
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
) -> None:
    _old, harness, _entry = attachment
    window = source.open(harness)
    # Use the same public ordinary input with the newly declared model.
    window.plan_panel.state.reset(
        FormInput(window.plan_panel.state.input.model, inline_input(inputs).data)
    )
    assert isinstance(window.start("old"), int)
    harness.complete_segment()
    status = window.controller.status()
    assert (
        status is not None
        and status.workflow == "reload-v1"
        and status.lifecycle == "done"
    )
    source.write(2)
    window.reload_source()
    window.plan_panel.state.write({("points",): 33}, window.plan_panel.env)
    with pytest.raises(ValueError, match="export-v1:21"):
        window.export_fluxdep("old")
    assert isinstance(window.start("new"), int)
    harness.complete_segment()
    status = window.controller.status()
    assert (
        status is not None
        and status.workflow == "reload-v2"
        and status.lifecycle == "done"
    )
    with pytest.raises(ValueError, match="export-v2:33"):
        window.export_fluxdep("new")
    assert window.tuning_panel is not None
    assert type(window.tuning_panel.panel) is type(window.tunables_panel)
    inspected = window.inspect_form("tunables", run_id="new")
    assert not isinstance(inspected, FormErrors)
    assert "new_knob" in inspected["schema"]["properties"]


def test_app_injects_bounded_loader_and_reloads_without_rebuilding_session(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    # Reload mutates Python's owned module cache. Restore originals after this
    # integration observation so test selection/order never changes declarations.
    original = {
        name: module
        for name, module in sys.modules.items()
        if name == "zcu_lab.workflows" or name.startswith("zcu_lab.workflows.")
    }
    app = WorkflowsGuiBehavior(clean=True, cache_dir=tmp_path)
    app.assemble(None)
    state, services, controller = app.state, app.services, app.window.controller
    try:
        app.window.reload_source()
        assert app.state is state and app.services is services
        assert app.window.controller is controller
        assert app.window.capture_persisted_state().knobs["guess_us"] == 20.0
    finally:
        app.window.close()
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()
        for name in tuple(sys.modules):
            if name == "zcu_lab.workflows" or name.startswith("zcu_lab.workflows."):
                del sys.modules[name]
        sys.modules.update(original)
        for name, module in original.items():
            parent, _, leaf = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], leaf, module)
