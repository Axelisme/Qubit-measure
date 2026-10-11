"""Fresh typed declarations through Window/FormState and bounded loader seams."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.catalog_loader import CatalogReloadError
from zcu_tools.gui.form import FormErrors

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.gui.window import T1WorkflowWindow

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
