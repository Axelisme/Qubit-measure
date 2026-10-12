"""Fresh typed declarations through Window/FormState and bounded loader seams."""

import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic, sleep

import pytest
from pydantic import JsonValue
from qtpy.QtWidgets import QApplication, QComboBox, QLabel
from zcu_tools.experiment.workflows import InvalidRunState
from zcu_tools.gui.catalog_loader import CatalogReloadError
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.form import FormErrors, FormInput
from zcu_tools.gui.remote.control_service import ControlOptions
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.resources.entry import ResultEntry

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.workflows.gui.app import WorkflowsGuiBehavior
from zcu_lab.workflows.gui.window import T1WorkflowWindow

from ._coherence_input import t1_input
from ._plan_inputs import inline_input
from ._reload_source import ReloadSource


def expect_properties(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Require a model input schema's object properties before field assertions."""
    properties = schema["properties"]
    assert isinstance(properties, dict)
    return properties


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
        {("t1", "guess_us"): {"expr": "8 * 4"}, ("legacy_knob",): 7},
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
    assert t1_input(saved.knobs)["guess_us"] == {"expr": "8 * 4"}
    assert saved.plan["new_field"] == 9 and saved.knobs["new_knob"] == 10
    assert window.plan_panel.state.base.data == saved.plan
    assert window.tunables_panel.state.base.data == saved.knobs
    inspected = window.inspect_form("plan")
    assert not isinstance(inspected, FormErrors)
    assert "new_field" in expect_properties(inspected["schema"])
    assert any(
        item == {"path": ["pi_length_us"], "value": 0.25}
        for item in inspected["values"]
    )
    knobs = window.inspect_form("tunables")
    assert not isinstance(knobs, FormErrors)
    assert any(
        item == {"path": ["t1", "guess_us"], "value": 32.0} for item in knobs["values"]
    )


@pytest.mark.parametrize("version", [1, 2])
def test_reload_removed_selection_uses_new_default_and_reports_both_names(
    source: ReloadSource, version: int
) -> None:
    source.write(version)
    window = source.open(StartHarness())
    old_name = window.workflow_name
    window.plan_panel.state.write(
        {("pi_length_us",): {"expr": "1 / 4"}}, window.plan_panel.env
    )
    source.write(version + 1)
    window.reload_source()
    new_name = f"reload-v{version + 1}"
    assert window.workflow_name == new_name and window.default_workflow_name == new_name
    selector = window.findChild(QComboBox, "workflow.select")
    assert selector is not None and selector.currentText() == new_name
    message = window.findChild(QLabel, "workflow.run.message")
    assert message is not None
    assert old_name in message.text() and new_name in message.text()
    assert window.capture_persisted_state().plan["pi_length_us"] == {"expr": "1 / 4"}
    with pytest.raises(InvalidInputError, match="Unknown workflow"):
        window.select_workflow(old_name)
    with pytest.raises(InvalidInputError, match="Unknown workflow"):
        window.start("unknown", name=old_name)
    assert window.workflow_name == new_name


def test_reload_retains_same_named_selection(source: ReloadSource) -> None:
    window = source.open(StartHarness())
    plan = window.plan_panel
    source.write(1)
    window.reload_source()
    assert window.workflow_name == "reload-v1"
    assert window.plan_panel is not plan
    selector = window.findChild(QComboBox, "workflow.select")
    assert selector is not None and selector.currentText() == "reload-v1"


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
    with pytest.raises(FailedPreconditionError) as failure:
        window.start("empty")
    assert failure.value.reason_code == "no_context"


def test_current_start_and_tuning_use_new_catalog(
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
    assert isinstance(window.start("new"), int)
    harness.complete_segment()
    status = window.controller.status()
    assert (
        status is not None
        and status.workflow == "reload-v2"
        and status.lifecycle == "done"
    )
    assert window.tuning_panel is not None
    assert type(window.tuning_panel.panel) is type(window.tunables_panel)
    inspected = window.inspect_form("tunables", run_id="new")
    assert not isinstance(inspected, FormErrors)
    assert "new_knob" in expect_properties(inspected["schema"])


@pytest.mark.uses_wall_clock
def test_active_reload_rejects_before_preflight_in_pending_running_and_paused(
    source: ReloadSource,
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    inputs: Inputs,
) -> None:
    _old, harness, _entry = attachment
    source.write(1, wait=True)
    window = source.open(harness)
    window.plan_panel.state.reset(
        FormInput(window.plan_panel.state.input.model, inline_input(inputs).data)
    )
    assert isinstance(window.start("active"), int)
    # Invalid source would fail preflight if admission did not run first.
    (source.root / "declaration.py").write_text("invalid python syntax !!!")
    with pytest.raises(InvalidRunState):
        window.reload_source()
    with ThreadPoolExecutor(max_workers=1) as pool:
        producer = pool.submit(harness.background.execute)
        deadline = monotonic() + 3
        while window.controller.status() is None:
            assert monotonic() < deadline, "Native acceptance did not arrive"
            sleep(0.01)
        with pytest.raises(InvalidRunState):
            window.reload_source()
        window.pause("active")
        producer.result(timeout=3)
    harness.owner.pump_all()
    harness.background.deliver()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "paused"
    with pytest.raises(InvalidRunState):
        window.reload_source()
    window.stop("active")


def test_fixed_source_change_requires_restart_even_after_disk_repair(
    source: ReloadSource,
) -> None:
    window = source.open(StartHarness())
    saved = window.capture_persisted_state()
    (source.root / "fixed.py").write_text("VERSION = 2\n")
    with pytest.raises(CatalogReloadError) as changed:
        window.reload_source()
    assert changed.value.restart_required
    assert window.capture_persisted_state() == saved
    (source.root / "fixed.py").write_text("VERSION = 1\n")
    with pytest.raises(CatalogReloadError) as repaired:
        window.reload_source()
    assert repaired.value.restart_required
    with pytest.raises(CatalogReloadError) as blocked:
        window.start("blocked")
    assert blocked.value.restart_required


def test_reload_requires_owner_thread_before_reading_source(
    source: ReloadSource,
) -> None:
    window = source.open(StartHarness())
    source.write(2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(window.reload_source)
        with pytest.raises(CatalogReloadError, match="owner thread"):
            request.result(timeout=3)
    assert window.capture_persisted_state().plan["legacy"] == 4


def test_existing_form_rpc_observes_current_catalog(source: ReloadSource) -> None:
    harness = StartHarness()
    window = source.open(harness)
    source.write(2)
    window.reload_source()
    service = window.create_remote_service(ControlOptions(port=0), harness.bus)
    link = ClientLink("reload-form", token_required=False)
    service.on_client_open(link)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(
                service.route, link, Request("1", "workflow.form", {"form": "plan"})
            )
            deadline = monotonic() + 3
            while not request.done():
                assert monotonic() < deadline, "RPC did not settle on owner"
                harness.owner.pump_once(block=True, timeout=0.01)
            request.result(timeout=1)
        reply = json.loads(link.outbound.get(timeout=1))
        assert "new_field" in reply["result"]["schema"]["properties"]
        assert reply["result"]["input"]["new_field"] == 9
    finally:
        service.on_client_close(link, on_owner_thread=True)
        service.stop()


def test_app_injects_bounded_loader_and_reloads_without_rebuilding_session(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    # Reload mutates Python's owned module cache. Restore originals after this
    # integration observation so test selection/order never changes declarations.
    from zcu_lab import workflows as original_workflows

    original_overnight = original_workflows.overnight
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
        assert t1_input(app.window.capture_persisted_state().knobs)["guess_us"] == 20.0
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
        # The public callable shares a name with its module; parent relinking
        # must not replace the package's pre-reload callable with that module.
        original_workflows.overnight = original_overnight
