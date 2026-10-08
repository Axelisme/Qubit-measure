"""Concrete Qt startup uses one fixed native capture and no hardware."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import Done, InitEnv, Step, WorkflowEnv, workflow
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleSpec,
    component_registry,
    role_registry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from tests.resources.entry.fakes import registry_state, restore_registry
from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.components import QubitSchema
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.gui import window as window_module
from zcu_lab.workflows.gui.tunables import T1TunablesForm
from zcu_lab.workflows.gui.window import T1WorkflowWindow
from zcu_lab.workflows.t1 import T1Plan, T1Tunables

from ._plan_inputs import inline_input


@pytest.fixture
def qapp() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


@pytest.fixture
def attachment(
    qapp: QApplication,
    tmp_path: Path,
    inputs: Inputs,
) -> Iterator[tuple[T1WorkflowWindow, StartHarness, ResultEntry]]:
    before = registry_state()
    kind = "qubit/workflow-window-test"
    component_registry.register(kind, QubitSchema)
    try:
        role_registry.get("qubit")
    except ValueError:
        role_registry.register("qubit", RoleSpec("qubit/*"))
    installed = registry_state()
    harness = StartHarness()
    window = None
    try:
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component("Q1", kind=kind)
        entry.new_point("source", description="captured description")
        context = harness.services.context
        context.install_entry(entry)
        context.use_point(PointUseRequest("source"))
        context.set_component_focus("Q1")
        context.set_parameter(
            context.read_context().target, "general.ext.guess_us", 20.0
        )
        harness.state.set_context(
            replace(
                harness.state.session_env,
                soc=inputs.soc,
                soccfg=inputs.soccfg,
            )
        )
        harness.services.device.start_connect_device(
            ConnectDeviceRequest("FakeDevice", "flux", "none")
        )
        harness.background.execute()
        harness.background.deliver()
        window = T1WorkflowWindow(
            harness.state, harness.services, harness.runner, harness.owner
        )
        window.plan_panel.state.reset(inline_input(inputs))
        yield window, harness, entry
    finally:
        if window is not None:
            if harness.background.work is not None:
                window.stop()
                harness.background.execute()
                harness.background.deliver()
            window.close()
            window.deleteLater()
            qapp.processEvents()
        try:
            assert registry_state() == installed, (
                "workflow window test polluted registries"
            )
        finally:
            restore_registry(before)


def test_invalid_pi_does_not_submit_or_create_workflow_directories(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    window.plan_panel.state.write({("pi_length_us",): 0.0}, FormEnv())
    result = window.start("invalid")
    assert isinstance(result, FormErrors) and ("pi_length_us",) in result.messages
    assert window.controller is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()


def test_startup_source_failure_is_not_replaced_by_old_constructor_capture(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    harness.services.context.clear_context()
    with pytest.raises(FailedPreconditionError):
        window.start("empty")
    assert window.controller is None and harness.background.work is None
    assert not (entry.result_path / "workflows").exists()


def test_start_captures_current_values_then_keeps_them_fixed_until_execute(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, entry = attachment
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 31.0)
    window.plan_panel.state.write({("pi_length_us",): {"expr": "0.06 * 2"}}, FormEnv())
    observed: list[tuple[Calibrated, RunSnapshot, float]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> None:
        return None

    @workflow("capture-observer", plan=T1Plan, tunables=T1Tunables, init=init)
    def observe(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: None
    ) -> Step[None, int]:
        del state
        observed.append((env.context, plan.snapshot, tun.guess_us))
        yield from ()
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", observe)
    token = window.start("captured")
    assert isinstance(token, int)
    controller = window.controller
    assert controller is not None and controller.status() is None
    assert not (entry.result_path / "workflows/captured").exists()
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 99.0)
    entry.new_point("later", description="later description")
    context.use_point(PointUseRequest("later"))
    harness.background.execute()
    assert len(observed) == 1
    seed, snapshot, guess = observed[0]
    assert seed.pi_length_us == 0.12 and guess == 31.0
    assert str(snapshot.entry_id) == entry.entry_id
    assert snapshot.entry_name == "entry" and snapshot.point == "source"
    assert snapshot.description == "captured description" and snapshot.roles == {
        "qubit": "Q1"
    }
    status = controller.status()
    assert status is not None and status.lifecycle == "done"
    assert harness.gate.snapshot()
    harness.background.deliver()
    assert not harness.gate.snapshot()
    assert (entry.result_path / "workflows/captured").is_dir()
    assert (entry.database_path / "workflows/captured/manifest.json").is_file()
    assert window.tuning_panel is not None
    assert window.tuning_panel.host.accepted_input.data["guess_us"] == {
        "expr": "general.ext.guess_us"
    }


def test_submit_failure_leaves_no_accepted_run_or_paths(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    harness.background.fail_submit = True
    with pytest.raises(OSError, match="submit failed"):
        window.start("submit-error")
    assert window.controller is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()


def test_close_pending_start_requests_stop_without_destroying_producer_host(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    token = window.start("close-pending")
    assert isinstance(token, int)
    assert not window.close()
    assert harness.gate.snapshot()
    harness.background.execute()
    harness.background.deliver()
    assert not harness.gate.snapshot()
    assert window.close()
    assert not (entry.result_path / "workflows").exists()


def test_real_t1_gui_caller_runs_native_offline_acquisitions(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
) -> None:
    window, harness, entry = attachment
    window.plan_panel.state.write(
        {("fluxes",): {"kind": "list", "values": [1.0]}}, FormEnv()
    )
    window.tunables_panel.state.reset(
        FormInput(
            T1TunablesForm,
            {
                "cal": {"span_mhz": 2.0, "rabi_start_factor": 0.125},
                "rounds": 2,
            },
        )
    )
    token = window.start("native")
    assert isinstance(token, int)
    harness.background.execute()
    harness.background.deliver()
    controller = window.controller
    assert controller is not None
    status = controller.status()
    assert status is not None and status.lifecycle == "done"
    assert status.committed_seq == 1
    engine = controller.engine
    assert engine is not None
    record = engine.records(status.run_id)[0]
    assert len(record.run_files) >= 4
    for relative in record.run_files:
        assert (
            entry.result_path / "workflows/native" / record.iteration_dir / relative
        ).is_file()
    assert window.tuning_panel is not None and not window.tuning_panel.isEnabled()
    assert not harness.gate.snapshot()
