"""Production form ownership without tab widgets or live hardware."""

from collections.abc import Generator, Mapping
from pathlib import Path
from typing import ClassVar, NoReturn
from unittest.mock import MagicMock

import pytest
from pydantic import Field
from qtpy.QtWidgets import QApplication
from zcu_tools.cfg_model import Expr, Range
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import RunRequest
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.app_services import (
    AppServices,
    build_app_services,
)
from zcu_tools.gui.app.measure.services.form_contracts import (
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.app.measure.services.guard import GuardError, GuardService, RunInputs
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormModel
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.events import GateChangedPayload
from zcu_tools.gui.session.state import DeviceState, DeviceStatus
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import ComponentSchema, component_registry

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._synthetic import ALIAS_PRESET, SampleAdapter, SampleCfg
from tests.gui.app.measure.remote._helpers import Fixture


class SchemaAdapter(SampleAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[SampleCfg]:
        del ctx
        return Form(SampleCfg, {"gain": 1.0, "sweep": Range.between(5, 6, expts=11)})


@pytest.fixture
def application(qapp: QApplication) -> Generator[Fixture]:
    fixture = Fixture(headless=True)
    fixture.registry.register("cfg-contract", SchemaAdapter)
    try:
        yield fixture
    finally:
        fixture.stop()
        for tab_id in tuple(fixture.ctrl.list_tab_ids()):
            fixture.ctrl.close_tab(tab_id)


def test_headless_lifetime_snapshot_and_single_authority(application: Fixture) -> None:
    tab_id = application.ctrl.new_tab("cfg-contract")
    editor = application.ctrl.cfg_resources.lookup(tab_id)
    snapshot = application.ctrl.get_tab_snapshot(tab_id)
    snapshot.cfg_input["gain"] = 99.0
    assert editor.accept().values["gain"] == 1.0
    editor.write({("gain",): 7.0})
    assert application.ctrl.get_tab_snapshot(tab_id).cfg_input["gain"] == 7.0
    application.ctrl.close_tab(tab_id)
    with pytest.raises(CfgPreconditionError) as caught:
        editor.result()
    assert caught.value.reason is CfgPreconditionReason.RESOURCE_GONE
    fresh = application.ctrl.new_tab("cfg-contract")
    assert application.ctrl.cfg_resources.lookup(fresh).accept().values["gain"] == 1.0


def test_resource_write_publishes_input_and_advances_cfg_version(
    application: Fixture,
) -> None:
    tab_id = application.ctrl.new_tab("cfg-contract")
    resource = application.ctrl.cfg_resources.lookup(tab_id)
    version_key = f"tab:{tab_id}:cfg"
    before = application.state.version.get(version_key)

    resource.write({("gain",): 7.0})

    assert application.ctrl.get_tab_snapshot(tab_id).cfg_input["gain"] == 7.0
    assert application.state.version.get(version_key) > before


def test_run_rejects_unaccepted_cfg_without_preflight_or_operation(
    application: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab_id = application.ctrl.new_tab("cfg-contract")
    editor = application.ctrl.cfg_resources.lookup(tab_id)
    editor.write({("gain",): {"expr": "unfinished +"}})
    preflight = MagicMock()
    monkeypatch.setattr(
        application.state.get_tab(tab_id).adapter, "validate_run_request", preflight
    )
    with pytest.raises(CfgPreconditionError) as caught:
        application.ctrl.start_run(tab_id)
    assert caught.value.reason is CfgPreconditionReason.NOT_VALID
    preflight.assert_not_called()
    assert application.ctrl.run_analyze_control.active_tab_operations() == ()
    assert application.ctrl.get_running_tab_id() is None
    assert editor.input.data["gain"] == {"expr": "unfinished +"}


def test_definition_failure_does_not_publish_a_partial_tab(
    application: Fixture,
) -> None:
    class BrokenForm(Form[SampleCfg]):
        def default_input(self, env: FormEnv) -> FormInput[FormModel]:
            raise RuntimeError("definition failed")

    class BrokenAdapter(SchemaAdapter):
        @staticmethod
        def make_form(ctx: FormContext) -> Form[SampleCfg]:
            del ctx
            return BrokenForm(SampleCfg, {})

    tab_id = application.ctrl.new_tab("cfg-contract")
    before = application.ctrl.cfg_resources.lookup(tab_id).input.data
    application.registry.register("broken-cfg", BrokenAdapter)
    with pytest.raises(RuntimeError, match="definition failed"):
        application.ctrl.new_tab("broken-cfg")
    assert application.ctrl.list_tab_ids() == [tab_id]
    assert application.ctrl.cfg_resources.lookup(tab_id).input.data == before


def test_source_events_leave_every_tab_capture_unchanged(application: Fixture) -> None:
    parameters = application.ctrl.context_control.read_context()
    application.ctrl.context_control.set_parameter(
        parameters.target, "general.ext.x", 1.0
    )
    first = application.ctrl.new_tab("cfg-contract")
    second = application.ctrl.new_tab("cfg-contract")
    editors = [application.ctrl.cfg_resources.lookup(key) for key in (first, second)]
    publications: list[str] = []
    for editor in editors:
        editor.write({("gain",): {"expr": "general.ext.x"}})
    unsubscribe = editors[0].subscribe(lambda: publications.append("changed"))
    application.ctrl.context_control.set_parameter(
        parameters.target, "general.ext.x", 9.0
    )
    unsubscribe()
    assert publications == []
    assert [editor.accept().values["gain"] for editor in editors] == [1.0, 1.0]
    editors[0].reset()
    editors[0].write({("gain",): {"expr": "general.ext.x"}})
    assert editors[0].accept().values["gain"] == 9.0
    assert editors[1].accept().values["gain"] == 1.0


def test_cfg_notification_rejects_app_lifetime_and_run(application: Fixture) -> None:
    first = application.ctrl.new_tab("cfg-contract")
    second = application.ctrl.new_tab("cfg-contract")
    editor = application.ctrl.cfg_resources.lookup(first)
    failures: list[CfgPreconditionReason] = []

    def callback() -> None:
        for command in (
            lambda: application.ctrl.new_tab("cfg-contract"),
            lambda: application.ctrl.close_tab(second),
            lambda: application.ctrl.start_run(second),
        ):
            try:
                command()
            except CfgPreconditionError as exc:
                failures.append(exc.reason)

    unsubscribe = editor.subscribe(callback)
    editor.write({("gain",): 2.0})
    unsubscribe()
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION] * 3
    assert set(application.ctrl.list_tab_ids()) == {first, second}
    assert application.ctrl.get_running_tab_id() is None


@pytest.mark.parametrize("command", ["edit", "reset", "close"])
def test_run_registration_notification_blocks_tab_mutation(
    application: Fixture, qapp: QApplication, command: str
) -> None:
    tab_id = application.ctrl.new_tab("fake")
    cfg = application.ctrl.cfg_resources.lookup(tab_id)
    original = cfg.input.data
    outcomes: list[str] = []

    def during_registration(_event: GateChangedPayload) -> None:
        if outcomes:
            return
        outcomes.append("accepted")
        try:
            if command == "edit":
                cfg.write({("reps",): 9})
            elif command == "reset":
                cfg.reset()
            else:
                application.ctrl.close_tab(tab_id)
        except CfgPreconditionError as exc:
            if exc.reason is CfgPreconditionReason.MUTATION_BLOCKED:
                outcomes[0] = "blocked"
        except RuntimeError as exc:
            if command == "close" and "busy" in str(exc):
                outcomes[0] = "blocked"

    unsubscribe = application.bus.subscribe(GateChangedPayload, during_registration)
    try:
        application.ctrl.start_run(tab_id)
    finally:
        unsubscribe.unsubscribe()
        application.ctrl.cancel_run()
        # cancel_run() makes the submitted worker publish its terminal owner delivery.
        qt_until(
            lambda: application.ctrl.get_running_tab_id() is None, label="Run terminal"
        )
    assert outcomes == ["blocked"]
    assert cfg.input.data == original


class SourceDrive(ComponentSchema):
    gain: float
    module: dict[str, str | None] = Field(default_factory=dict)


class GlobalSourceAdapter(SampleAdapter):
    experiment_tag: ClassVar[str | None] = "fake.stub"

    @staticmethod
    def make_form(ctx: FormContext) -> Form[SampleCfg]:
        del ctx
        return Form(
            SampleCfg,
            {"gain": Expr("$qubit.gain"), "sweep": Range.between(5, 6, expts=11)},
        )

    def run(self, req: RunRequest, cfg: SampleCfg, *, context: RunContext) -> NoReturn:
        raise AssertionError("Admission-only adapter must not execute")


class MissingModuleSlotAdapter(GlobalSourceAdapter):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[SampleCfg]:
        del ctx
        return Form(
            SampleCfg,
            {
                "gain": Expr("$qubit.module.x180.gain"),
                "sweep": Range.between(5, 6, expts=11),
            },
        )


@pytest.fixture
def global_sources(
    qapp: QApplication, tmp_path: Path
) -> Generator[tuple[AppServices, State], None, None]:
    before = component_registry.kinds()
    component_registry.register("cfg/drive", SourceDrive)
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    registry = Registry()
    registry.register("global-source", GlobalSourceAdapter)
    registry.register("missing-slot", MissingModuleSlotAdapter)
    try:
        services = build_app_services(
            alias_presets={"default": ALIAS_PRESET},
            state=state,
            bus=EventBus(),
            registry=registry,
            progress_transport=MagicMock(),
            resource_versions=state.version.snapshot,
            render_host=lambda: None,
            owner_scheduler=QtOwnerScheduler(),
            project_root=str(tmp_path),
        )
        try:
            services.setup_control.new_entry(EntryRequest("sources"))
            services.context.add_component(
                "Q1", "cfg/drive", {"gain": 10.0, "module": {"x180": None}}, setup=True
            )
            services.context.add_component(
                "Q2", "cfg/drive", {"gain": 20.0}, setup=True
            )
            services.context.new_point(
                PointCreateRequest("first", "", CommandOrigin("gui"))
            )
            services.context.set_alias_rows(
                (AliasRow("qubit", "Q1"), AliasRow("custom", "Q2"))
            )
            yield services, state
        finally:
            for tab_id in tuple(state.tabs):
                services.tab.close_tab(tab_id)
            services.background.quiesce()
            services.background.deleteLater()
            qapp.processEvents()
    finally:
        component_registry.unregister("cfg/drive")
        assert component_registry.kinds() == before


def test_authored_missing_module_expression_blocks_run(
    global_sources: tuple[AppServices, State],
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("missing-slot")
    cfg = state.get_tab(tab_id).cfg
    assert "gain" not in cfg.input.data
    cfg.write({("gain",): {"expr": "$qubit.module.x180.gain"}})
    result = cfg.result()
    assert isinstance(result, FormErrors)
    assert ("gain",) in result.messages
    assert "Q1.module.x180" in result.messages[("gain",)]
    with pytest.raises(CfgPreconditionError) as caught:
        services.guard.capture_run_inputs(tab_id)
    assert caught.value.reason is CfgPreconditionReason.NOT_VALID


def test_reset_aliases_capture_once_and_run_rejects_live_override(
    global_sources: tuple[AppServices, State],
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    rows = services.context.get_alias_rows()
    with pytest.raises(GuardError) as caught:
        services.guard.capture_run_inputs(tab_id, aliases={"qubit": "Q2"})
    assert caught.value.reason_code == "run_aliases_not_supported"
    assert cfg.accept().values["gain"] == 10.0
    cfg.reset(aliases={"qubit": "Q2"})
    assert cfg.accept().values["gain"] == 20.0
    assert services.context.get_alias_rows() == rows
    admitted = services.guard.capture_run_inputs(tab_id)
    assert admitted.snapshot is not None
    assert admitted.snapshot.roles == {"qubit": "Q2", "custom": "Q2"}
    persisted = services.workspace.capture_session()
    services.workspace.close_tab(tab_id)
    assert services.workspace.apply_session(persisted).restored_tabs == 1
    reopened = state.get_tab(state.list_tab_ids()[0]).cfg
    assert reopened.input.data["gain"] == 20.0
    assert reopened.accept().values["gain"] == 20.0


@pytest.mark.parametrize("aliases", [None, {}], ids=["none", "empty"])
def test_empty_aliases_preserve_cached_acceptance_and_deferred_context(
    global_sources: tuple[AppServices, State], aliases: Mapping[str, str] | None
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    admitted = GuardService(state).capture_run_inputs(tab_id, aliases=aliases)
    assert admitted.accepted_cfg.values["gain"] == 10.0
    services.context.clear_context()
    deferred = services.tab.new_tab("global-source")
    assert services.tab.capture_parameters(deferred, aliases=aliases) is None
    with pytest.raises(FailedPreconditionError) as caught:
        services.tab.capture_parameters(deferred, aliases={"custom": "Q2"})
    assert caught.value.reason_code == "no_context"


def test_existing_tab_and_run_keep_point_until_reset(
    global_sources: tuple[AppServices, State],
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    first = services.guard.capture_run_inputs(tab_id)
    assert first.snapshot is not None
    assert first.snapshot.point == "first"
    services.context.new_point(PointCreateRequest("second", "", CommandOrigin("gui")))
    parameters = services.context.read_context()
    services.context.set_parameter(parameters.target, "Q1.gain", 30.0)
    before_reset = services.guard.capture_run_inputs(tab_id)
    assert before_reset.accepted_cfg.values["gain"] == 10.0
    assert before_reset.snapshot == first.snapshot
    state.get_tab(tab_id).cfg.reset()
    second = services.guard.capture_run_inputs(tab_id)
    assert second.accepted_cfg.values["gain"] == 30.0
    assert second.snapshot is not None
    assert second.snapshot.point == "second"
    assert second.storage is not None
    assert second.storage.point == "second"
    assert first.parameters is not None
    assert first.parameters.value("Q1.gain") == 10.0


def test_dollar_defaults_materialize_fixed_alias_capture_until_reset(
    global_sources: tuple[AppServices, State],
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    services.context.set_alias_rows((AliasRow("qubit", "Q2"), AliasRow("custom", "Q1")))
    assert cfg.accept().values["gain"] == 10.0
    assert cfg.input.data["gain"] == 10.0
    cfg.reset()
    assert cfg.accept().values["gain"] == 20.0
    assert cfg.input.data["gain"] == 20.0


@pytest.mark.parametrize("target,qubit", [(None, "Q1"), ("Q2", "Q2")])
def test_reset_target_capture_delivers_displayed_values_to_execution(
    global_sources: tuple[AppServices, State],
    monkeypatch: pytest.MonkeyPatch,
    target: str | None,
    qubit: str,
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    cfg.reset(aliases=None if target is None else {"qubit": target})
    original = cfg.input.data
    rows = services.context.get_alias_rows()
    admitted: list[RunInputs] = []

    def receive_run(
        _self: RunService, inputs: RunInputs, *, plots: Plots, origin: CommandOrigin
    ) -> int:
        admitted.append(inputs)
        return 99

    monkeypatch.setattr(RunService, "start_run", receive_run)
    assert services.run_analyze_control.start_run(tab_id) == 99
    assert admitted[0].snapshot is not None
    assert admitted[0].snapshot.roles == {"qubit": qubit, "custom": "Q2"}
    assert admitted[0].accepted_cfg.values["gain"] == (10.0 if target is None else 20.0)
    assert cfg.input.data == original
    assert services.context.get_alias_rows() == rows
    cfg.write({("gain",): 15.0})
    assert admitted[0].accepted_cfg.values["gain"] == (10.0 if target is None else 20.0)


def test_run_control_rejects_nonempty_aliases_before_execution(
    global_sources: tuple[AppServices, State], monkeypatch: pytest.MonkeyPatch
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    before = cfg.input.data
    executions: list[RunInputs] = []

    def receive_run(
        _self: RunService, inputs: RunInputs, *, plots: Plots, origin: CommandOrigin
    ) -> int:
        executions.append(inputs)
        return 99

    monkeypatch.setattr(RunService, "start_run", receive_run)
    with pytest.raises(GuardError) as caught:
        services.run_analyze_control.start_run(tab_id, aliases={"qubit": "missing"})
    assert caught.value.reason_code == "run_aliases_not_supported"
    assert executions == []
    assert cfg.input.data == before


def test_unbound_alias_reports_named_invalid_input(
    global_sources: tuple[AppServices, State],
) -> None:
    services, state = global_sources
    services.context.set_alias_rows((AliasRow("qubit", ""), AliasRow("custom", "Q2")))
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    assert "gain" not in cfg.input.data
    cfg.write({("gain",): {"expr": "$qubit.gain"}})
    result = cfg.result()
    assert isinstance(result, FormErrors)
    assert "qubit" in result.messages[("gain",)]
    with pytest.raises(CfgPreconditionError):
        services.guard.capture_run_inputs(tab_id)


@pytest.mark.parametrize("fail_submit", [False, True])
def test_run_registration_and_failed_submission_allow_operation_reads(
    application: Fixture, qapp, monkeypatch: pytest.MonkeyPatch, fail_submit: bool
) -> None:
    if fail_submit:

        def reject_submission(*_args, **_kwargs) -> None:
            raise RuntimeError("submission rejected")

        monkeypatch.setattr(BackgroundRunner, "submit", reject_submission)
    tab_id = application.ctrl.new_tab("fake")
    observed: list[tuple[bool, tuple[int, ...], tuple[int, ...]]] = []
    errors: list[str] = []

    def during_registration(_event: GateChangedPayload) -> None:
        try:
            tab_ops = application.ctrl.run_analyze_control.active_tab_operations()
            all_ops = application.ctrl.operation_control.active_operations()
            observed.append(
                (
                    application.ctrl.get_running_tab_id() == tab_id,
                    tuple(op.op for op in tab_ops),
                    tuple(op.op for op in all_ops),
                )
            )
        except RuntimeError as exc:
            errors.append(str(exc))

    subscription = application.bus.subscribe(GateChangedPayload, during_registration)
    try:
        if fail_submit:
            with pytest.raises(RuntimeError, match="submission rejected"):
                application.ctrl.start_run(tab_id)
            assert application.ctrl.get_running_tab_id() is None
            assert application.ctrl.operation_control.active_operations() == ()
        else:
            token = application.ctrl.start_run(tab_id)
            assert tuple(
                op.op for op in application.ctrl.operation_control.active_operations()
            ) == (token,)
    finally:
        subscription.unsubscribe()
        application.ctrl.cancel_run()
        # cancel_run() makes the submitted worker publish its terminal owner delivery.
        qt_until(
            lambda: application.ctrl.get_running_tab_id() is None, label="Run terminal"
        )
    assert errors == []
    assert observed == [(True, (), ())] * (2 if fail_submit else 1)


class CustomCachedInfo(BaseDeviceInfo):
    """A non-native device has a nested cached publication, not a live provider."""

    type: str = "custom"
    notes: dict[str, float]


def test_open_captures_detached_native_and_custom_infos_but_not_absent_info(
    application: Fixture,
) -> None:
    native = FakeDeviceInfo(address="native", value=0.25)
    custom = CustomCachedInfo(address="custom", notes={"gain": 1.0})
    for name, info in (("native", native), ("custom", custom), ("absent", None)):
        application.state.put_device(
            DeviceState(
                name=name,
                type_name="memory",
                address=name,
                remember=True,
                status=DeviceStatus.MEMORY_ONLY,
                info=info,
            )
        )
    tab_id = application.ctrl.new_tab("cfg-contract")
    resource = application.ctrl.cfg_resources.lookup(tab_id)
    custom.notes["gain"] = 9.0
    application.state.set_device_info(
        "native", FakeDeviceInfo(address="native", value=0.75)
    )
    assert set(resource.env.devices) == {"native", "custom"}
    assert resource.env.devices["native"].model_dump()["value"] == 0.25
    assert resource.env.devices["custom"].model_dump()["notes"] == {"gain": 1.0}
    resource.reset()
    assert resource.env.devices["native"].model_dump()["value"] == 0.75
    assert resource.env.devices["custom"].model_dump()["notes"] == {"gain": 9.0}
