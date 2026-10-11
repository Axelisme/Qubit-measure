"""Production cfg owner wiring without tab widgets or live hardware."""

from collections.abc import Generator, Mapping
from pathlib import Path
from time import monotonic, sleep
from typing import ClassVar, NoReturn
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import RunRequest, SessionEnv
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.app_services import (
    AppServices,
    build_app_services,
)
from zcu_tools.gui.app.measure.services.guard import GuardService, RunInputs
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.cfg.model import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgObservation,
    CfgPreconditionError,
    CfgPreconditionReason,
    CfgStaleError,
    CfgStatus,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    GateChangedPayload,
)
from zcu_tools.gui.session.state import DeviceState, DeviceStatus
from zcu_tools.plotting.plots import Plots
from zcu_tools.resources.entry import (
    ComponentSchema,
    RoleResolutionError,
    component_registry,
)

from tests.gui.app.measure.remote._helpers import Fixture
from zcu_lab.definitions import ALIAS_PRESET
from zcu_lab.v2.fake.stub.gui import FakeAdapter


class SchemaAdapter(FakeAdapter):
    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        return CfgSchema(
            CfgSectionSpec(fields={"gain": ScalarSpec("Gain", float)}),
            CfgSectionValue(fields={"gain": DirectValue(1.0)}),
        )


@pytest.fixture
def application(qapp):
    fixture = Fixture(headless=True)
    fixture.registry.register("cfg-contract", SchemaAdapter)
    yield fixture
    for tab_id in tuple(fixture.ctrl.list_tab_ids()):
        fixture.ctrl.close_tab(tab_id)


def test_headless_lifetime_snapshot_and_single_authority(application: Fixture) -> None:
    tab_id = application.ctrl.new_tab("cfg-contract")
    editor = application.ctrl.cfg_resources.lookup(tab_id)
    before = editor.observe()
    assert before.status is CfgStatus.VALID
    snapshot = application.ctrl.get_tab_snapshot(tab_id)
    snapshot.cfg_schema.value.fields["gain"] = DirectValue(99.0)
    assert application.state.get_tab(tab_id).cfg.accept(before.ref.revision).values == {
        "gain": 1.0
    }
    editor.edit(before.ref.revision, (CfgEdit(("gain",), 7.0),))
    assert application.ctrl.get_tab_snapshot(tab_id).cfg_schema.value.fields[
        "gain"
    ] == DirectValue(7.0)
    application.ctrl.close_tab(tab_id)
    with pytest.raises(CfgPreconditionError) as caught:
        editor.observe()
    assert caught.value.reason is CfgPreconditionReason.RESOURCE_GONE
    recreated = application.ctrl.new_tab("cfg-contract")
    assert (
        application.ctrl.cfg_resources.lookup(recreated).observe().ref.cfg_id
        != before.ref.cfg_id
    )


@pytest.mark.parametrize("failure", ["stale", "invalid", "wrong_resource"])
def test_run_rejects_unaccepted_cfg_without_preflight_or_operation(
    application: Fixture, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    tab_id = application.ctrl.new_tab("cfg-contract")
    editor = application.ctrl.cfg_resources.lookup(tab_id)
    expected = editor.observe().ref
    publication = editor.edit(
        expected.revision,
        (
            CfgEdit(
                ("gain",),
                DirectValue(None, raw="unfinished") if failure == "invalid" else 2.0,
            ),
        ),
    )
    if failure == "invalid":
        expected = publication.ref
    elif failure == "wrong_resource":
        other = application.ctrl.new_tab("cfg-contract")
        expected = application.ctrl.cfg_resources.lookup(other).observe().ref
    preflight = MagicMock()
    monkeypatch.setattr(
        application.state.get_tab(tab_id).adapter, "validate_run_request", preflight
    )

    with pytest.raises(CfgPreconditionError) as caught:
        application.ctrl.start_run(tab_id, expected)

    if failure == "invalid":
        assert caught.value.reason is CfgPreconditionReason.NOT_VALID
    else:
        assert isinstance(caught.value, CfgStaleError)
        assert caught.value.actual == publication.ref
    preflight.assert_not_called()
    assert application.ctrl.run_analyze_control.active_tab_operations() == ()
    assert application.ctrl.get_running_tab_id() is None
    assert editor.observe().ref == publication.ref


def test_definition_failure_does_not_publish_a_partial_tab(
    application: Fixture,
) -> None:
    class BrokenAdapter(SchemaAdapter):
        def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
            raise RuntimeError("definition failed")

    tab_id = application.ctrl.new_tab("cfg-contract")
    before = application.ctrl.cfg_resources.lookup(tab_id).observe()
    application.registry.register("broken-cfg", BrokenAdapter)
    with pytest.raises(RuntimeError, match="definition failed"):
        application.ctrl.new_tab("broken-cfg")
    assert application.ctrl.list_tab_ids() == [tab_id]
    assert application.ctrl.cfg_resources.lookup(tab_id).observe() == before


def test_production_capture_uses_published_cache_without_live_provider(
    application: Fixture, monkeypatch
) -> None:
    application.state.put_device(
        DeviceState(
            name="flux",
            type_name="FakeDevice",
            address="none",
            remember=True,
            status=DeviceStatus.CONNECTED,
            info=FakeDeviceInfo(address="none", value=0.25),
        )
    )
    application.bus.emit(DeviceChangedPayload(name="flux"))
    tab_id = application.ctrl.new_tab("cfg-contract")
    editor = application.ctrl.cfg_resources.lookup(tab_id)

    def live_read_forbidden(*_args, **_kwargs):
        raise AssertionError("Capture must not query a live value provider")

    monkeypatch.setattr(application.ctrl, "read_value_source", live_read_forbidden)
    published = editor.edit(
        editor.observe().ref.revision,
        (CfgEdit(("gain",), EvalValue("@{device.flux.value}")),),
    )
    accepted = application.state.get_tab(tab_id).cfg.accept(published.ref.revision)
    assert accepted.values == {"gain": 0.25}
    assert any(source.source_id == "values" for source in accepted.source_basis)
    application.state.set_device_info(
        "flux", FakeDeviceInfo(address="none", value=0.75)
    )
    application.bus.emit(DeviceChangedPayload(name="flux"))
    after = editor.observe()
    assert after.ref.revision > published.ref.revision
    assert application.state.get_tab(tab_id).cfg.accept(after.ref.revision).values == {
        "gain": 0.25
    }
    assert accepted.values == {"gain": 0.25}


def test_source_event_publishes_every_tab_before_callback(application: Fixture) -> None:
    parameters = application.ctrl.context_control.read_context()
    application.ctrl.context_control.set_parameter(
        parameters.target, "general.ext.x", 1.0
    )
    first = application.ctrl.new_tab("cfg-contract")
    second = application.ctrl.new_tab("cfg-contract")
    editors = [
        application.ctrl.cfg_resources.lookup(tab_id) for tab_id in (first, second)
    ]
    for editor in editors:
        editor.edit(
            editor.observe().ref.revision,
            (CfgEdit(("gain",), EvalValue("general.ext.x")),),
        )
    before = editors[0].observe().ref
    observed = []

    def callback(publication: CfgObservation) -> None:
        if publication.ref != before:
            observed.append(tuple(editor.observe() for editor in editors))

    unsubscribe = editors[0].watch(callback)
    application.ctrl.context_control.set_parameter(
        parameters.target, "general.ext.x", 9.0
    )
    unsubscribe()
    assert len(observed) == 1
    assert observed[0][0].source_basis == observed[0][1].source_basis
    for tab_id, publication in zip((first, second), observed[0], strict=True):
        assert application.state.get_tab(tab_id).cfg.accept(
            publication.ref.revision
        ).values == {"gain": 9.0}


def test_cfg_notification_rejects_app_lifetime_and_run(application: Fixture) -> None:
    first = application.ctrl.new_tab("cfg-contract")
    second = application.ctrl.new_tab("cfg-contract")
    editor = application.ctrl.cfg_resources.lookup(first)
    initial = editor.observe().ref
    failures = []

    def callback(publication: CfgObservation) -> None:
        if publication.ref == initial:
            return
        for command in (
            lambda: application.ctrl.new_tab("cfg-contract"),
            lambda: application.ctrl.close_tab(second),
            lambda: application.ctrl.start_run(
                second, application.ctrl.cfg_resources.lookup(second).observe().ref
            ),
        ):
            try:
                command()
            except CfgPreconditionError as exc:
                failures.append(exc.reason)

    unsubscribe = editor.watch(callback)
    editor.edit(initial.revision, (CfgEdit(("gain",), 2.0),))
    unsubscribe()
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION] * 3
    assert set(application.ctrl.list_tab_ids()) == {first, second}
    assert application.ctrl.get_running_tab_id() is None


@pytest.mark.parametrize("command", ["edit", "reset", "replace", "close"])
def test_run_registration_notification_blocks_tab_mutation(
    application: Fixture, qapp, command: str
) -> None:
    tab_id = application.ctrl.new_tab("fake")
    cfg = application.state.get_tab(tab_id).cfg
    expected = cfg.observe().ref
    original = cfg.snapshot_inputs()
    outcomes: list[str] = []

    def during_registration(_event: GateChangedPayload) -> None:
        if outcomes:
            return
        outcomes.append("accepted")
        try:
            if command == "edit":
                cfg.edit(expected.revision, (CfgEdit(("reps",), 9),))
            elif command == "reset":
                cfg.reset(expected.revision)
            elif command == "replace":
                cfg.replace_inputs(expected.revision, original)
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
        application.ctrl.start_run(tab_id, expected)
    finally:
        unsubscribe.unsubscribe()
        application.ctrl.cancel_run()
        deadline = monotonic() + 3.0
        while (
            application.ctrl.get_running_tab_id() is not None and monotonic() < deadline
        ):
            qapp.processEvents()
            sleep(0.001)
        assert application.ctrl.get_running_tab_id() is None
    assert outcomes == ["blocked"]
    assert cfg.observe().ref == expected
    assert cfg.snapshot_inputs() == original


class SourceDrive(ComponentSchema):
    gain: float


class GlobalSourceAdapter(FakeAdapter):
    experiment_tag: ClassVar[str | None] = "fake.stub"

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, object],
        *,
        context: RunContext,
    ) -> NoReturn:
        """Reject execution: these source-contract tests end at admission."""
        raise AssertionError("Admission-only adapter must not execute")

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        schema = super().make_default_cfg(ctx)
        schema.value.fields["gain"] = EvalValue("Q1.gain")
        return schema


@pytest.fixture
def global_sources(
    qapp: QApplication, tmp_path: Path
) -> Generator[tuple[AppServices, State], None, None]:
    before = component_registry.kinds()
    component_registry.register("cfg/drive", SourceDrive)
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    registry = Registry()
    registry.register("global-source", GlobalSourceAdapter)
    try:
        services = build_app_services(
            alias_preset=ALIAS_PRESET,
            state=state,
            bus=EventBus(),
            registry=registry,
            cfg_editor_ctrl=MagicMock(),
            progress_transport=MagicMock(),
            resource_versions=state.version.snapshot,
            render_host=lambda: None,
            owner_scheduler=QtOwnerScheduler(),
            project_root=str(tmp_path),
        )
        try:
            services.setup_control.new_entry(EntryRequest("sources"))
            services.context.add_component(
                "Q1", "cfg/drive", {"gain": 10.0}, setup=True
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


def test_named_aliases_preview_and_admission_share_capture_without_publication(
    global_sources: tuple[AppServices, State],
) -> None:
    services, state = global_sources
    services.context.set_alias_rows((AliasRow("custom", "Q2"), AliasRow("qubit", "Q1")))
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    expression = "2 * $custom.gain + $qubit.gain"
    observed = cfg.edit(
        cfg.observe().ref.revision, (CfgEdit(("gain",), EvalValue(expression)),)
    )
    rows = services.context.get_alias_rows()
    versions = state.version.snapshot()
    aliases = {"custom": "Q1"}
    preview = services.tab.preview_cfg(tab_id, observed.ref, aliases=aliases)
    assert preview.ref == observed.ref
    assert preview.status is CfgStatus.VALID
    admitted = services.guard.capture_run_inputs(
        tab_id, expected_revision=observed.ref.revision, aliases=aliases
    )
    assert admitted.accepted_cfg.ref == observed.ref
    assert admitted.accepted_cfg.values["gain"] == 30.0
    assert admitted.snapshot is not None
    assert admitted.snapshot.roles == {"custom": "Q1", "qubit": "Q1"}
    assert admitted.parameters is not None
    assert admitted.parameters.value("$custom.gain") == 10.0
    assert admitted.parameters.value("$qubit.gain") == 10.0
    assert cfg.accept(observed.ref.revision).values["gain"] == 50.0
    assert cfg.observe() == observed
    assert services.context.get_alias_rows() == rows
    assert state.version.snapshot() == versions
    services.context.set_parameter(admitted.parameters.target, "Q1.gain", 40.0)
    assert admitted.parameters.value("$custom.gain") == 10.0
    assert admitted.accepted_cfg.values["gain"] == 30.0
    with pytest.raises(CfgStaleError):
        services.tab.preview_cfg(tab_id, observed.ref, aliases=aliases)
    with pytest.raises(CfgStaleError):
        services.guard.capture_run_inputs(
            tab_id, expected_revision=observed.ref.revision, aliases=aliases
        )


@pytest.mark.parametrize("aliases", [None, {}], ids=["none", "empty"])
def test_empty_aliases_preserve_cached_acceptance_and_deferred_context(
    global_sources: tuple[AppServices, State], aliases: Mapping[str, str] | None
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    observed = state.get_tab(tab_id).cfg.observe()

    def unavailable_resolution(_parameters: object) -> NoReturn:
        raise AssertionError("No override must accept the cached publication")

    guard = GuardService(
        state,
        captured_parameters=lambda overrides: services.tab.capture_parameters(
            tab_id, aliases=overrides
        ),
        cfg_resolution=unavailable_resolution,
    )
    admitted = guard.capture_run_inputs(
        tab_id, expected_revision=observed.ref.revision, aliases=aliases
    )
    assert admitted.accepted_cfg.values["gain"] == 10.0
    assert admitted.accepted_cfg.ref == observed.ref
    services.context.clear_context()
    deferred = services.tab.new_tab("global-source")
    assert services.tab.capture_parameters(deferred, aliases=aliases) is None
    with pytest.raises(FailedPreconditionError) as caught:
        services.tab.capture_parameters(deferred, aliases={"custom": "Q2"})
    assert caught.value.reason_code == "no_context"


def test_existing_tab_and_run_admission_use_current_point(
    global_sources: tuple[AppServices, State],
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    first = services.guard.capture_run_inputs(
        tab_id, expected_revision=cfg.observe().ref.revision
    )
    assert first.accepted_cfg.values["gain"] == 10.0
    assert first.snapshot is not None
    assert first.snapshot.point == "first"
    assert first.snapshot.roles == {"qubit": "Q1", "custom": "Q2"}

    services.context.new_point(PointCreateRequest("second", "", CommandOrigin("gui")))
    parameters = services.context.read_context()
    services.context.set_parameter(parameters.target, "Q1.gain", 30.0)
    second = services.guard.capture_run_inputs(
        tab_id, expected_revision=cfg.observe().ref.revision
    )
    assert second.accepted_cfg.values["gain"] == 30.0
    assert second.snapshot is not None
    assert second.snapshot.point == "second"
    assert second.parameters is not None
    assert second.parameters.value("Q1.gain") == 30.0
    assert second.storage is not None
    assert second.storage.point == "second"
    assert first.parameters is not None
    assert first.parameters.value("Q1.gain") == 10.0
    assert first.snapshot.point == "first"
    assert first.storage is not None
    assert first.storage.point == "first"


def test_dollar_cfg_refreshes_with_aliases_and_retains_authored_input(
    global_sources: tuple[AppServices, State],
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    initial = cfg.edit(
        cfg.observe().ref.revision, (CfgEdit(("gain",), EvalValue("$qubit.gain")),)
    )
    assert cfg.accept(initial.ref.revision).values["gain"] == 10.0
    services.context.set_alias_rows((AliasRow("qubit", "Q2"), AliasRow("custom", "Q1")))
    refreshed = cfg.observe()
    assert refreshed.ref.cfg_id == initial.ref.cfg_id
    assert cfg.accept(refreshed.ref.revision).values["gain"] == 20.0
    value = refreshed.tree.children["gain"].value
    assert isinstance(value, EvalValue) and value.expr == "$qubit.gain"


def test_run_target_capture_does_not_mutate_global_rows_or_cfg(
    global_sources: tuple[AppServices, State],
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    observed = cfg.observe()
    rows = services.context.get_alias_rows()
    observed = cfg.edit(
        observed.ref.revision, (CfgEdit(("gain",), EvalValue("$qubit.gain")),)
    )
    admitted = services.guard.capture_run_inputs(
        tab_id, expected_revision=observed.ref.revision, aliases={"qubit": "Q2"}
    )
    assert admitted.snapshot is not None
    assert admitted.snapshot.roles == {"qubit": "Q2", "custom": "Q2"}
    assert admitted.parameters is not None
    assert admitted.parameters.value("$qubit.gain") == 20.0
    assert admitted.accepted_cfg.values["gain"] == 20.0
    assert cfg.accept(observed.ref.revision).values["gain"] == 10.0
    assert services.context.get_alias_rows() == rows
    assert cfg.observe() == observed
    cfg.edit(observed.ref.revision, (CfgEdit(("gain",), 15.0),))
    with pytest.raises(CfgStaleError):
        services.guard.capture_run_inputs(
            tab_id, expected_revision=observed.ref.revision, aliases={"qubit": "Q2"}
        )


@pytest.mark.parametrize("target,qubit", [(None, "Q1"), ("Q2", "Q2")])
def test_run_control_delivers_target_capture_to_execution_without_publication_change(
    global_sources: tuple[AppServices, State],
    monkeypatch: pytest.MonkeyPatch,
    target: str | None,
    qubit: str,
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    observed = cfg.observe()
    rows = services.context.get_alias_rows()
    observed = cfg.edit(
        observed.ref.revision, (CfgEdit(("gain",), EvalValue("$qubit.gain")),)
    )
    admitted: list[RunInputs] = []

    def receive_run(
        _self: RunService, inputs: RunInputs, *, plots: Plots, origin: CommandOrigin
    ) -> int:
        admitted.append(inputs)
        return 99

    monkeypatch.setattr(RunService, "start_run", receive_run)
    assert (
        services.run_analyze_control.start_run(
            tab_id, observed.ref, aliases=None if target is None else {"qubit": target}
        )
        == 99
    )
    assert len(admitted) == 1
    inputs = admitted[0]
    assert inputs.snapshot is not None
    assert inputs.snapshot.roles == {"qubit": qubit, "custom": "Q2"}
    assert inputs.parameters is not None
    assert inputs.parameters.value("$qubit.gain") == (10.0 if target is None else 20.0)
    assert inputs.accepted_cfg.ref == observed.ref
    assert inputs.accepted_cfg.values["gain"] == (10.0 if target is None else 20.0)
    assert cfg.accept(observed.ref.revision).values["gain"] == 10.0
    assert cfg.observe() == observed
    assert services.context.get_alias_rows() == rows
    cfg.edit(observed.ref.revision, (CfgEdit(("gain",), 15.0),))
    with pytest.raises(CfgStaleError):
        services.run_analyze_control.start_run(
            tab_id, observed.ref, aliases=None if target is None else {"qubit": target}
        )
    assert len(admitted) == 1


def test_run_control_rejects_unresolvable_target_before_execution(
    global_sources: tuple[AppServices, State],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, state = global_sources
    tab_id = services.tab.new_tab("global-source")
    cfg = state.get_tab(tab_id).cfg
    observed = cfg.observe()
    rows = services.context.get_alias_rows()
    executions: list[RunInputs] = []

    def receive_run(
        _self: RunService, inputs: RunInputs, *, plots: Plots, origin: CommandOrigin
    ) -> int:
        executions.append(inputs)
        return 99

    monkeypatch.setattr(RunService, "start_run", receive_run)
    with pytest.raises(RoleResolutionError):
        services.run_analyze_control.start_run(
            tab_id, observed.ref, aliases={"qubit": "missing"}
        )
    assert executions == []
    assert cfg.observe() == observed
    assert services.context.get_alias_rows() == rows


@pytest.mark.parametrize("fail_submit", [False, True])
def test_run_registration_and_failed_submission_allow_operation_reads(
    application: Fixture, qapp, monkeypatch: pytest.MonkeyPatch, fail_submit: bool
) -> None:
    if fail_submit:

        def reject_submission(*_args, **_kwargs) -> None:
            raise RuntimeError("submission rejected")

        monkeypatch.setattr(BackgroundRunner, "submit", reject_submission)
    tab_id = application.ctrl.new_tab("fake")
    ref = application.ctrl.cfg_resources.lookup(tab_id).observe().ref
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
                application.ctrl.start_run(tab_id, ref)
            assert application.ctrl.get_running_tab_id() is None
            assert application.ctrl.operation_control.active_operations() == ()
        else:
            token = application.ctrl.start_run(tab_id, ref)
            assert tuple(
                op.op for op in application.ctrl.operation_control.active_operations()
            ) == (token,)
    finally:
        subscription.unsubscribe()
        application.ctrl.cancel_run()
        deadline = monotonic() + 3.0
        while (
            application.ctrl.get_running_tab_id() is not None and monotonic() < deadline
        ):
            qapp.processEvents()
            sleep(0.001)
        assert application.ctrl.get_running_tab_id() is None
    assert errors == []
    assert observed == [(True, (), ())] * (2 if fail_submit else 1)
