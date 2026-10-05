"""Production cfg owner wiring without tab widgets or live hardware."""

from time import monotonic, sleep
from unittest.mock import MagicMock

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.app.measure.adapter import SessionEnv
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
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    GateChangedPayload,
    MdChangedPayload,
)
from zcu_tools.gui.session.state import DeviceState, DeviceStatus

from tests.gui.app.measure.remote._helpers import Fixture
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
        (CfgEdit(("gain",), EvalValue("$device.flux.value")),),
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
    application.state.session_env.md.update(x=1.0)
    first = application.ctrl.new_tab("cfg-contract")
    second = application.ctrl.new_tab("cfg-contract")
    editors = [
        application.ctrl.cfg_resources.lookup(tab_id) for tab_id in (first, second)
    ]
    for editor in editors:
        editor.edit(
            editor.observe().ref.revision, (CfgEdit(("gain",), EvalValue("x")),)
        )
    before = editors[0].observe().ref
    observed = []

    def callback(publication: CfgObservation) -> None:
        if publication.ref != before:
            observed.append(tuple(editor.observe() for editor in editors))

    unsubscribe = editors[0].watch(callback)
    application.state.session_env.md.update(x=9.0)
    application.state.version.bump("context")
    application.bus.emit(MdChangedPayload(md=application.state.session_env.md))
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
