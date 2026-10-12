"""Headless form resources fix sources until explicit Reset and guard lifetime."""

from collections.abc import Mapping

import pytest
from zcu_tools.cfg_model import ConfigBase, Expr
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.app.measure.services import tab_cfg as resource_api
from zcu_tools.gui.cfg.resource import CfgPreconditionError, CfgPreconditionReason
from zcu_tools.gui.form import FormEnv, FormErrors
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class ScalarCore(ConfigBase):
    """A concrete experiment cfg accepts one positive measurement setting."""

    amount: float = 1.0


def captured_env(amount: float) -> FormEnv:
    """Construct a detached source with a known arithmetic value."""
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    )
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="1.0",
            general=PointGeneral(
                created_at="2026-10-13T00:00:00Z", ext={"amount": amount}
            ),
        ),
        library=library,
        target=SavedPoint(entry_id="synthetic", point="working"),
        roles=None,
    )
    return FormEnv(parameters=parameters, library=library)


def test_resource_edit_preview_run_use_capture_until_reset() -> None:
    assert hasattr(resource_api, "TabFormResource"), (
        "Missing tab-owned FormState resource"
    )
    assert hasattr(resource_api, "TabFormSources"), "Missing fixed form sources"
    latest = 2.0
    captures: list[Mapping[str, str] | None] = []

    def capture(aliases: Mapping[str, str] | None) -> resource_api.TabFormSources:
        captures.append(aliases)
        return resource_api.TabFormSources(env=captured_env(latest))

    resource = resource_api.TabFormResource(
        Form(ScalarCore, {"amount": Expr("general.ext.amount * 2")}),
        capture=capture,
        mutation_allowed=lambda: True,
    )
    assert resource.result() == ScalarCore(amount=4.0)
    latest = 5.0
    resource.write({("amount",): {"expr": "general.ext.amount * 3"}})
    assert resource.preview().values[("amount",)] == 6.0
    assert resource.result() == ScalarCore(amount=6.0)
    assert resource.accept().values == {"amount": 6.0}
    assert captures == [None]
    resource.reset(aliases={"selected": "Q2"})
    assert resource.result() == ScalarCore(amount=10.0)
    assert captures == [None, {"selected": "Q2"}]
    resource.write({("amount",): {"expr": "unfinished +"}})
    assert isinstance(resource.result(), FormErrors)
    with pytest.raises(CfgPreconditionError) as caught:
        resource.accept()
    assert caught.value.reason is CfgPreconditionReason.NOT_VALID
    assert resource.input.data["amount"] == {"expr": "unfinished +"}


def test_capture_tokens_are_fixed_before_atomic_input_write() -> None:
    latest = 2.0

    def capture(_aliases: Mapping[str, str] | None) -> resource_api.TabFormSources:
        return resource_api.TabFormSources(
            env=captured_env(latest), captures={"device.flux.value": latest}
        )

    resource = resource_api.TabFormResource(
        Form(ScalarCore, {}), capture=capture, mutation_allowed=lambda: True
    )
    latest = 5.0
    resource.write({("amount",): {"expr": "2 * @{device.flux.value}"}})
    assert resource.result() == ScalarCore(amount=4.0)
    assert resource.input.data["amount"] == {"expr": "2 * (2.0)"}
    before = resource.input.data
    with pytest.raises(CfgPreconditionError) as caught:
        resource.write({("amount",): {"expr": "@{missing.value}"}})
    assert caught.value.reason is CfgPreconditionReason.CAPTURE_UNAVAILABLE
    assert resource.input.data == before
    resource.reset()
    resource.write({("amount",): {"expr": "2 * @{device.flux.value}"}})
    assert resource.result() == ScalarCore(amount=10.0)


def test_compatible_restore_preserves_unfinished_input_and_reports_dropped_paths() -> (
    None
):
    resource = resource_api.TabFormResource(
        Form(ScalarCore, {}),
        capture=lambda _aliases: resource_api.TabFormSources(captured_env(2.0)),
        mutation_allowed=lambda: True,
    )
    assert hasattr(resource, "write_compatible"), (
        "Missing public compatible input restoration"
    )
    dropped = resource.write_compatible(
        {"amount": {"expr": "unfinished +"}, "removed": 9}
    )
    assert dropped == (("removed",),)
    assert resource.input.data == {"amount": {"expr": "unfinished +"}}
    assert isinstance(resource.result(), FormErrors)
    assert resource.write_compatible({"amount": "not a number"}) == (("amount",),)
    assert resource.input.data == {"amount": {"expr": "unfinished +"}}


def test_subscriber_failure_keeps_committed_result_and_notifies_other_subscribers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resource = resource_api.TabFormResource(
        Form(ScalarCore, {}),
        capture=lambda _aliases: resource_api.TabFormSources(captured_env(2.0)),
        mutation_allowed=lambda: True,
    )
    observed: list[ConfigBase | FormErrors] = []

    def failing_subscriber() -> None:
        raise RuntimeError("observer failed")

    resource.subscribe(failing_subscriber)
    resource.subscribe(lambda: observed.append(resource.result()))
    try:
        resource.write({("amount",): 4.0})
    except RuntimeError:
        pytest.fail("A subscriber must not reject a committed write")
    assert resource.input.data == {"amount": 4.0}
    assert observed == [ScalarCore(amount=4.0)]
    resource.write({("amount",): 5.0})
    assert resource.result() == ScalarCore(amount=5.0)
    assert observed == [ScalarCore(amount=4.0), ScalarCore(amount=5.0)]
    faults = [record for record in caplog.records if record.exc_info is not None]
    assert len(faults) == 2
    assert all(
        record.exc_info and isinstance(record.exc_info[1], RuntimeError)
        for record in faults
    )


def test_pool_guards_peer_notification_busy_commands_and_revoked_handles() -> None:
    assert hasattr(resource_api, "TabFormResources"), (
        "Missing state-owned Form resource pool"
    )
    allowed = True
    pool = resource_api.TabFormResources(mutation_allowed=lambda _tab: allowed)
    form = Form(ScalarCore, {})
    capture = lambda _aliases: resource_api.TabFormSources(captured_env(2.0))
    first = pool.create("first", form, capture=capture)
    second = pool.create("second", form, capture=capture)
    failures: list[CfgPreconditionReason] = []

    def notification() -> None:
        for command in (
            lambda: second.write({("amount",): 9.0}),
            second.reset,
            lambda: second.load_cfg(ScalarCore(amount=9.0)),
            second.accept,
            lambda: pool.retire("second"),
            lambda: pool.create("third", form, capture=capture),
        ):
            try:
                command()
            except CfgPreconditionError as exc:
                failures.append(exc.reason)

    first.subscribe(notification)
    first.write({("amount",): 4.0})
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION] * 6
    assert second.result() == ScalarCore(amount=0.0)
    allowed = False
    for command in (
        lambda: second.write({("amount",): 9.0}),
        lambda: second.write_compatible({"amount": 9.0}),
        second.reset,
        lambda: second.load_cfg(ScalarCore(amount=9.0)),
    ):
        with pytest.raises(CfgPreconditionError) as caught:
            command()
        assert caught.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
    assert second.result() == ScalarCore(amount=0.0)
    allowed = True
    pool.retire("second")
    with pytest.raises(CfgPreconditionError) as caught:
        second.write({("amount",): 9.0})
    assert caught.value.reason is CfgPreconditionReason.RESOURCE_GONE
    assert pool.create("second", form, capture=capture).result() == ScalarCore(
        amount=0.0
    )


def test_pool_reset_capture_failure_keeps_authored_input_and_old_sources() -> None:
    assert hasattr(resource_api, "TabFormResources"), (
        "Missing state-owned Form resource pool"
    )
    amount = 1.0
    fail_capture = False
    captures = 0

    def capture(_aliases: object) -> resource_api.TabFormSources:
        nonlocal captures
        captures += 1
        if fail_capture:
            raise RuntimeError("capture failed")
        return resource_api.TabFormSources(captured_env(amount))

    pool = resource_api.TabFormResources(mutation_allowed=lambda _tab: True)
    resource = pool.create(
        "tab", Form(ScalarCore, {"amount": Expr("general.ext.amount")}), capture=capture
    )
    resource.write({("amount",): 5.0})
    amount, fail_capture = 2.0, True
    with pytest.raises(RuntimeError, match="capture failed"):
        resource.reset()
    assert resource.result() == ScalarCore(amount=5.0)
    assert resource.input.data == {"amount": 5.0}
    parameters = resource.env.parameters
    assert parameters is not None
    assert parameters.value("general.ext.amount") == 1.0
    assert captures == 2
    fail_capture = False
    resource.reset()
    assert resource.result() == ScalarCore(amount=2.0)
    assert captures == 3
