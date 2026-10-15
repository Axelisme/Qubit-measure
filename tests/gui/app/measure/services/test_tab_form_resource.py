"""Headless form resources fix sources until explicit Reset and guard lifetime."""

from collections.abc import Generator, Mapping
from inspect import signature

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase, Expr, Hidden, PulseSpec
from zcu_tools.gui.app.measure.services import tab_form as resource_api
from zcu_tools.gui.cfg.resource import CfgPreconditionError, CfgPreconditionReason
from zcu_tools.gui.form import FormEnv, FormErrors
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry import component_registry, role_registry
from zcu_tools.resources.entry.schema import (
    ComponentSchema,
    PointDocument,
    PointGeneral,
)


class ScalarCore(ConfigBase):
    """A concrete experiment cfg accepts one positive measurement setting."""

    amount: float = 1.0


def captured_env(amount: float) -> FormEnv:
    """Construct a detached source with a known arithmetic value."""
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
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


class FactoryComponent(ComponentSchema):
    """A synthetic component provides a valid direct library-key namespace."""


@pytest.fixture
def factory_component() -> Generator[None]:
    models = {kind: component_registry.get(kind) for kind in component_registry.kinds()}
    roles = {name: role_registry.get(name) for name in role_registry.names()}
    shorthand, focus_kinds = role_registry.shorthand, role_registry.focus_kinds
    component_registry.register("factory/shape", FactoryComponent)
    try:
        yield
    finally:
        component_registry.unregister("factory/shape")
        assert {
            kind: component_registry.get(kind) for kind in component_registry.kinds()
        } == models
        assert {
            name: role_registry.get(name) for name in role_registry.names()
        } == roles
        assert (role_registry.shorthand, role_registry.focus_kinds) == (
            shorthand,
            focus_kinds,
        )


def factory_env(amount: float, *, library: LibrarySnapshot | None = None) -> FormEnv:
    """Capture a known component with no pulse entry in the library."""
    env = captured_env(amount)
    parameters = env.parameters
    assert parameters is not None
    point = PointDocument.model_validate(
        {
            "format": "zcu.parameter-container",
            "format_version": "2.0",
            "components": {"Q1": {"kind": "factory/shape"}},
            "general": {
                "created_at": "2026-10-14T00:00:00Z",
                "ext": {"amount": amount},
            },
        }
    )
    fixed_library = env.library if library is None else library
    assert fixed_library is not None
    return FormEnv(
        parameters=ParameterContext(
            point=point,
            library=fixed_library,
            target=parameters.target,
            roles=None,
        ),
        library=fixed_library,
    )


@pytest.mark.usefixtures("factory_component")
def test_open_and_reset_factory_replace_shape_and_values_from_current_capture() -> None:
    assert "make_form" in signature(resource_api.TabFormResource).parameters, (
        "Missing captured Open/Reset form factory"
    )
    latest = factory_env(2.0)
    calls: list[bool] = []

    def make_form(ctx: FormContext) -> Form[ScalarCore]:
        has_pulse = ctx.has("Q1.pulses.pi")
        calls.append(has_pulse)
        return Form(
            ScalarCore,
            {"amount": Hidden() if has_pulse else Expr("general.ext.amount")},
        )

    resource = resource_api.TabFormResource(
        make_form=make_form,
        capture=lambda _aliases: resource_api.TabFormSources(latest),
        mutation_allowed=lambda: True,
    )
    assert resource.input.data == {"amount": 2.0}
    resource.write({("amount",): 7.0})
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="2.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={
                        "pi": PulseSpec(
                            ch=0,
                            nqz=1,
                            freq=100.0,
                            gain=0.25,
                            waveform="Q1.waveforms.unresolved",
                        )
                    }
                )
            },
        )
    )
    parameters = factory_env(5.0).parameters
    assert parameters is not None
    latest = factory_env(5.0, library=library)
    assert resource.result() == ScalarCore(amount=7.0)
    assert calls == [False]
    published: list[dict[str, JsonValue]] = []
    resource.subscribe(lambda: published.append(resource.input.data))
    resource.reset()
    assert resource.input.data == {}
    assert resource.result() == ScalarCore(amount=1.0)
    assert published == [{}]
    assert calls == [False, True]
    latest = factory_env(9.0)
    resource.reset()
    assert resource.input.data == {"amount": 9.0}
    assert resource.result() == ScalarCore(amount=9.0)
    assert calls == [False, True, False]


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
        lambda _ctx: Form(ScalarCore, {"amount": Expr("general.ext.amount * 2")}),
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


def test_rejected_batch_preserves_input_without_publication() -> None:
    resource = resource_api.TabFormResource(
        lambda _ctx: Form(ScalarCore, {}),
        capture=lambda _aliases: resource_api.TabFormSources(captured_env(2.0)),
        mutation_allowed=lambda: True,
    )
    before = resource.input.data
    published: list[ConfigBase | FormErrors] = []
    resource.subscribe(lambda: published.append(resource.result()))

    with pytest.raises(ValueError, match="missing"):
        resource.write({("amount",): 4.0, ("missing",): 9.0})

    assert resource.input.data == before
    assert resource.result() == ScalarCore(amount=1.0)
    assert published == []


def test_compatible_restore_preserves_unfinished_input_and_reports_dropped_paths() -> (
    None
):
    resource = resource_api.TabFormResource(
        lambda _ctx: Form(ScalarCore, {}),
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
        lambda _ctx: Form(ScalarCore, {}),
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

    def capture(_aliases: Mapping[str, str] | None) -> resource_api.TabFormSources:
        return resource_api.TabFormSources(captured_env(2.0))

    first = pool.create("first", lambda _ctx: form, capture=capture)
    second = pool.create("second", lambda _ctx: form, capture=capture)
    failures: list[CfgPreconditionReason] = []

    def notification() -> None:
        for command in (
            lambda: second.write({("amount",): 9.0}),
            second.reset,
            second.accept,
            lambda: pool.retire("second"),
            lambda: pool.create("third", lambda _ctx: form, capture=capture),
        ):
            try:
                command()
            except CfgPreconditionError as exc:
                failures.append(exc.reason)

    first.subscribe(notification)
    first.write({("amount",): 4.0})
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION] * 5
    assert second.result() == ScalarCore(amount=1.0)
    allowed = False
    for command in (
        lambda: second.write({("amount",): 9.0}),
        lambda: second.write_compatible({"amount": 9.0}),
        second.reset,
    ):
        with pytest.raises(CfgPreconditionError) as caught:
            command()
        assert caught.value.reason is CfgPreconditionReason.MUTATION_BLOCKED
    assert second.result() == ScalarCore(amount=1.0)
    allowed = True
    pool.retire("second")
    with pytest.raises(CfgPreconditionError) as caught:
        second.write({("amount",): 9.0})
    assert caught.value.reason is CfgPreconditionReason.RESOURCE_GONE
    assert pool.create(
        "second", lambda _ctx: form, capture=capture
    ).result() == ScalarCore(amount=1.0)


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
        "tab",
        lambda _ctx: Form(ScalarCore, {"amount": Expr("general.ext.amount")}),
        capture=capture,
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


def test_editor_state_borrows_the_same_input_for_reading_and_subscribing() -> None:
    resource = resource_api.TabFormResource(
        lambda _ctx: Form(ScalarCore, {}),
        capture=lambda _aliases: resource_api.TabFormSources(captured_env(2.0)),
        mutation_allowed=lambda: True,
    )
    state = resource.editor_state
    assert state is resource.editor_state
    published: list[JsonValue] = []
    state.subscribe(lambda: published.append(state.input.data["amount"]))

    resource.write({("amount",): 3.0})

    assert published == [3.0]
    assert state.input.data == resource.input.data == {"amount": 3.0}
    assert resource.result() == ScalarCore(amount=3.0)

    resource.revoke()
    with pytest.raises(CfgPreconditionError) as caught:
        _ = resource.editor_state
    assert caught.value.reason is CfgPreconditionReason.RESOURCE_GONE


def test_duplicate_create_does_not_replace_existing_resource() -> None:
    """A duplicate public tab identity cannot replace its edited Form resource."""
    pool = resource_api.TabFormResources(mutation_allowed=lambda _tab: True)
    first = pool.create(
        "tab",
        lambda _ctx: Form(ScalarCore, {}),
        capture=lambda _aliases: resource_api.TabFormSources(captured_env(2.0)),
    )
    first.write({("amount",): 4.0})
    with pytest.raises(ValueError, match="already registered"):
        pool.create(
            "tab",
            lambda _ctx: Form(ScalarCore, {"amount": 9.0}),
            capture=lambda _aliases: resource_api.TabFormSources(captured_env(3.0)),
        )
    assert pool.lookup("tab") is first
    assert first.result() == ScalarCore(amount=4.0)


def test_failed_creation_does_not_reserve_tab_identity() -> None:
    """A source failure before publication leaves the identity available."""
    pool = resource_api.TabFormResources(mutation_allowed=lambda _tab: True)

    def unavailable(_aliases: Mapping[str, str] | None) -> resource_api.TabFormSources:
        raise ValueError("capture unavailable")

    with pytest.raises(ValueError, match="capture unavailable"):
        pool.create("tab", lambda _ctx: Form(ScalarCore, {}), capture=unavailable)
    resource = pool.create(
        "tab",
        lambda _ctx: Form(ScalarCore, {}),
        capture=lambda _aliases: resource_api.TabFormSources(captured_env(2.0)),
    )
    assert pool.lookup("tab") is resource
    assert resource.result() == ScalarCore(amount=1.0)
