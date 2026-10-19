"""Headless form resources fix sources until explicit Reset and guard lifetime."""

from collections.abc import Generator, Mapping
from inspect import signature

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase, Expr, Hidden, PulseSpec
from zcu_tools.gui.app.measure.services import tab_form as resource_api
from zcu_tools.gui.app.measure.services.form_contracts import (
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormInputRejected
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


def test_elements_aggregate_keeps_partial_values_and_errors_without_recapture():
    from tests.gui.form_adapter._indexed_models import indexed_env, indexed_form

    captures = []

    def capture(_aliases):
        captures.append("capture")
        return resource_api.TabFormSources(indexed_env())

    resource = resource_api.TabFormResource(
        lambda _ctx: indexed_form(),
        capture=capture,
        mutation_allowed=lambda: True,
    )
    assert hasattr(resource, "read_elements"), "Missing single owner aggregate read"
    events = []
    resource.subscribe(lambda: events.append("changed"))
    try:
        resource.write(
            {("tones", 0, "phase"): {"expr": "1 / 0"}, ("participants",): ["B"]}
        )
        resource.select(("tones", 1, "sweep"), {"kind": "center"})
        before, base = resource.input, resource.editor_state.base
        observed = resource.read_elements(("tones",))
        assert [(item.index, item.name, item.path) for item in observed.elements] == [
            (0, "A", ("tones", 0)),
            (1, "B", ("tones", 1)),
        ]
        first, second = observed.elements
        assert first.input["phase"] == {"expr": "1 / 0"}
        assert (
            first.errors is not None and ("tones", 0, "phase") in first.errors.messages
        )
        assert first.values[("tones", 0, "sweep", "start")] == 2
        assert second.values[("tones", 1, "sweep", "start")] == 10
        assert second.errors is None
        assert hasattr(second, "selections"), (
            "Aggregate must retain authored selection overrides"
        )
        assert second.selections == {("tones", 1, "sweep"): {"kind": "center"}}
        assert second.input["sweep"] == {
            "kind": "linrange",
            "start": 10,
            "stop": 14,
            "n": 3,
        }
        first.input["name"] = "detached"
        assert resource.input.same_as(before)
        assert resource.editor_state.base.same_as(base)
        assert captures == ["capture"] and events == ["changed", "changed"]
        with pytest.raises(CfgPreconditionError, match="not valid"):
            resource.accept()
    finally:
        resource.revoke()
    with pytest.raises(CfgPreconditionError):
        resource.read_elements(("tones",))


def test_indexed_restore_uses_saved_positions_selections_and_reports_dropped():
    from tests.gui.form_adapter._indexed_models import (
        element,
        indexed_env,
        indexed_form,
    )

    resource = resource_api.TabFormResource(
        lambda _ctx: indexed_form(),
        capture=lambda _aliases: resource_api.TabFormSources(indexed_env()),
        mutation_allowed=lambda: True,
    )
    try:
        resource.run_command(("tones", 1, "drive"), "to_inline")
        resource.write({("tones", 1, "drive", "module", "gain"): {"expr": "2 / 5"}})
        saved = resource.input
        raw = saved.data
        retained = element(raw, 1)
        retained.pop("phase")
        retained["old_gain"] = 99
        raw["tones"] = [retained]
        raw["participants"] = ["B"]
        saved = FormInput.from_json(
            saved.model,
            FormInput(
                saved.model,
                raw,
                selections={("tones", 0, "sweep"): {"kind": "center"}},
            ).to_json(),
        )
        dropped = resource.restore_input(saved)
        assert ("tones", 0, "old_gain") in dropped
        restored = resource.input.data["tones"]
        assert isinstance(restored, list) and len(restored) == 1
        assert element(resource.input.data, 0)["phase"] == 0
        assert resource.input.selections[("tones", 0, "sweep")] == {"kind": "center"}
        sweep = element(resource.input.data, 0)["sweep"]
        assert isinstance(sweep, dict) and sweep["kind"] == "linrange"
        assert resource.read_module(("tones", 0, "drive")).value["gain"] == 0.4
        before = resource.input
        with pytest.raises(ValueError, match="identity"):
            resource.write({("tones", 0, "name"): "A"})
        assert resource.input.same_as(before)
        resource.reset()
        reset = resource.input.data["tones"]
        assert isinstance(reset, list) and len(reset) == 2
    finally:
        resource.revoke()


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


def test_refresh_preserves_authored_expression_and_direct_defaults() -> None:
    latest = 2.0
    captures: list[Mapping[str, str] | None] = []

    def capture(aliases: Mapping[str, str] | None) -> resource_api.TabFormSources:
        captures.append(aliases)
        return resource_api.TabFormSources(captured_env(latest))

    resource = resource_api.TabFormResource(
        lambda _ctx: Form(ScalarCore, {"amount": Expr("general.ext.amount * 2")}),
        capture=capture,
        mutation_allowed=lambda: True,
    )
    assert hasattr(resource, "refresh_sources"), (
        "Tabs need an explicit capture-only command that preserves authored input"
    )
    published: list[object] = []
    resource.subscribe(lambda: published.append(resource.preview().values[("amount",)]))
    latest = 5.0
    resource.refresh_sources(aliases={"selected": "Q2"})
    assert resource.input.data == {"amount": 4.0}
    assert resource.result() == ScalarCore(amount=4.0)
    resource.write({("amount",): {"expr": "general.ext.amount * 3"}})
    assert resource.result() == ScalarCore(amount=15.0)
    before = resource.input.to_json()
    latest = 7.0
    resource.refresh_sources()
    assert resource.input.to_json() == before
    assert resource.result() == ScalarCore(amount=21.0)
    assert published == [4.0, 15.0, 21.0]
    assert captures == [None, {"selected": "Q2"}, None]
    resource.reset()
    assert resource.input.data == {"amount": 14.0}
    assert resource.result() == ScalarCore(amount=14.0)
    assert captures == [None, {"selected": "Q2"}, None, None]


def test_describe_keeps_fixed_capture_and_revokes_retained_handle() -> None:
    captures: list[Mapping[str, str] | None] = []
    current = 2.0

    def capture(aliases: Mapping[str, str] | None) -> resource_api.TabFormSources:
        captures.append(aliases)
        return resource_api.TabFormSources(captured_env(current))

    resource = resource_api.TabFormResource(
        lambda _ctx: Form(ScalarCore, {}),
        capture=capture,
        mutation_allowed=lambda: True,
    )
    assert hasattr(resource, "describe"), "Missing fixed-capture description read"
    resource.write({("amount",): {"expr": "general.ext.amount"}})
    current = 9.0
    before = resource.input.data
    notifications: list[str] = []
    resource.subscribe(lambda: notifications.append("changed"))
    description = resource.describe(("amount",))
    assert description.path == ("amount",)
    assert description.children == ()
    assert resource.preview().values[("amount",)] == 2.0
    assert resource.input.data == before
    assert captures == [None]
    assert notifications == []
    resource.revoke()
    with pytest.raises(CfgPreconditionError) as gone:
        resource.describe(("amount",))
    assert gone.value.reason is CfgPreconditionReason.RESOURCE_GONE


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
    dropped = resource.restore_input(
        FormInput(
            resource.input.model, {"amount": {"expr": "unfinished +"}, "removed": 9}
        )
    )
    assert dropped == (("removed",),)
    assert resource.input.data == {"amount": {"expr": "unfinished +"}}
    assert isinstance(resource.result(), FormErrors)
    assert (
        resource.restore_input(
            FormInput(resource.input.model, {"amount": "not a number"})
        )
        == ()
    )
    assert resource.input.data == {"amount": "not a number"}
    assert isinstance(resource.result(), FormErrors)


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
        lambda: second.restore_input(FormInput(second.input.model, {"amount": 9.0})),
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


def test_fixed_elements_open_edit_observe_and_run_admission() -> None:
    from pydantic import create_model

    from tests.gui.form_adapter._fixed_models import named_form, named_input

    model = named_form()
    core = create_model(
        "NamedCore",
        __base__=ConfigBase,
        tones=(
            model.model_fields["tones"].rebuild_annotation(),
            named_input(model).data["tones"],
        ),
        participants=(model.model_fields["participants"].rebuild_annotation(), ["B"]),
    )
    resource = resource_api.TabFormResource(
        lambda _ctx: Form(core, {}, fixed_elements=model.fixed_elements),
        capture=lambda _aliases: resource_api.TabFormSources(env=FormEnv()),
        mutation_allowed=lambda: True,
    )
    assert resource.accept().values["participants"] == ["B"]
    target = resource.describe(("tones", 1, "gain")).target
    assert target is not None
    assert target.path == ("tones", 1, "gain")
    resource.write({("tones", 0, "gain"): -2})
    result = resource.result()
    assert isinstance(result, FormErrors)
    assert ("tones", 0, "gain") in result.messages
    with pytest.raises(CfgPreconditionError) as caught:
        resource.accept()
    assert caught.value.reason is CfgPreconditionReason.NOT_VALID
    resource.write({("tones", 0, "gain"): 5, ("participants",): ["B", "A"]})
    assert resource.accept().values["participants"] == ["B", "A"]
    assert resource.preview().values[("tones", 0, "gain")] == 5
    before = resource.input
    with pytest.raises(FormInputRejected, match="Readonly"):
        resource.write({("tones",): []})
    assert resource.input.same_as(before)


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
