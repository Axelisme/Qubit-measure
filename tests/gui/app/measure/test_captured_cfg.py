"""Fixed captures at measure binding and cross-tab publication seams."""

from collections.abc import Generator, Mapping
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, ContextReadiness
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.services.tab_cfg import TabCfgResources
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.types import ParameterContext, RoleBinding, SessionEnv
from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import ComponentSchema, RoleSpec, component_registry
from zcu_tools.resources.entry.schema import PointDocument


class CaptureDriver(ComponentSchema):
    rate: float


@pytest.fixture
def captures() -> Generator[tuple[ParameterContext, ParameterContext]]:
    before = component_registry.kinds()
    component_registry.register("capture/driver", CaptureDriver)
    try:
        point = PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-08T00:00:00Z"},
                "components": {
                    "Q1": {"kind": "capture/driver", "rate": 10.0},
                    "Q2": {"kind": "capture/driver", "rate": 20.0},
                },
            }
        )
        library = LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        )
        values = tuple(
            ParameterContext(
                point=point,
                library=library,
                target=EntryDraft(entry_id="entry"),
                roles=RoleBinding(
                    name, {"qubit": RoleSpec("capture/driver")}, {"qubit": name}
                ),
            )
            for name in ("Q1", "Q2")
        )
        yield values[0], values[1]
    finally:
        component_registry.unregister("capture/driver")
        assert component_registry.kinds() == before


class PublishedHost:
    def __init__(self, current: ParameterContext) -> None:
        self.current = current
        self.arb_waveforms = self

    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        return ParameterContext(
            point=self.current.document(),
            library=self.current.library,
            target=self.current.target,
            roles=roles,
        )

    def list_device_names(self) -> list[str]:
        return []

    def list_data_keys(self) -> list[str]:
        return []

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        raise AssertionError("Fixed cfg bindings must not query live value providers")


def expression_schema() -> CfgSchema:
    return CfgSchema(
        CfgSectionSpec(fields={"value": ScalarSpec("Value", float)}),
        CfgSectionValue(fields={"value": EvalValue("qubit.rate")}),
    )


def test_binding_snapshot_keeps_supplied_capture_after_live_selection_changes(
    captures: tuple[ParameterContext, ParameterContext],
) -> None:
    first, second = captures
    host = PublishedHost(first)
    bindings = MeasureCfgBindings(host)
    source = bindings.snapshot((), captured_values={}, parameters=first)
    host.current = second
    assert source.evaluate_expression("qubit.rate") == 10.0
    draft = bindings.new_draft(expression_schema(), parameters=first)
    assert draft.is_valid()
    assert (
        bindings.snapshot(
            (), captured_values={}, parameters=second
        ).evaluate_expression("qubit.rate")
        == 20.0
    )


def test_tab_refresh_preserves_distinct_captures_and_publishes_before_notification(
    captures: tuple[ParameterContext, ParameterContext],
) -> None:
    first, second = captures
    host = PublishedHost(first)
    bindings = MeasureCfgBindings(host)
    owner = TabCfgResources(mutation_allowed=lambda _tab: True)
    one = owner.create(
        "one",
        expression_schema,
        resolution=lambda: bindings.snapshot((), captured_values={}, parameters=first),
    )
    two = owner.create(
        "two",
        expression_schema,
        resolution=lambda: bindings.snapshot((), captured_values={}, parameters=second),
    )
    initial = one.observe().ref
    observed = []

    def subscriber(publication) -> None:
        if publication.ref != initial:
            observed.append((one.observe(), two.observe()))

    dispose = one.watch(subscriber)
    host.current = second
    publications = owner.refresh_all()
    dispose()
    assert observed == [publications]
    assert one.accept(publications[0].ref.revision).values == {"value": 10.0}
    assert two.accept(publications[1].ref.revision).values == {"value": 20.0}


def test_tab_creation_and_guard_keep_the_capture_after_focus_changes(
    captures: tuple[ParameterContext, ParameterContext],
) -> None:
    first, second = captures
    host = PublishedHost(first)
    bindings = MeasureCfgBindings(host)
    state = State(
        SessionEnv(
            parameters=first,
            soc=None,
            soccfg=None,
            readiness=ContextReadiness.ACTIVE,
        )
    )
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(requires_soc=False)
    adapter.parameter_roles.return_value = {"qubit": RoleSpec("capture/driver")}
    adapter.make_default_cfg.return_value = expression_schema()
    registry = MagicMock(spec=Registry)
    registry.create.return_value = adapter
    resources = TabCfgResources(mutation_allowed=lambda _tab: True)
    service = TabService(
        state,
        registry,
        MagicMock(),
        resources,
        cfg_resolution=lambda parameters: bindings.snapshot(
            (), captured_values={}, parameters=parameters
        ),
        captured_parameters=lambda _roles: host.current,
    )
    one = service.new_tab("capture")
    host.current = second
    two = service.new_tab("capture")
    assert state.get_tab(one).parameters is first
    assert state.get_tab(two).parameters is second
    assert [
        call.args[0].parameters for call in adapter.make_default_cfg.call_args_list
    ] == [first, second]
    resources.refresh_all()
    permit = GuardService(state).acquire_run_permit(
        one, expected_revision=state.get_tab(one).cfg.observe().ref.revision
    )
    assert permit.accepted_cfg.values == {"value": 10.0}
    assert permit.request.cfg_context is not None
    assert permit.request.cfg_context.modules is first.library
    assert state.get_tab(two).cfg.accept(
        state.get_tab(two).cfg.observe().ref.revision
    ).values == {"value": 20.0}


def test_role_capture_failure_does_not_register_a_tab() -> None:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    registry = MagicMock(spec=Registry)
    registry.create.return_value.parameter_roles.return_value = {
        "qubit": RoleSpec("capture/driver")
    }

    def unavailable(_roles: Mapping[str, RoleSpec]) -> ParameterContext | None:
        raise ValueError("role target is unavailable")

    service = TabService(
        state,
        registry,
        MagicMock(),
        TabCfgResources(mutation_allowed=lambda _tab: True),
        cfg_resolution=MagicMock(),
        captured_parameters=unavailable,
    )
    with pytest.raises(ValueError, match="role target is unavailable"):
        service.new_tab("capture")
    assert state.list_tab_ids() == []
    registry.create.return_value.make_default_cfg.assert_not_called()
