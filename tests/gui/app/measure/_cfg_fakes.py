"""Real cfg resources for application tests that do not mount a frontend."""

from collections.abc import Mapping
from unittest.mock import MagicMock

from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.services.tab_cfg import TabCfgResources
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.cfg.model import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.cfg.resource import CfgResolution, CfgResource
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.types import ParameterContext, RoleBinding, SessionEnv
from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.registry import component_registry
from zcu_tools.resources.entry.schema import ComponentSchema, PointDocument


class PublishedHost:
    """Published option sources without live value or hardware reads."""

    def __init__(self, state: State) -> None:
        self.state = state

    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        """Read the published capture, optionally binding caller roles."""
        parameters = self.state.session_env.parameters
        if parameters is None:
            raise RuntimeError("Test context is unavailable")
        return ParameterContext(
            point=parameters.document(),
            library=parameters.library,
            target=parameters.target,
            roles=roles,
        )

    def list_device_names(self) -> list[str]:
        return [device.name for device in self.state.list_devices()]

    @property
    def arb_waveforms(self) -> "PublishedHost":
        return self

    def list_data_keys(self) -> list[str]:
        return []

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        raise RuntimeError("Test sources do not query live values")


def parameter_capture(
    components: Mapping[str, ComponentSchema] | None = None,
    library: LibrarySnapshot | None = None,
    /,
    **values: float,
) -> ParameterContext:
    """Capture original component models and optional general.ext numeric values.

    components maps native names such as R1/Q1 to existing ComponentSchema
    instances. Missing kinds are registered only while validating the point;
    pre-existing registrations retain their model identity and order on failure
    as well as success. No source files or consumer roles are created.
    """
    components = {} if components is None else components
    before = component_registry.kinds()
    added: list[str] = []
    try:
        for component in components.values():
            if component.kind not in component_registry.kinds():
                component_registry.register(component.kind, type(component))
                added.append(component.kind)
            if component_registry.get(component.kind) is not type(component):
                raise ValueError(
                    f"Different model already registered for {component.kind!r}"
                )
        point = PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-08T00:00:00Z", "ext": values},
                "components": {
                    name: component.model_dump(mode="python", exclude_unset=True)
                    for name, component in components.items()
                },
            }
        )
    finally:
        for kind in reversed(added):
            component_registry.unregister(kind)
        assert component_registry.kinds() == before, (
            "parameter_capture polluted component registry"
        )
    if library is None:
        library = LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        )
    return ParameterContext(
        point=point, library=library, target=EntryDraft(entry_id="entry"), roles=None
    )


def cfg_resolution(state: State, parameters: ParameterContext | None) -> CfgResolution:
    """Snapshot current options and provenance with the supplied fixed capture."""
    return MeasureCfgBindings(PublishedHost(state)).snapshot_from_state(
        state, captured_values={}, parameters=parameters
    )


def cfg_resources(state: State) -> TabCfgResources:
    return TabCfgResources(
        mutation_allowed=lambda tab_id: (
            not state.has_tab(tab_id) or not state.is_tab_busy(tab_id)
        ),
    )


def make_cfg(schema: CfgSchema, *, state: State | None = None) -> CfgResource:
    """Create a resource retaining the state's creation capture across refresh."""
    sources = (
        State(SessionEnv(parameters=None, soc=None, soccfg=None))
        if state is None
        else state
    )
    parameters = sources.session_env.parameters
    return CfgResource(
        lambda: schema,
        resolution=lambda: cfg_resolution(sources, parameters),
    )


def configure_cfg_lookup(ctrl: MagicMock) -> None:
    editors: dict[str, CfgResource] = {}

    def lookup(tab_id: str) -> CfgResource:
        if tab_id not in editors:
            editors[tab_id] = make_cfg(CfgSchema(CfgSectionSpec(), CfgSectionValue()))
        return editors[tab_id]

    ctrl.cfg_resources.lookup.side_effect = lookup
