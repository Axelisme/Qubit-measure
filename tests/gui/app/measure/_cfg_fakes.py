"""Real cfg resources for application tests that do not mount a frontend."""

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
from zcu_tools.resources.entry.schema import PointDocument


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


def parameter_capture(**values: float) -> ParameterContext:
    """Build an entry draft capture with numeric values under general.ext."""
    point = PointDocument.model_validate(
        {
            "format": "zcu.parameter-container",
            "format_version": "1.0",
            "general": {"created_at": "2026-10-08T00:00:00Z", "ext": values},
            "components": {},
        }
    )
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
