"""Real cfg resources for application tests that do not mount a frontend."""

from unittest.mock import MagicMock

from zcu_tools.gui.app.measure.adapter.lowering import make_sweep_range
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.services.tab_cfg import TabCfgResources
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.cfg.model import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.cfg.resource import CfgResource
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo
from zcu_tools.resources.context import MetaDict, ModuleLibrary


class PublishedHost:
    def __init__(self, state: State) -> None:
        self.state = state
        self._empty_md = MetaDict(None)
        self._empty_ml = ModuleLibrary(None)

    def get_current_md(self) -> MetaDict:
        return self.state.session_env.md or self._empty_md

    def get_current_ml(self) -> ModuleLibrary:
        return self.state.session_env.ml or self._empty_ml

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


def cfg_resources(state: State) -> TabCfgResources:
    bindings = MeasureCfgBindings(PublishedHost(state))
    return TabCfgResources(
        resolution=lambda: bindings.snapshot_from_state(state, captured_values={}),
        make_range=make_sweep_range,
        mutation_allowed=lambda tab_id: (
            not state.has_tab(tab_id) or not state.is_tab_busy(tab_id)
        ),
    )


def make_cfg(schema: CfgSchema, *, state: State | None = None) -> CfgResource:
    sources = (
        State(
            SessionEnv(md=MetaDict(None), ml=ModuleLibrary(None), soc=None, soccfg=None)
        )
        if state is None
        else state
    )
    bindings = MeasureCfgBindings(PublishedHost(sources))
    return CfgResource(
        lambda: schema,
        resolution=lambda: bindings.snapshot_from_state(sources, captured_values={}),
        make_range=make_sweep_range,
    )


def configure_cfg_lookup(ctrl: MagicMock) -> None:
    editors: dict[str, CfgResource] = {}

    def lookup(tab_id: str) -> CfgResource:
        if tab_id not in editors:
            editors[tab_id] = make_cfg(CfgSchema(CfgSectionSpec(), CfgSectionValue()))
        return editors[tab_id]

    ctrl.cfg_resources.lookup.side_effect = lookup
