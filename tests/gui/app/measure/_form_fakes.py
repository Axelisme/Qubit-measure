"""Fixed native form sources for headless measure service contracts."""

from __future__ import annotations

from collections.abc import Mapping
from unittest.mock import MagicMock

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.services import tab_form as resource_api
from zcu_tools.gui.app.measure.services.form_contracts import source_basis
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.aliases import AliasTable
from zcu_tools.gui.session.types import SessionEnv


def form_sources(
    state: State, aliases: Mapping[str, str] | None = None
) -> TabFormSources:
    """Fix current native parameters/library, without I/O or device choices."""
    parameters = state.session_env.parameters
    if parameters is not None:
        table = AliasTable(parameters.aliases)
        for alias, component in (aliases or {}).items():
            table = table.with_override(alias, component)
        parameters = parameters.bind_aliases(table.rows, {})
    return TabFormSources(
        FormEnv(
            parameters=parameters,
            library=None if parameters is None else parameters.library,
        ),
        source_basis=source_basis(state),
    )


def form_resources(state: State) -> resource_api.TabFormResources:
    """Create a lifetime owner enforcing actual State busy policy."""
    return resource_api.TabFormResources(
        mutation_allowed=lambda tab_id: (
            not state.has_tab(tab_id) or not state.is_tab_busy(tab_id)
        )
    )


def make_form[C: ConfigBase](form: Form[C], *, state: State) -> TabFormResource:
    """Prepare one form resource with owner-sequence fixed current sources."""
    resource = TabFormResource(
        lambda _ctx: form,
        capture=lambda aliases: form_sources(state, aliases),
        mutation_allowed=lambda: (
            not any(
                state.get_tab(tab_id).cfg is resource and state.is_tab_busy(tab_id)
                for tab_id in state.list_tab_ids()
            )
        ),
    )
    return resource


class LookupCfg(ConfigBase):
    """Three plain scalars a measure tab view can display and edit."""

    reps: int = 100
    rounds: int = 10
    gain: float = 0.1


def configure_form_lookup(ctrl: MagicMock) -> None:
    """Make ctrl.cfg_resources.lookup return one fixed LookupCfg resource per tab."""
    resources: dict[str, TabFormResource] = {}

    def lookup(tab_id: str) -> TabFormResource:
        if tab_id not in resources:
            state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
            resources[tab_id] = make_form(Form(LookupCfg, {}), state=state)
        return resources[tab_id]

    ctrl.cfg_resources.lookup.side_effect = lookup
