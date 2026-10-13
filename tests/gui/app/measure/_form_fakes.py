"""Fixed native form sources for headless measure service contracts."""

from __future__ import annotations

from collections.abc import Mapping

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.services import tab_form as resource_api
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.session.aliases import AliasTable


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
        source_basis=MeasureCfgBindings.source_basis(state),
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
        form,
        capture=lambda aliases: form_sources(state, aliases),
        mutation_allowed=lambda: (
            not any(
                state.get_tab(tab_id).cfg is resource and state.is_tab_busy(tab_id)
                for tab_id in state.list_tab_ids()
            )
        ),
    )
    return resource
