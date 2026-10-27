"""Caller-injected component flux context shared by standalone analysis apps."""

from collections.abc import Callable

from zcu_tools.analysis.flux import ComponentFluxContext
from zcu_tools.resources.entry import snapshot
from zcu_tools.resources.entry.schema import ComponentSchema

from .entry import AnalysisTarget

ComponentFluxProvider = Callable[[ComponentSchema], ComponentFluxContext]


def component_flux_context(
    target: AnalysisTarget, provider: ComponentFluxProvider | None
) -> ComponentFluxContext:
    """Ask the injected lab provider for the selected component's flux context.

    Framework code never interprets component fields. Missing/rejecting providers
    raise ValueError identifying both component and kind. An uncalibrated offset
    remains frame=None, not a synthesized zero-offset calibration.
    """
    component = snapshot(target.resource.setup).components[target.component]
    identity = f"component {target.component!r} (kind {component.kind!r})"
    if provider is None:
        raise ValueError(f"No component flux provider configured for {identity}")
    try:
        return provider(component)
    except ValueError as error:
        raise ValueError(f"Flux provider rejected {identity}: {error}") from error
