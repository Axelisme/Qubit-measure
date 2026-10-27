"""Lab-owned projection of component flux calibration for analysis apps."""

from zcu_tools.analysis.flux import ComponentFluxContext, DevicePeriod, FluxFrame
from zcu_tools.resources.entry.schema import ComponentSchema

from zcu_lab.components.qubit import QubitBaseSchema


def component_flux_context(component: ComponentSchema) -> ComponentFluxContext:
    """Project a lab qubit's offset, signed periods and default instrument.

    Component periods use native instrument units; the schema stores no SI unit
    declaration. Null offset remains frame=None. Non-qubit kinds raise ValueError
    naming their kind. No driver, cfg or current entry state is read.
    """
    if not isinstance(component, QubitBaseSchema):
        raise ValueError(f"Unsupported component kind {component.kind!r}")
    units = {name: "native" for name in component.flux.period}
    frame = None
    if component.flux.offset is not None:
        frame = FluxFrame(
            component.flux.offset,
            {
                name: DevicePeriod(period)
                for name, period in component.flux.period.items()
            },
        )
    return ComponentFluxContext(frame, component.flux_dev, units)
