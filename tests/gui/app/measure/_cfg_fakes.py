"""Fixed parameter captures for application tests."""

from collections.abc import Mapping

from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.registry import component_registry
from zcu_tools.resources.entry.schema import ComponentSchema, PointDocument


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
                "format_version": "2.0",
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
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        )
    return ParameterContext(
        point=point,
        library=library,
        target=SavedPoint(entry_id="entry", point="working"),
        roles=None,
    )
