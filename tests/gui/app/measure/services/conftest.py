"""Native role-bound sources for result/analysis service contracts."""

from collections.abc import Iterator

import pytest
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import (
    ComponentSchema,
    RoleSpec,
    component_registry,
    role_registry,
)
from zcu_tools.resources.entry.schema import PointDocument


class SourceQubit(ComponentSchema):
    """Memory-only qubit source; rate is a numeric analysis input."""

    rate: float


@pytest.fixture
def result_sources() -> Iterator[tuple[ParameterContext, ParameterContext]]:
    """Yield Q1/Q2 bindings on one fixed point, restoring kind registration."""
    before = component_registry.kinds()
    component_registry.register("run-source/qubit", SourceQubit)
    role_registry.register("source_qubit", RoleSpec("run-source/qubit"))
    try:
        source = ParameterContext(
            point=PointDocument.model_validate(
                {
                    "format": "zcu.parameter-container",
                    "format_version": "1.0",
                    "general": {"created_at": "2026-10-08T00:00:00Z"},
                    "components": {
                        "Q1": {"kind": "run-source/qubit", "rate": 1.0},
                        "Q2": {"kind": "run-source/qubit", "rate": 2.0},
                    },
                }
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
            ),
            target=SavedPoint(entry_id="entry", point="working"),
            roles=None,
        )
        declarations = {"source_qubit": RoleSpec("run-source/qubit")}
        yield (
            source.bind_roles(declarations, focus="Q1", explicit={}),
            source.bind_roles(declarations, focus="Q2", explicit={}),
        )
    finally:
        role_registry.unregister("source_qubit")
        component_registry.unregister("run-source/qubit")
        assert component_registry.kinds() == before
