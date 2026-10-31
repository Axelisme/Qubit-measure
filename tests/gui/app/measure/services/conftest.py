"""Native role-bound sources for result/analysis service contracts."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.measure.services.pane_commit import commit_run
from zcu_tools.gui.app.measure.state import RunPaneState
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameter_capture import capture_parameters
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import (
    ComponentSchema,
    RoleSpec,
    component_registry,
    role_registry,
)
from zcu_tools.resources.entry.schema import PointDocument

from tests.gui.app.measure._native_save_support import (
    NativeSaveSession,
    native_save_session,
)
from tests.gui.app.measure._record_support import require_record, require_target


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
                    "format_version": "2.0",
                    "schema_version": 1,
                    "general": {"created_at": "2026-10-08T00:00:00Z"},
                    "components": {
                        "Q1": {"kind": "run-source/qubit", "rate": 1.0},
                        "Q2": {"kind": "run-source/qubit", "rate": 2.0},
                    },
                }
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
            ),
            target=SavedPoint(entry_name="entry", point="working"),
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


@pytest.fixture
def role_load_session(
    tmp_path: Path,
    qapp: QApplication,
    result_sources: tuple[ParameterContext, ParameterContext],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[NativeSaveSession]:
    """Save Q1 roles at p, then update persisted Q1.rate before native Load."""
    del result_sources  # Fixture owns the component-kind registration lifecycle.
    with native_save_session(tmp_path, qapp) as session:
        control = session.context
        tab = session.state.get_tab("tab")
        control.add_component("Q1", "run-source/qubit", {"rate": 1.0})
        control.add_component("Q2", "run-source/qubit", {"rate": 2.0})
        session.context.apply_alias_rows(
            (
                AliasRow("source_qubit", "Q1"),
                AliasRow("custom", "Q2"),
            )
        )
        captured = session.context.read_context()
        assert captured is not None and captured.roles is not None
        session.metadata = session.metadata.model_copy(
            update={
                "snapshot": replace(
                    session.metadata.snapshot, roles=captured.roles.names
                ),
                "parameter_snapshot": capture_parameters(captured),
            }
        )
        commit_run(
            session.state,
            "tab",
            RunPaneState(
                result=replace(
                    require_record(tab.run),
                    metadata=session.metadata.model_dump(mode="json"),
                ),
                storage=captured.storage,
                parameters=captured,
            ),
        )
        session.save()
        control.set_parameter(require_target(captured), "Q1.rate", 7.0)
        yield session
