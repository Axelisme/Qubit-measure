"""Form factories query captured module references without materializing them."""

from collections.abc import Generator
from pathlib import Path

import pytest
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.gui import form_adapter
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    SavedPoint,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry import component_registry, role_registry
from zcu_tools.resources.entry.schema import (
    ComponentSchema,
    PointDocument,
)


class SourceComponent(ComponentSchema):
    """A synthetic point owns named module slots."""

    module: dict[str, str | None]


@pytest.fixture(autouse=True)
def synthetic_component() -> Generator[None]:
    models = {kind: component_registry.get(kind) for kind in component_registry.kinds()}
    roles = {name: role_registry.get(name) for name in role_registry.names()}
    shorthand, focus_kinds = role_registry.shorthand, role_registry.focus_kinds
    component_registry.register("factory/source", SourceComponent)
    try:
        yield
    finally:
        component_registry.unregister("factory/source")
        assert {
            kind: component_registry.get(kind) for kind in component_registry.kinds()
        } == models
        assert {
            name: role_registry.get(name) for name in role_registry.names()
        } == roles
        assert (role_registry.shorthand, role_registry.focus_kinds) == (
            shorthand,
            focus_kinds,
        )


def parameters(*, faults: tuple[ContextSourceFault, ...] = ()) -> ParameterContext:
    """Capture a Spec with an unresolved child, so existence needs no expansion."""
    return ParameterContext(
        point=PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "2.0",
                "general": {"created_at": "2026-10-14T00:00:00Z"},
                "components": {
                    "Q1": {
                        "kind": "factory/source",
                        "module": {
                            "drive": "Q1.pulses.pi",
                            "empty": None,
                            "dangling": "Q1.pulses.missing",
                        },
                    }
                },
            }
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q1": ComponentLibrary(
                        pulses={
                            "pi": PulseSpec(
                                ch=0,
                                nqz=1,
                                freq=100.0,
                                gain=0.25,
                                waveform="Q1.waveforms.unresolved",
                            )
                        }
                    )
                },
            )
        ),
        target=SavedPoint(entry_id="synthetic", point="working"),
        roles=None,
        source_faults=faults,
    )


def test_factory_context_has_bound_slots_and_direct_roots_without_expansion() -> None:
    assert hasattr(form_adapter, "FormContext"), (
        "Missing captured factory reference query"
    )
    captured = parameters().bind_aliases(
        (AliasRow("qubit", "Q1"), AliasRow("empty", "")), {}
    )
    ctx = form_adapter.FormContext(captured)
    assert ctx.has("$qubit.module.drive")
    assert ctx.has("Q1.pulses.pi")
    assert not ctx.has("$qubit.module.empty")
    assert not ctx.has("$qubit.module.dangling")
    assert not ctx.has("Q1.pulses.missing")
    assert not ctx.has("$empty.module.drive")
    assert not ctx.has("$unknown.module.drive")
    assert not form_adapter.FormContext(None).has("$qubit.module.drive")


def test_factory_context_source_fault_is_not_absence() -> None:
    assert hasattr(form_adapter, "FormContext"), (
        "Missing captured factory reference query"
    )
    fault = ContextSourceFault(
        document="module_cfg",
        path=Path("module_cfg.yaml"),
        error_type="ValueError",
        message="broken source",
    )
    ctx = form_adapter.FormContext(parameters(faults=(fault,)))
    with pytest.raises(ContextSourceUnavailableError, match="broken source"):
        ctx.has("Q1.pulses.pi")
