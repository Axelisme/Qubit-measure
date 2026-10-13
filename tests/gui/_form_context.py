"""Memory-only parameter captures for form core and editor tests."""

from pydantic import JsonValue
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


def captured_ext_env(**ext: JsonValue) -> FormEnv:
    """Detach extension values in a fixed point capture with no roles or I/O."""
    return FormEnv(
        parameters=ParameterContext(
            point=PointDocument(
                format="zcu.parameter-container",
                format_version="2.0",
                general=PointGeneral(created_at="2026-10-08T00:00:00Z", ext=ext),
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(
                    format="zcu.module-library",
                    format_version="1.0",
                )
            ),
            target=SavedPoint(entry_id="entry", point="synthetic"),
            roles=None,
            storage=None,
        )
    )
