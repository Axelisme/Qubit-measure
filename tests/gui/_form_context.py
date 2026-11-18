"""Memory-only parameter captures for form core and editor tests."""

from pydantic import JsonValue
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.session.context_types import DocumentSetRef
from zcu_tools.gui.session.parameters import DocumentSetCapture
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import ParameterDocument, ParameterGeneral


def captured_ext_env(**ext: JsonValue) -> FormEnv:
    """Detach extension values in a fixed point capture with no roles or I/O."""
    return FormEnv(
        parameters=DocumentSetCapture(
            point=ParameterDocument(
                format="zcu.parameter-container",
                format_version="2.0",
                schema_version=1,
                general=ParameterGeneral(created_at="2026-10-08T00:00:00Z", ext=ext),
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(
                    format="zcu.module-library",
                    format_version="2.0",
                )
            ),
            target=DocumentSetRef(entry_name="entry", point="synthetic"),
            roles=None,
            storage=None,
        )
    )
