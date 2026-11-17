"""Native fixed parameter captures for shared expression UI contracts."""

from zcu_tools.format_version import YamlMap
from zcu_tools.gui.session.context_types import DocumentSetRef
from zcu_tools.gui.session.parameters import DocumentSetCapture
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import ParameterDocument


def make_parameters(values: YamlMap) -> DocumentSetCapture:
    """Capture numeric test values under general.ext in a memory-only entry draft."""
    return DocumentSetCapture(
        point=ParameterDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "2.0",
                "schema_version": 1,
                "general": {"created_at": "2026-10-08T00:00:00Z", "ext": values},
                "components": {},
            }
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        ),
        target=DocumentSetRef(entry_name="entry", point="working"),
        roles=None,
    )
