"""Native fixed parameter captures for shared expression UI contracts."""

from zcu_tools.format_version import YamlMap
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument


def make_parameters(values: YamlMap) -> ParameterContext:
    """Capture numeric test values under general.ext in a memory-only entry draft."""
    return ParameterContext(
        point=PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-08T00:00:00Z", "ext": values},
                "components": {},
            }
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=EntryDraft(entry_id="entry"),
        roles=None,
    )
