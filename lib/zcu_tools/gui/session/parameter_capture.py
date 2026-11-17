"""Session capture/restore adapter for experiment-owned native snapshots."""

from zcu_tools.experiment.metadata import ParameterSnapshot
from zcu_tools.gui.session.parameters import DocumentSetCapture
from zcu_tools.resources.context.library import LibrarySnapshot


def capture_parameters(parameters: DocumentSetCapture) -> ParameterSnapshot:
    """Detach fixed point, library, authored aliases and names for RunMetadata.

    Never refresh files or selection. Resource or alias source errors propagate.
    No persistent target or storage capability is serialized.
    """
    return ParameterSnapshot(
        point=parameters.point_content(),
        library=parameters.library.content(),
        aliases=parameters.alias_content(),
        names=parameters.name_content(),
    )


def restore_parameters(snapshot: ParameterSnapshot) -> DocumentSetCapture:
    """Restore recorded content without I/O, eager validation or a write target.

    Resource and session owners validate only the values requested by readers.
    Missing or malformed unused sections remain recorded, not replaced with empty.
    """
    return DocumentSetCapture.from_content(
        point=snapshot.point,
        library=LibrarySnapshot.from_content(snapshot.library),
        aliases=snapshot.aliases,
        names=snapshot.names,
    )
