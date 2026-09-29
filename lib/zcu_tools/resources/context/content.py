"""In-memory context content operations, separate from file synchronization.

The caller serializes access and owns version/event publication. These operations
preserve live store identities and paths; they neither load nor save files.
"""

from .library import ModuleLibrary
from .metadict import MetaDict


def snapshot_context_contents(
    md: MetaDict, ml: ModuleLibrary
) -> tuple[MetaDict, ModuleLibrary]:
    """Copy the currently published memory into writable, unbacked stores."""
    return md.snapshot(), ml.snapshot()


def replace_context_contents(
    md: MetaDict,
    ml: ModuleLibrary,
    *,
    metadata: MetaDict | None = None,
    library: ModuleLibrary | None = None,
) -> None:
    """Consume prepared stores after all permission checks and copies succeed.

    Unselected stores remain untouched. Candidate data must have been constructed
    through the stores' validation APIs. No callbacks or file I/O occur between
    assignments on the caller's owner sequence. Candidates retain their own data;
    the fresh copies used for swapping are not exposed to callers.
    """
    if metadata is not None:
        md.require_writable()
    if library is not None:
        ml.require_writable()
    metadata = metadata.snapshot() if metadata is not None else None
    library = library.snapshot() if library is not None else None
    if metadata is not None:
        md.swap_contents(metadata)
    if library is not None:
        ml.swap_contents(library)
