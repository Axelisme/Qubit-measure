from __future__ import annotations


class DressedLabelingError(RuntimeError):
    """A requested bare state could not be mapped to a dressed state."""


def require_dressed_index(
    index: int | None,
    bare_state: tuple[int, ...],
    *,
    context: str,
) -> int:
    if index is None:
        raise DressedLabelingError(
            f"scqubits could not label dressed state {bare_state} ({context})"
        )
    return int(index)
