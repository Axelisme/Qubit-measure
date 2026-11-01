"""Wave-output port sharing rules for board profiles and compiled programs."""

from collections.abc import Iterable


def wave_port_conflicts(
    selector: int | None, used_selectors: Iterable[int | None]
) -> bool:
    """Return whether a selector conflicts with users of the same wave port.

    An unselected user (None) needs exclusive ownership. Selected users may share
    a port only with distinct non-None selectors. An empty iterable never conflicts.
    This predicate reads used_selectors without modifying any port registry.
    """
    return any(
        selector is None or used is None or selector == used for used in used_selectors
    )
