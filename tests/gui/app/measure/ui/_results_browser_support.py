"""Visible results navigation captions shared by browser interaction scenarios."""

from tests.gui._user_input import ComponentDriver


def navigation_path(ui: ComponentDriver, path: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve semantic captions to their currently visible decorated path."""
    matches = [
        row.path
        for row in ui.visible().tree_rows
        if tuple(part.split("  (", 1)[0] for part in row.path) == path
    ]
    if len(matches) != 1:
        raise LookupError(f"{path!r}: {len(matches)} visible paths")
    return matches[0]
