"""User-owned experiment definitions and frontend attachments."""

from pathlib import Path

from zcu_tools.gui.session.alias_presets import AliasPresets, validate_alias_presets
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.resources.entry.aliases import load_entry_aliases


def load_alias_presets() -> AliasPresets:
    """Read this package's alias_presets/*.yaml once for GUI startup.

    File stems are preset names, ordered by filename. Preserve authored row
    order and raw links using the resources zcu.aliases reader. Require a
    default preset (its table may be empty). Missing default raises ValueError;
    vanished files raise FileNotFoundError. Reader format, validation and I/O
    errors, or invalid AliasTable syntax, propagate without skipping or fallback.
    No experiments are imported. Changes require a new GUI startup.
    """
    presets: dict[str, tuple[AliasRow, ...]] = {}
    for path in sorted(Path(__file__).with_name("alias_presets").glob("*.yaml")):
        document = load_entry_aliases(path)
        if document is None:
            raise FileNotFoundError(path)
        presets[path.stem] = tuple(
            AliasRow(alias, value) for alias, value in document.aliases.items()
        )
    validate_alias_presets(presets)
    return presets


__all__ = ["load_alias_presets"]
