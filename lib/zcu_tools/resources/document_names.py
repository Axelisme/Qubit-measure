"""Canonical entry/point document filenames and explicit offline rename pairs."""

PARAMETERS_FILENAME = "parameters.yaml"
LIBRARY_FILENAME = "library.yaml"
ALIASES_FILENAME = "aliases.yaml"

LEGACY_SETUP_FILENAME = "setup.yaml"
LEGACY_POINT_FILENAME = "point.yaml"
LEGACY_LIBRARY_FILENAME = "module_cfg.yaml"

LEGACY_DOCUMENT_NAMES = (
    (LEGACY_SETUP_FILENAME, PARAMETERS_FILENAME),
    (LEGACY_POINT_FILENAME, PARAMETERS_FILENAME),
    (LEGACY_LIBRARY_FILENAME, LIBRARY_FILENAME),
)
"""Old basename and canonical basename pairs, used only for explicit upgrades."""
