"""Failures that a CLI may translate without swallowing execution errors."""


class MigrationInputError(ValueError):
    """Malformed input or conflicting migration identity/state; CLI exit code 2.

    The message locates the source or conflicting path. No caller should treat
    this as an empty successful report. Files already published before a
    conflict remain owned by the manifest and are not automatically rolled back.
    """
