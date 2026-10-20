"""Failures that a CLI may translate without swallowing execution errors."""

from .models import MigrationDifference


class MigrationInputError(ValueError):
    """Malformed input or conflicting migration identity/state; CLI exit code 2.

    The message locates the source or conflicting path. No caller should treat
    this as an empty successful report. Files already published before a
    conflict remain owned by the manifest and are not automatically rolled back.
    differences contains located typed comparisons when available, otherwise ().
    The message includes those comparisons so existing CLI exception rendering
    remains informative without writing a diagnostic into the inspected tree.
    """

    def __init__(
        self, message: str, *, differences: tuple[MigrationDifference, ...] = ()
    ) -> None:
        self.differences = differences
        details = "\n".join(
            f"{item.source} -> {item.destination} [{item.comparison}]: "
            f"expected {item.expected}; actual {item.actual}"
            for item in differences
        )
        super().__init__(f"{message}\n{details}" if details else message)
