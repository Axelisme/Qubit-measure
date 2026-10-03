"""Errors that identify partial entry filesystem changes."""

from pathlib import Path


class UnknownFieldError(AttributeError):
    def __init__(self, path: str, field: str, suggestions: tuple[str, ...]) -> None:
        self.path = path
        self.field = field
        self.suggestions = suggestions
        super().__init__(f"{path}: unknown field {field!r}; suggestions={suggestions}")


class UnknownKindError(ValueError):
    def __init__(
        self,
        source: Path | None,
        component: str | None,
        kind: str,
        suggestions: tuple[str, ...],
    ) -> None:
        self.source = source
        self.component = component
        self.kind = kind
        self.suggestions = suggestions
        super().__init__(
            f"{source}: {component}: unknown kind {kind!r}; suggestions={suggestions}"
        )


class PartialCommitError(OSError):
    def __init__(
        self,
        *,
        completed: tuple[Path, ...],
        pending: tuple[Path, ...],
        recovery_failed: tuple[Path, ...],
        cause: OSError,
        recovery_cause: OSError,
    ) -> None:
        self.completed = completed
        self.pending = pending
        self.recovery_failed = recovery_failed
        self.cause = cause
        self.recovery_cause = recovery_cause
        super().__init__(
            f"Partial commit: completed={completed!r}, pending={pending!r}, "
            f"recovery_failed={recovery_failed!r}; cause={cause}; recovery={recovery_cause}"
        )
