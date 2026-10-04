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


class MissingReferenceError(ValueError):
    def __init__(self, source: Path, component: str, field: str, target: str) -> None:
        self.source = source
        self.component = component
        self.field = field
        self.target = target
        super().__init__(
            f"{source}: {component}.{field}: component reference {target!r} does not exist"
        )


class LayerConflictError(ValueError):
    """A component leaf is supplied by both setup and point.

    path is its logical dotted field path, such as Q1.t1. setup_file and
    point_file locate the two conflicting documents; their values are not read
    by this exception. These three arguments are also exposed as attributes.
    """

    def __init__(self, path: str, setup_file: Path, point_file: Path) -> None:
        """Describe the duplicated leaf and files without changing either file."""
        self.path = path
        self.setup_file = setup_file
        self.point_file = point_file
        super().__init__(f"{path}: present in both {setup_file} and {point_file}")


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
