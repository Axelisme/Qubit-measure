"""Errors that locate invalid components, role choices and rename failures."""

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


class RoleResolutionError(ValueError):
    """A role cannot be resolved from this point, focus and declaration."""

    def __init__(
        self, role: str, focus: str | None, required_kind: str | None, reason: str
    ) -> None:
        self.role = role
        self.focus = focus
        self.required_kind = required_kind
        self.reason = reason
        super().__init__(
            f"Role {role!r} requires kind {required_kind!r}; focus={focus!r}: {reason}"
        )


class RenameRecoveryError(OSError):
    """The Database rename failed and the Result directory could not be restored.

    moved_result is the Result directory at its new name. pending_database is
    the intended Database destination; that move failed. recovery_destination is
    the original Result path, which recovery could not reach. These are filesystem
    paths derived from rename_entry's explicit roots. cause and recovery_cause
    are the two original I/O errors. No repair runs in this exception.
    """

    def __init__(
        self,
        *,
        moved_result: Path,
        pending_database: Path,
        recovery_destination: Path,
        cause: OSError,
        recovery_cause: OSError,
    ) -> None:
        self.moved_result = moved_result
        self.pending_database = pending_database
        self.recovery_destination = recovery_destination
        self.cause = cause
        self.recovery_cause = recovery_cause
        super().__init__(
            f"Rename recovery failed: moved_result={moved_result!r}, "
            f"pending_database={pending_database!r}, "
            f"recovery_destination={recovery_destination!r}; "
            f"cause={cause}; recovery={recovery_cause}"
        )
