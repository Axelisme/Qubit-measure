"""Errors that locate invalid components and existing destinations."""

from pathlib import Path


class DestinationExistsError(FileExistsError):
    """An entry-owned new destination already exists, including a symlink.

    This marks the explicit destination check, not arbitrary filesystem failures
    while creating, cloning or committing resources. FileExistsError errno,
    filename and message remain available to public resource callers.
    """


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
    """A role or alias cannot be resolved from this point, focus and declaration.

    role is the role or alias name, without a $ prefix. focus is the focus
    component name, or None when resolution used no focus. required_kind is the
    consumer's fnmatch kind pattern, or None when no kind is required. reason is
    a short classification, such as "Alias is not bound", that callers may
    compare. message is the complete user-facing text when the raising owner has
    better wording, such as the session alias table naming $alias and its
    remedy. Without message, str(exc) names the role, the required kind and
    focus only when they are set, and the reason.
    """

    def __init__(
        self,
        role: str,
        focus: str | None,
        required_kind: str | None,
        reason: str,
        *,
        message: str | None = None,
    ) -> None:
        self.role = role
        self.focus = focus
        self.required_kind = required_kind
        self.reason = reason
        if message is None:
            message = f"Role {role!r}"
            if required_kind is not None:
                message += f" requires kind {required_kind!r}"
            if focus is not None:
                message += f" (focus {focus!r})"
            message += f": {reason}"
        super().__init__(message)
