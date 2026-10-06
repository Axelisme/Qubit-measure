"""Component model declarations, independent of experiment definitions."""

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

from .errors import UnknownKindError
from .schema import ComponentSchema


@dataclass(frozen=True)
class RoleSpec:
    """Role declaration: kind is a required fnmatch pattern.

    via is a dotted path starting with an already resolved role, or None to
    look for the role-name field on resolved components.
    """

    kind: str
    via: str | None = None


class RoleRegistry:
    """Register notebook/adapter role names without loading experiment modules."""

    def __init__(self) -> None:
        self._roles: dict[str, RoleSpec] = {}
        self._shorthand: tuple[str, ...] = ()
        self._focus_kinds: tuple[str, ...] = ()

    def configure(
        self, *, shorthand: Sequence[str], focus_kinds: Sequence[str]
    ) -> None:
        """Declare notebook roles and kind patterns eligible for default focus.

        All shorthand roles must already be registered and unique. Kind patterns
        must be nonempty. Reject invalid declarations without changing either
        setting. Empty tuples disable shorthand or automatic focus respectively.
        """
        names, patterns = tuple(shorthand), tuple(focus_kinds)
        for name in names:
            self.get(name)
        if len(set(names)) != len(names):
            raise ValueError("Shorthand roles must be unique")
        if any(not pattern for pattern in patterns):
            raise ValueError("A focus kind pattern must not be empty")
        self._shorthand, self._focus_kinds = names, patterns

    @property
    def shorthand(self) -> tuple[str, ...]:
        """Return the declared ordered notebook roles."""
        return self._shorthand

    @property
    def focus_kinds(self) -> tuple[str, ...]:
        """Return the declared patterns eligible for automatic focus."""
        return self._focus_kinds

    def register(self, name: str, spec: RoleSpec) -> None:
        """Register a unique public role name; invalid declarations reserve nothing."""
        if not name.isidentifier() or name.startswith("_") or name == "components":
            raise ValueError(f"Invalid role name {name!r}")
        if name in self._roles:
            raise ValueError(f"Role {name!r} is already registered")
        if not spec.kind:
            raise ValueError("A role kind pattern must not be empty")
        if spec.via is not None:
            parts = spec.via.split(".")
            if len(parts) < 2 or any(
                not part.isidentifier() or part.startswith("_") for part in parts
            ):
                raise ValueError(f"Invalid role reference path {spec.via!r}")
        self._roles[name] = spec

    def unregister(self, name: str) -> None:
        """Remove a declaration; missing names raise KeyError."""
        del self._roles[name]

    def get(self, name: str) -> RoleSpec:
        """Return a registered declaration; unknown names raise ValueError."""
        try:
            return self._roles[name]
        except KeyError as cause:
            raise ValueError(f"Unknown role {name!r}") from cause


def _validate_component_model(model: object) -> None:
    if not isinstance(model, type) or not issubclass(model, ComponentSchema):
        raise TypeError("Registered models must derive from ComponentSchema")


class ComponentRegistry:
    """Map kind names to original models; roles belongs to this registry."""

    def __init__(self) -> None:
        self._models: dict[str, type[ComponentSchema]] = {}
        self.roles = RoleRegistry()

    def register(self, kind: str, model: type[ComponentSchema]) -> None:
        """Register a unique kind and a ComponentSchema subclass.

        The model owns extra policy, defaults, validators and post-init hooks.
        No trial validation runs. Duplicates raise ValueError and invalid model
        types raise TypeError; either failure leaves the registry unchanged.
        """
        if kind in self._models:
            raise ValueError(f"Kind {kind!r} is already registered")
        _validate_component_model(model)
        self._models[kind] = model

    def unregister(self, kind: str) -> None:
        """Remove a kind; missing kinds raise KeyError."""
        del self._models[kind]

    def get(
        self, kind: str, *, source: Path | None = None, component: str | None = None
    ) -> type[ComponentSchema]:
        """Look up kind or raise UnknownKindError with nearby names.

        source and component optionally locate the error in a document; neither
        changes lookup.
        """
        try:
            return self._models[kind]
        except KeyError as cause:
            raise UnknownKindError(
                source, component, kind, tuple(get_close_matches(kind, self._models))
            ) from cause


component_registry = ComponentRegistry()
