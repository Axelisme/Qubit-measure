"""Ordered role declarations and views over a bound point."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from types import MappingProxyType

from pydantic import BaseModel

from .errors import RoleResolutionError
from .registry import component_registry
from .schema import ComponentSchema
from .views import ComponentView


@dataclass(frozen=True)
class RoleSpec:
    """Required component kind pattern and optional resolved-role reference path."""

    kind: str
    via: str | None = None


class RoleRegistry:
    """Register notebook/adapter role names without loading experiment modules."""

    def __init__(self) -> None:
        self._roles: dict[str, RoleSpec] = {}

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


class RoleView:
    """Frozen name mapping with component values read/written through its point."""

    __slots__ = ("_components", "_component")

    def __init__(
        self,
        components: Mapping[str, str],
        component: Callable[[str], ComponentView],
    ) -> None:
        self._components = MappingProxyType(dict(components))
        self._component = component

    @property
    def components(self) -> Mapping[str, str]:
        return self._components

    def __getattr__(self, role: str) -> ComponentView:
        try:
            name = self._components[role]
        except KeyError as cause:
            raise AttributeError(f"Unknown resolved role {role!r}") from cause
        return self._component(name)


def _reference_target(component: ComponentSchema, path: str) -> str | None:
    if path not in component_registry.references(component.kind):
        return None
    node: object = component
    for part in path.split("."):
        node = getattr(node, part, None) if isinstance(node, BaseModel) else None
    return node if isinstance(node, str) else None


def _declarations(
    roles: Mapping[str, RoleSpec] | Sequence[str] | None,
    focus: str | None,
    explicit: Mapping[str, str],
) -> dict[str, RoleSpec]:
    """Normalize only registered role names, rejecting undeclared choices."""
    if roles is None:
        roles = ("qubit", "readout")
    declarations: dict[str, RoleSpec] = {}
    for role in roles:
        try:
            registered = role_registry.get(role)
        except ValueError as cause:
            raise RoleResolutionError(role, focus, None, str(cause)) from cause
        declarations[role] = roles[role] if isinstance(roles, Mapping) else registered
    for role in explicit:
        if role not in declarations:
            raise RoleResolutionError(role, focus, None, "Role is not declared")
    return declarations


def _derived_name(
    components: Mapping[str, ComponentSchema],
    resolved: Mapping[str, str],
    role: str,
    spec: RoleSpec,
    focus: str | None,
) -> str:
    if spec.via is not None:
        source_role, separator, path = spec.via.partition(".")
        target = (
            _reference_target(components[resolved[source_role]], path)
            if separator and source_role in resolved
            else None
        )
        candidates = {target} if target is not None else set()
    else:
        candidates = {
            target
            for name in resolved.values()
            if (target := _reference_target(components[name], role)) is not None
        }
    if len(candidates) != 1:
        reason = (
            f"Ambiguous references: {sorted(candidates)}"
            if candidates
            else f"No component choice via {spec.via!r} or same-name reference"
        )
        raise RoleResolutionError(role, focus, spec.kind, reason)
    return candidates.pop()


def resolve_names(
    components: Mapping[str, ComponentSchema],
    roles: Mapping[str, RoleSpec] | Sequence[str] | None,
    focus: str | None,
    explicit: Mapping[str, str],
) -> dict[str, str]:
    """Resolve declared names from one already validated working-unit snapshot."""
    declarations = _declarations(roles, focus, explicit)
    if focus is None:
        qubits = [
            name
            for name, component in components.items()
            if fnmatchcase(component.kind, "qubit/*")
        ]
        if len(qubits) == 1:
            focus = qubits[0]
    resolved: dict[str, str] = {}
    focus_used = False
    for role, spec in declarations.items():
        if role in explicit:
            name = explicit[role]
        elif (
            not focus_used
            and focus is not None
            and focus in components
            and fnmatchcase(components[focus].kind, spec.kind)
        ):
            name = focus
            focus_used = True
        else:
            name = _derived_name(components, resolved, role, spec, focus)
        if name not in components:
            raise RoleResolutionError(
                role, focus, spec.kind, f"Unknown component {name!r}"
            )
        if not fnmatchcase(components[name].kind, spec.kind):
            raise RoleResolutionError(
                role,
                focus,
                spec.kind,
                f"Component {name!r} has kind {components[name].kind!r}",
            )
        resolved[role] = name
    return resolved


role_registry = RoleRegistry()
for _name, _kind in (
    ("qubit", "qubit/*"),
    ("readout", "resonator"),
    ("control", "qubit/*"),
    ("target", "qubit/*"),
    ("coupler", "coupler/*"),
):
    role_registry.register(_name, RoleSpec(_kind))
