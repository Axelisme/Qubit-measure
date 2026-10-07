"""Ordered role declarations and views over a bound point."""

from collections.abc import Callable, Mapping, Sequence
from fnmatch import fnmatchcase
from types import MappingProxyType
from uuid import UUID

from pydantic import BaseModel

from ._point_origin import PointOrigin
from .errors import RoleResolutionError
from .registry import RoleSpec, component_registry
from .schema import ComponentSchema
from .views import ComponentView


class RoleView:
    """Fixed role-to-component names and values bound to one owning point.

    Obtain this view through PointView.resolve. Component values follow that
    point; resolved names and constructor origin never follow another selection.
    """

    __slots__ = ("_components", "_component", "__dict__")

    def __init__(
        self,
        components: Mapping[str, str],
        component: Callable[[str], ComponentView],
        *,
        origin: PointOrigin,
    ) -> None:
        """Copy resolved names and bind the owning point's constructor identity.

        components maps declared role names to concrete component names.
        component reads/edits that same point; unknown names raise AttributeError.
        origin is its immutable PointOrigin, not an inferred callback identity.
        PointView.resolve supplies all three from one bound point.
        """
        self._origin = origin
        self._components = MappingProxyType(dict(components))
        self._component = component

    @property
    def components(self) -> Mapping[str, str]:
        """Return the fixed, read-only role-to-concrete-name mapping."""
        return self._components

    def __getattr__(self, role: str) -> ComponentView:
        try:
            name = self._components[role]
        except KeyError as cause:
            raise AttributeError(f"Unknown resolved role {role!r}") from cause
        return self._component(name)


def capture_role_components(roles: RoleView, origin: PointOrigin) -> Mapping[str, str]:
    """Copy fixed names for a RoleView from this complete point origin.

    Entry-internal helper, not a package export. roles comes from PointView.resolve;
    origin is the destination point's constructor identity. Compare UUID,
    resolved source and both owning roots, not handle/callback identity.
    Missing or mismatched origin raises ValueError even for an empty mapping.
    No YAML contents or role declarations are read or re-resolved.
    """
    # Entry-internal metadata must not reserve another public role attribute.
    bound: object = vars(roles).get("_origin")
    if not isinstance(bound, PointOrigin):
        raise ValueError("RoleView has no bound point origin")
    if (
        UUID(bound.entry_id) != UUID(origin.entry_id)
        or bound.result_root != origin.result_root
        or bound.database_root != origin.database_root
        or bound.source.resolve() != origin.source.resolve()
    ):
        raise ValueError("RoleView must belong to the same bound point and entry")
    return MappingProxyType(dict(roles.components))


def _reference_target(component: ComponentSchema, path: str) -> str | None:
    node: object = component
    for part in path.split("."):
        if isinstance(node, BaseModel):
            node = (
                getattr(node, part)
                if part in type(node).model_fields
                else (node.model_extra or {}).get(part)
            )
        elif isinstance(node, Mapping):
            node = node.get(part)
        else:
            return None
    return node if isinstance(node, str) else None


def _declarations(
    roles: Mapping[str, RoleSpec] | Sequence[str] | None,
    focus: str | None,
    explicit: Mapping[str, str],
) -> dict[str, RoleSpec]:
    """Normalize only registered role names, rejecting undeclared choices."""
    if roles is None:
        roles = role_registry.shorthand
        if not roles:
            raise RoleResolutionError(
                "shorthand", focus, None, "No shorthand roles declared"
            )
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
        focus = role_registry.default_focus(components)
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


role_registry = component_registry.roles
