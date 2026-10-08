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
    Declared but unbound roles raise RoleResolutionError on getattr and hasattr;
    this ValueError is not treated as a missing attribute. Unknown attributes
    still raise AttributeError. Introspection tools must handle declared-role
    read failures instead of treating them as absent attributes.
    """

    __slots__ = ("_components", "_component", "__dict__")

    def __init__(
        self,
        components: Mapping[str, str],
        component: Callable[[str], ComponentView],
        *,
        origin: PointOrigin,
        roles: Mapping[str, RoleSpec] | Sequence[str] | None,
        focus: str | None,
    ) -> None:
        """Copy resolved names and bind the owning point's constructor identity.

        components maps declared role names to concrete component names.
        component reads/edits that same point; unknown names raise AttributeError.
        origin is its immutable PointOrigin, not an inferred callback identity.
        roles is the same ordered declaration passed to PointView.resolve;
        None uses the registered notebook shorthand. focus is that call\'s
        component name or None. These fix unbound-role error context without
        rerunning inference. PointView.resolve supplies all inputs.
        """
        self._origin = origin
        self._declarations = _declarations(roles, focus, {})
        self._focus = focus
        self._components = MappingProxyType(dict(components))
        self._component = component

    @property
    def components(self) -> Mapping[str, str]:
        """Return the fixed, read-only role-to-concrete-name mapping."""
        return self._components

    def __getattr__(self, role: str) -> ComponentView:
        """Read a bound role; declared unbound roles raise RoleResolutionError.

        Unknown attributes raise AttributeError. hasattr for a declared unbound
        role propagates RoleResolutionError because it is a ValueError.
        """
        try:
            name = self._components[role]
        except KeyError as cause:
            if role in self._declarations:
                raise RoleResolutionError(
                    role,
                    self._focus,
                    self._declarations[role].kind,
                    "Role is not bound",
                ) from cause
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
    explicit: Mapping[str, str | None],
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


def _validate_target(
    components: Mapping[str, ComponentSchema],
    name: str,
    role: str,
    spec: RoleSpec,
    focus: str | None,
) -> None:
    """Reject unknown or wrong-kind targets before binding or ambiguity omission."""
    if name not in components:
        raise RoleResolutionError(role, focus, spec.kind, f"Unknown component {name!r}")
    if not fnmatchcase(components[name].kind, spec.kind):
        raise RoleResolutionError(
            role,
            focus,
            spec.kind,
            f"Component {name!r} has kind {components[name].kind!r}",
        )


def _derived_name(
    components: Mapping[str, ComponentSchema],
    resolved: Mapping[str, str],
    role: str,
    spec: RoleSpec,
    focus: str | None,
) -> str | None:
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
    for target in sorted(candidates):
        _validate_target(components, target, role, spec, focus)
    if len(candidates) != 1:
        return None
    return candidates.pop()


def resolve_names(
    components: Mapping[str, ComponentSchema],
    roles: Mapping[str, RoleSpec] | Sequence[str] | None,
    focus: str | None,
    explicit: Mapping[str, str | None],
) -> dict[str, str]:
    """Infer fixed role aliases from one validated component snapshot.

    roles is an ordered registered declaration mapping/sequence; None uses the
    notebook shorthand. explicit maps declared roles to concrete names or None
    and wins over focus/via inference. None explicitly leaves that role unbound
    and consumes the one focus assignment when the focus matches its kind; it
    never shifts that focus to another role. Omitted keys allow inference. focus is a concrete name or None for the registry
    default. Automatic zero/multiple candidates omit the role; missing via
    source roles omit their dependents. Return only bound aliases. Unknown
    declarations/explicit choices and dangling/wrong-kind targets raise
    RoleResolutionError. No source reads or working-unit mutation occur.
    """
    declarations = _declarations(roles, focus, explicit)
    if focus is None:
        focus = role_registry.default_focus(components)
    resolved: dict[str, str] = {}
    focus_used = False
    for role, spec in declarations.items():
        if role in explicit:
            name = explicit[role]
            if name is None:
                if (
                    focus is not None
                    and focus in components
                    and fnmatchcase(components[focus].kind, spec.kind)
                ):
                    focus_used = True
                continue
            if name == focus:
                focus_used = True
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
            if name is None:
                continue
        _validate_target(components, name, role, spec, focus)
        resolved[role] = name
    return resolved


role_registry = component_registry.roles
