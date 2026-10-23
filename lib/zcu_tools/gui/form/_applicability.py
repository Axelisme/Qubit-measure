"""Membership-only applicability projection; never mutates authored drafts."""

from collections.abc import Mapping
from typing import get_origin

from pydantic import JsonValue

from ._json import copy_object
from ._selection import resolve
from ._structure import alternatives
from .model import FormEnv, FormModel, FormWritePath


def _at(data: JsonValue, path: FormWritePath) -> JsonValue:
    for part in path:
        if isinstance(data, dict) and isinstance(part, str):
            data = data.get(part)
        elif isinstance(data, list) and type(part) is int and 0 <= part < len(data):
            data = data[part]
        else:
            return None
    return data


def disabled_paths(
    model: type[FormModel], data: Mapping[str, JsonValue], env: FormEnv
) -> tuple[FormWritePath, ...]:
    """Resolve root declarations and evaluate only their list dependencies.

    Invalid dependency values disable their dependents, but remain in the input
    for normal decoding and validation. Declaration faults propagate as ValueError.
    """
    from ._evaluate import evaluate_dependency

    declarations = model.applicability(copy_object(data))
    dependencies = {item.dependency for item in declarations}
    controlled: list[FormWritePath] = []
    disabled: list[FormWritePath] = []
    for item in declarations:
        if not item.dependency or not item.paths:
            raise ValueError("Applicability requires a dependency and nullable paths")
        dependency = resolve(model, data, item.dependency, env)
        if get_origin(dependency.info.annotation) is not list:
            raise ValueError(
                f"{item.dependency!r}: applicability dependency requires a list"
            )
        values = evaluate_dependency(
            dependency.info,
            _at(copy_object(data), item.dependency),
            item.dependency,
            env,
        )
        applicable = values is not None and item.value in values
        for path in item.paths:
            if not path or any(
                path[: len(other)] == other or other[: len(path)] == path
                for other in (*controlled, *dependencies)
            ):
                raise ValueError(
                    f"{path!r}: overlapping or chained applicability declaration"
                )
            target = resolve(model, data, path, env)
            if type(None) not in alternatives(target.info.annotation):
                raise ValueError(f"{path!r}: applicability requires a nullable field")
            controlled.append(path)
            if not applicable:
                disabled.append(path)
    return tuple(disabled)


def project(
    model: type[FormModel], data: Mapping[str, JsonValue], env: FormEnv
) -> dict[str, JsonValue]:
    """Copy input and replace inapplicable nullable subtrees with None."""
    result = copy_object(data)
    for path in disabled_paths(model, data, env):
        parent = _at(result, path[:-1])
        name = path[-1]
        if isinstance(parent, dict) and isinstance(name, str):
            parent[name] = None
            continue
        if isinstance(parent, list) and type(name) is int and 0 <= name < len(parent):
            parent[name] = None
            continue
        raise ValueError(f"{path!r}: applicability parent is not reachable")
    return result
