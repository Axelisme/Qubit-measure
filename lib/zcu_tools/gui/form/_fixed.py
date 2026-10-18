"""Fixed element identity and participant validation for FixedList fields.

The captured input names each element once. These helpers read those names
from raw input, reject duplicates, require that later inputs keep them
unchanged, and report invalid names or participants as ordinary form errors.
They know no domain meaning of a name.
"""

from collections.abc import Iterator, Mapping

from pydantic import BaseModel, JsonValue

from ._structure import (
    StructureError,
    element_model,
    fixed_list,
    participants,
    selected,
)
from .model import FormPath, FormWritePath

type _Names = tuple[JsonValue, ...]


def _fixed_lists(
    model: type[BaseModel], data: Mapping[str, JsonValue], prefix: FormWritePath = ()
) -> Iterator[tuple[FormWritePath, str, list[JsonValue]]]:
    """Yield each active FixedList as (path, name_field, raw elements)."""
    for name, info in model.model_fields.items():
        value = data.get(name)
        path = (*prefix, name)
        fixed = fixed_list(info)
        if fixed is not None and isinstance(value, list):
            yield path, fixed.name_field, value
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    yield from _fixed_lists(element_model(info), item, (*path, index))
            continue
        child = selected(info, value)
        if child is not None and isinstance(value, dict):
            yield from _fixed_lists(child, value, path)


def _names(elements: list[JsonValue], name_field: str) -> _Names:
    return tuple(
        item.get(name_field) if isinstance(item, dict) else None for item in elements
    )


def identity(
    model: type[BaseModel], data: Mapping[str, JsonValue]
) -> dict[FormWritePath, _Names]:
    """Map each active FixedList path to its ordered raw element names."""
    return {
        path: _names(elements, name_field)
        for path, name_field, elements in _fixed_lists(model, data)
    }


def check_unique_names(model: type[BaseModel], data: Mapping[str, JsonValue]) -> None:
    """Raise ValueError naming the original indices of a duplicate element name."""
    for path, name_field, elements in _fixed_lists(model, data):
        first: dict[str, int] = {}
        for index, name in enumerate(_names(elements, name_field)):
            if not isinstance(name, str):
                continue
            if name in first:
                raise ValueError(
                    f"Duplicate element name {name!r} at {path!r} "
                    f"indices {first[name]} and {index}"
                )
            first[name] = index


def require_same_identity(
    model: type[BaseModel],
    before: Mapping[str, JsonValue],
    after: Mapping[str, JsonValue],
) -> None:
    """Raise StructureError when element names, count or order differ."""
    old, new = identity(model, before), identity(model, after)
    for path in (*old, *(path for path in new if path not in old)):
        if old.get(path) != new.get(path):
            raise StructureError(
                f"Fixed element identity changed at {path!r}: "
                f"{old.get(path)!r} -> {new.get(path)!r}",
                path=path,
            )


def _participants(
    model: type[BaseModel], data: Mapping[str, JsonValue], prefix: FormPath = ()
) -> Iterator[tuple[FormPath, JsonValue, _Names | None]]:
    """Yield each Participants field's path, raw value and its element names."""
    for name, info in model.model_fields.items():
        value = data.get(name)
        path = (*prefix, name)
        declared = participants(info)
        if declared is not None:
            yield path, value, _sibling_names(model, data, declared.list_path)
            continue
        child = selected(info, value)
        if child is not None and isinstance(value, dict):
            yield from _participants(child, value, path)


def _sibling_names(
    model: type[BaseModel], data: Mapping[str, JsonValue], list_path: tuple[str, ...]
) -> _Names | None:
    current = data
    holder = model
    for name in list_path[:-1]:
        info = holder.model_fields[name]
        value = current.get(name)
        child = selected(info, value)
        if child is None or not isinstance(value, dict):
            return None
        current, holder = value, child
    info = holder.model_fields[list_path[-1]]
    fixed = fixed_list(info)
    elements = current.get(list_path[-1])
    if fixed is None or not isinstance(elements, list):
        return None
    return _names(elements, fixed.name_field)


def fixed_errors(
    model: type[BaseModel], data: Mapping[str, JsonValue]
) -> dict[FormPath, str]:
    """Report invalid element names and participants of raw input by path.

    Every element name must be a literal string. A participants list
    must be nonempty, unique and name existing elements. Shapes Pydantic
    already reports, such as a non-list value, are left to it.
    """
    errors: dict[FormPath, str] = {}
    for path, name_field, elements in _fixed_lists(model, data):
        for index, name in enumerate(_names(elements, name_field)):
            if not isinstance(name, str):
                errors[(*path, index, name_field)] = (
                    "Element name must be a literal string"
                )
    for path, value, names in _participants(model, data):
        if not isinstance(value, list) or names is None:
            continue
        if not value:
            errors[path] = "Participants must not be empty"
        seen: set[str] = set()
        for index, entry in enumerate(value):
            if not isinstance(entry, str):
                continue
            if entry in seen:
                errors[(*path, index)] = f"Duplicate participant {entry!r}"
            elif entry not in names:
                errors[(*path, index)] = f"Unknown participant {entry!r}"
            seen.add(entry)
    return errors
