"""Admit readonly and owner-declared mutations, preserving request paths."""

from collections.abc import Iterable, Iterator, Mapping

from pydantic import BaseModel, JsonValue

from ._structure import alternatives, list_model, selected
from .model import Display, FormInput, FormInputRejected, FormModel, FormWritePath
from .options import Option


def _equal(before: JsonValue, after: JsonValue) -> bool:
    # JSON booleans must not compare equal to numbers, including inside containers.
    if isinstance(before, bool) or isinstance(after, bool):
        return type(before) is type(after) and before == after
    if isinstance(before, dict) and isinstance(after, dict):
        return before.keys() == after.keys() and all(
            _equal(value, after[key]) for key, value in before.items()
        )
    if isinstance(before, list) and isinstance(after, list):
        return len(before) == len(after) and all(
            _equal(old, new) for old, new in zip(before, after, strict=True)
        )
    return before == after


def _readonly_fields(
    model: type[BaseModel],
    before: JsonValue,
    after: JsonValue,
    prefix: FormWritePath = (),
) -> Iterator[tuple[FormWritePath, JsonValue, bool]]:
    """Yield readonly paths, current raw values and raw-change flags.

    Recurse through the old structure at matching branches/list positions.
    An ancestor branch switch or nullable deactivation is not a child edit.
    Missing fields/items in a replacement still count as removals, not switches.
    """
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    for name, info in model.model_fields.items():
        path = (*prefix, name)
        old, new = before.get(name), after.get(name)
        display = next(
            (item for item in reversed(info.metadata) if isinstance(item, Display)),
            Display(),
        )
        if display.readonly:
            changed = (name in before) != (name in after) or not _equal(old, new)
            yield path, old, changed
            continue
        element = list_model(info)
        if element is not None and isinstance(old, list):
            items = new if isinstance(new, list) else []
            for index, item in enumerate(old):
                replacement = items[index] if index < len(items) else None
                yield from _readonly_fields(
                    element,
                    item,
                    replacement,
                    (*path, index),
                )
            continue
        if type(None) in alternatives(info.annotation) and (
            old is None or (name in after and new is None)
        ):
            continue
        child = selected(info, old)
        next_child = selected(info, new)
        if child is not None and (next_child is None or child is next_child):
            yield from _readonly_fields(
                child,
                old,
                new,
                path,
            )


def _raw_at(value: JsonValue, path: FormWritePath) -> JsonValue:
    for part in path:
        if isinstance(value, dict) and isinstance(part, str):
            value = value.get(part)
        elif isinstance(value, list) and isinstance(part, int) and part < len(value):
            value = value[part]
        else:
            return None
    return value


def _selection_changed(
    before: Mapping[FormWritePath, JsonValue],
    after: Mapping[FormWritePath, JsonValue],
    path: FormWritePath,
    current: JsonValue,
) -> bool:
    for key in before.keys() | after.keys():
        if key[: len(path)] != path:
            continue
        raw = _raw_at(current, key[len(path) :])
        old, new = before.get(key, raw), after.get(key, raw)
        if _equal(old, new):
            continue
        # A redundant override and the matching raw branch select the same value.
        if all(Option("Current selection", value).matches(raw) for value in (old, new)):
            continue
        return True
    return False


def check_change(
    before: FormInput[FormModel],
    after: FormInput[FormModel],
    paths: Iterable[FormWritePath],
) -> None:
    """Check the complete candidate, mapping a directed refusal to its request."""
    try:
        for path, current, changed in _readonly_fields(
            before.model, before.data, after.data
        ):
            if changed or _selection_changed(
                before.selections, after.selections, path, current
            ):
                raise FormInputRejected(
                    path,
                    f"Readonly field {path!r} cannot change from {current!r}",
                    reason="readonly",
                    current=current,
                )
        before.model.check_input_change(before.data, after.data)
    except FormInputRejected as exc:
        requested = next(
            (
                path
                for path in paths
                if path[: len(exc.path)] == exc.path or exc.path[: len(path)] == path
            ),
            exc.path,
        )
        raise FormInputRejected(
            requested, str(exc), reason=exc.reason, current=exc.current
        ) from exc
