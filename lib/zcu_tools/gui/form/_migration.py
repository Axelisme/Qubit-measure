"""Lossy migration of authored data and annotation-compatible choices."""

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo

from ._persistence import check_selection
from ._structure import element_model, fixed_list, selected
from .model import FormPath, FormWritePath


def _selection_info(
    model: type[BaseModel], data: dict[str, JsonValue], path: FormWritePath
) -> FieldInfo:
    name, *tail = path
    if not isinstance(name, str) or (info := model.model_fields.get(name)) is None:
        raise ValueError("Selection field was removed")
    if not tail:
        return info
    value = data.get(name)
    if fixed_list(info) is not None:
        index, *tail = tail
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not isinstance(value, list)
            or not 0 <= index < len(value)
            or not tail
        ):
            raise ValueError("Selection element is no longer reachable")
        value = value[index]
        child = element_model(info)
    else:
        child = selected(info, value)
    if child is None or not isinstance(value, dict):
        raise ValueError("Selection branch is no longer reachable")
    return _selection_info(child, value, tuple(tail))


def migrate_selections(
    model: type[BaseModel],
    data: dict[str, JsonValue],
    selections: dict[FormWritePath, JsonValue],
) -> tuple[dict[FormWritePath, JsonValue], tuple[FormPath, ...]]:
    """Keep only choices accepted by the new annotation and migrated ancestry.

    Return retained overrides and discarded paths. No option hook, converter,
    source lookup or ordinary value validation runs here.
    """
    retained: dict[FormWritePath, JsonValue] = {}
    dropped: list[FormPath] = []
    for path, value in selections.items():
        try:
            if not path:
                raise ValueError("Selection requires a field path")
            info = _selection_info(model, data, path)
            check_selection(info, value)
        except ValueError:
            dropped.append(path)
        else:
            retained[path] = value
    return retained, tuple(dropped)
