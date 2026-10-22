"""Public Pydantic static defaults for input authors and branch creation."""

from typing import Annotated, TypeAliasType, get_args, get_origin

from pydantic import BaseModel, JsonValue
from pydantic.fields import FieldInfo

from ._json import copy_json, default_json
from .model import InputDefault


def _input_default(info: FieldInfo) -> InputDefault | None:
    for item in reversed(info.metadata):
        if isinstance(item, InputDefault):
            return item
    annotation = info.annotation
    while isinstance(annotation, TypeAliasType) or get_origin(annotation) is Annotated:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
            continue
        annotation, *metadata = get_args(annotation)
        for item in reversed(metadata):
            if isinstance(item, InputDefault):
                return item
    return None


def static_defaults(model: type[BaseModel]) -> dict[str, JsonValue]:
    """Encode declared defaults as fresh input JSON without validating a model.

    ``model`` may be any Pydantic BaseModel class, including persistent Specs.
    Required fields remain absent unless InputDefault metadata supplies fresh
    raw JSON. Declared defaults and default_factory take precedence, including
    defaults equal to None, which remain absent so before validators can derive
    omitted values. Factories run once per call. Values use
    the form complex envelope and detached JSON containers. Invalid non-finite
    or non-JSON defaults raise ValueError; factory failures propagate. No source
    lookup, evaluation or I/O is performed by this function.
    """
    result: dict[str, JsonValue] = {}
    for name, info in model.model_fields.items():
        if info.is_required():
            fresh = _input_default(info)
            if fresh is not None:
                result[name] = copy_json(fresh.factory())
        else:
            value = info.get_default(call_default_factory=True)
            if value is not None:
                result[name] = default_json(value)
    return result
