"""Read suggestion declarations through string and nullable annotations."""

from types import UnionType
from typing import Annotated, TypeAliasType, Union, get_args, get_origin

from pydantic.fields import FieldInfo

from .model import InputSuggestions


def field_suggestions(info: FieldInfo) -> InputSuggestions | None:
    """Find the outermost suggestion declaration, without searching containers."""
    for item in reversed(info.metadata):
        if isinstance(item, InputSuggestions):
            return item
    pending: list[object] = [info.annotation]
    while pending:
        annotation = pending.pop(0)
        if isinstance(annotation, TypeAliasType):
            pending.insert(0, annotation.__value__)
            continue
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is Annotated:
            for item in reversed(args[1:]):
                if isinstance(item, InputSuggestions):
                    return item
            pending[0:0] = args[:1]
        elif origin in (Union, UnionType):
            pending[0:0] = args
    return None
