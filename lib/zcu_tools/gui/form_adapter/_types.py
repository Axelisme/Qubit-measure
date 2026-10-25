"""Annotation operations at the dynamic core-to-form translation boundary."""

from operator import methodcaller
from types import UnionType
from typing import Annotated, Literal, TypeAliasType, Union, get_args, get_origin

from pydantic import BaseModel


def members(annotation: object) -> tuple[object, ...]:
    if isinstance(annotation, TypeAliasType):
        return members(annotation.__value__)
    args = get_args(annotation)
    if get_origin(annotation) is Annotated:
        return members(args[0])
    if get_origin(annotation) in (UnionType, Union):
        return tuple(member for arg in args for member in members(arg))
    return (annotation,)


def branch_metadata(annotation: object, member: type) -> tuple[object, ...]:
    """Read metadata on one type branch, including annotations inside nullable unions."""
    if isinstance(annotation, TypeAliasType):
        return branch_metadata(annotation.__value__, member)
    args = get_args(annotation)
    if get_origin(annotation) is Annotated:
        if member not in members(args[0]):
            return ()
        return (*branch_metadata(args[0], member), *args[1:])
    if get_origin(annotation) in (UnionType, Union):
        return tuple(item for arg in args for item in branch_metadata(arg, member))
    return ()


def union_type(items: tuple[object, ...]) -> object:
    return items[0] if len(items) == 1 else methodcaller("__getitem__", items)(Union)


def annotated_type(annotation: object, metadata: object) -> object:
    return methodcaller("__getitem__", (annotation, metadata))(Annotated)


def type_tag(model: type[BaseModel]) -> str | None:
    info = model.model_fields.get("type")
    if info is not None and get_origin(info.annotation) is Literal:
        tags = get_args(info.annotation)
        if len(tags) == 1 and isinstance(tag := tags[0], str):
            return tag
    return None
