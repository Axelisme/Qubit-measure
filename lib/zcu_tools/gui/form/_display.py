"""Resolve Display overlays without changing other metadata policies."""

from typing import Annotated, TypeAliasType, get_args, get_origin

from pydantic.fields import FieldInfo

from .model import Display


def field_display(info: FieldInfo) -> Display:
    """Resolve explicit Display properties, outer declarations winning per key.

    FieldInfo metadata is ordered inner-to-outer. Named aliases retain their
    own Annotated layers, so collect those too before applying the overlays.
    The result retains omission information for callers that reuse metadata.
    """
    layers = [item for item in reversed(info.metadata) if isinstance(item, Display)]
    annotation = info.annotation
    while isinstance(annotation, TypeAliasType) or get_origin(annotation) is Annotated:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
            continue
        annotation, *metadata = get_args(annotation)
        layers.extend(item for item in reversed(metadata) if isinstance(item, Display))
    display = Display()
    for layer in reversed(layers):
        display = display.overlay(layer)
    return display
