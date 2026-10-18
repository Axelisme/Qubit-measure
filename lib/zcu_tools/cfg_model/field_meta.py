"""Front-end-neutral descriptions attached to configuration fields."""

from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Union, get_args, get_origin

from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class FieldMeta:
    """Describe a cfg field without changing its validation or serialized value.

    Attach with ``Annotated[T, FieldMeta(...)]``. All attributes are optional;
    None leaves a hint unspecified, except group where it means no group.
    readonly defaults to False and never enforces a write restriction.

    Attributes
    ----------
    label:
        Human-readable field name, without the unit suffix.
    unit:
        Unit of the stored value, such as "MHz" or "us"; does not convert values.
    decimals:
        Suggested number of fractional decimal places for numeric display.
        Use a non-negative integer, or None for the consumer's default precision.
    description:
        One-sentence explanation of what the field controls.
    group:
        Presentation group name, such as "Advanced", or None for no group.
        An empty string raises ValueError at construction.
    readonly:
        Request non-editable presentation; does not restrict writes or validation.
    element_label:
        Name of a string field used to label model-list elements, or None to
        display their original indexes. Does not change identity or ordering.
    """

    label: str | None = None
    unit: str | None = None
    decimals: int | None = None
    description: str | None = None
    group: str | None = None
    readonly: bool = False
    element_label: str | None = None

    def __post_init__(self) -> None:
        """Reject an empty presentation group with ValueError."""
        if self.group == "":
            raise ValueError("group must be non-empty or None")


def get_field_meta(field: FieldInfo) -> FieldMeta | None:
    """Read a Pydantic ``model_fields`` entry's description, or return None.

    Read field-level metadata first, then annotations inside Annotated and
    union wrappers (including optional fields). The first FieldMeta wins;
    union members are visited in declaration order. Other metadata is ignored.
    Container elements and nested model fields are separate fields, not searched.
    The field is not modified.
    """
    for metadata in field.metadata:
        if isinstance(metadata, FieldMeta):
            return metadata

    pending: list[object] = [field.annotation]
    while pending:
        annotation = pending.pop(0)
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is Annotated:
            for metadata in args[1:]:
                if isinstance(metadata, FieldMeta):
                    return metadata
            pending[0:0] = args[:1]
        elif origin in (Union, UnionType):
            pending[0:0] = args
    return None
