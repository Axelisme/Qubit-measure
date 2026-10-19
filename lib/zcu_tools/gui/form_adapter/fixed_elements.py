"""Declaration-owned fixed identities and participant value validation.

Core only calls the neutral input-change contract. This adapter capability
assigns domain meaning to named list elements; widgets and transports do not.
"""

from dataclasses import dataclass
from typing import ClassVar, Self, get_args, get_origin

from pydantic import BaseModel, JsonValue, model_validator
from pydantic.fields import FieldInfo

from zcu_tools.core.element import ElementIdentity
from zcu_tools.gui.form import (
    FormEnv,
    FormInput,
    FormInputRejected,
    FormModel,
    FormWritePath,
    field_error,
)

from ._types import members


def _field(model: type[BaseModel], path: FormWritePath) -> FieldInfo:
    info = FieldInfo.from_annotation(model)
    for part in path:
        (annotation,) = members(info.annotation)
        if (
            isinstance(part, str)
            and isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
        ):
            field = annotation.model_fields.get(part)
            if field is None:
                raise ValueError(f"Unknown fixed-elements declaration path: {path!r}")
            info = field
        elif (
            isinstance(part, int)
            and not isinstance(part, bool)
            and get_origin(annotation) is list
        ):
            info = FieldInfo.from_annotation(get_args(annotation)[0])
        else:
            raise ValueError(f"Invalid fixed-elements declaration path: {path!r}")
    return info


def _at(data: JsonValue, path: FormWritePath) -> JsonValue:
    for part in path:
        if isinstance(data, dict) and isinstance(part, str):
            data = data.get(part)
        elif isinstance(data, list) and isinstance(part, int) and 0 <= part < len(data):
            data = data[part]
        else:
            return None
    return data


@dataclass(frozen=True)
class FixedElements:
    """One owner's named list and its independent ordered participants field.

    list_path and participants_path are nonempty root-relative model-field
    paths, optionally through model-list indexes. name_field is a literal str
    field on every element. The list's length, order and names cannot change
    through ordinary mutation; restored input establishes a new boundary.
    Participants remain ordinary invalid drafts until model validation.
    Declare Display(element_label=name_field) on the list and
    Display(readonly=True) on the name for presentation; neither enforces this
    policy. No sources, captured identities or secondary ordering are stored.
    """

    list_path: FormWritePath
    name_field: str
    participants_path: FormWritePath

    def __post_init__(self) -> None:
        """Reject empty paths or name_field with ValueError."""
        if not self.list_path or not self.participants_path or not self.name_field:
            raise ValueError(
                "Fixed elements require list/participants paths and a name field"
            )

    def check_declaration(self, model: type[BaseModel]) -> None:
        """Reject paths not naming list[Model], a str name, and list[str]."""
        info = _field(model, self.list_path)
        if get_origin(info.annotation) is not list:
            raise ValueError(f"{self.list_path!r}: fixed elements require list[Model]")
        element = get_args(info.annotation)[0]
        if not isinstance(element, type) or not issubclass(element, BaseModel):
            raise ValueError(f"{self.list_path!r}: fixed elements require list[Model]")
        name = element.model_fields.get(self.name_field)
        if name is None or members(name.annotation) != (str,):
            raise ValueError(
                f"{self.list_path!r}: {self.name_field!r} must name a str field"
            )
        if _field(model, self.participants_path).annotation != list[str]:
            raise ValueError(
                f"{self.participants_path!r}: participants require list[str]"
            )

    def names(self, data: dict[str, JsonValue]) -> tuple[JsonValue, ...] | None:
        """Read ordered raw names, or None for a missing/non-list container."""
        elements = _at(data, self.list_path)
        if not isinstance(elements, list):
            return None
        return tuple(
            item.get(self.name_field) if isinstance(item, dict) else None
            for item in elements
        )

    def identify(
        self, data: dict[str, JsonValue], path: FormWritePath
    ) -> ElementIdentity | None:
        """Read a declared element identity at path from the same input capture.

        path is a root-relative field path through this declaration's list and
        its original integer index. Descendants share that element identity.
        Return None for paths outside this list. Missing/nonliteral names and
        invalid indexes inside the list raise ValueError; never use participants.
        """
        prefix = len(self.list_path)
        if path[:prefix] != self.list_path or len(path) <= prefix:
            return None
        index = path[prefix]
        names = self.names(data)
        if type(index) is not int or names is None or not 0 <= index < len(names):
            raise ValueError(f"Invalid fixed element path: {path!r}")
        name = names[index]
        if not isinstance(name, str):
            raise ValueError(f"Missing literal element name at {path!r}")
        return ElementIdentity(index=index, name=name)

    def check_unique_names(self, data: dict[str, JsonValue]) -> None:
        """Reject duplicate capture names, naming the first and repeated indices."""
        names = self.names(data)
        if names is None:
            raise ValueError(f"Missing captured list at {self.list_path!r}")
        seen: dict[str, int] = {}
        for index, name in enumerate(names):
            if not isinstance(name, str):
                raise ValueError(
                    f"{(*self.list_path, index, self.name_field)!r}: name must be a literal string"
                )
            if name in seen:
                raise ValueError(
                    f"Duplicate element name {name!r} at {self.list_path!r} "
                    f"indices {seen[name]} and {index}"
                )
            seen[name] = index

    def check_input_change(
        self, before: dict[str, JsonValue], after: dict[str, JsonValue]
    ) -> None:
        """Refuse ordinary changes to ordered raw identities, not ordinary values."""
        old, new = self.names(before), self.names(after)
        if old != new:
            raise FormInputRejected(
                self.list_path,
                f"Fixed element identity changed at {self.list_path!r}: {old!r} -> {new!r}",
            )

    def validate_participants(self, data: dict[str, JsonValue]) -> None:
        """Raise a directed ordinary Pydantic error for illegal participant values.

        Report empty, duplicate and unknown names at the declared field/index.
        Never reorder or filter either list. Pydantic handles non-list shapes
        and non-string entries before this successful-model check.
        """
        names = self.names(data)
        values = _at(data, self.participants_path)
        if names is None or not isinstance(values, list):
            return
        if not values:
            raise field_error(self.participants_path, "Participants must not be empty")
        seen: set[str] = set()
        for index, value in enumerate(values):
            if not isinstance(value, str):
                continue
            if value in seen:
                raise field_error(
                    (*self.participants_path, index), f"Duplicate participant {value!r}"
                )
            if value not in names:
                raise field_error(
                    (*self.participants_path, index), f"Unknown participant {value!r}"
                )
            seen.add(value)


class FixedElementsForm(FormModel):
    """Adapter declaration with shared identity admission and value validation.

    fixed_elements lists root-relative capabilities. Subclasses declare fields,
    defaults and Display metadata normally. Fresh default_input rejects duplicate
    captured names before exposing a form. FormInput/restore remain neutral;
    ordinary mutations compare with the currently installed raw list, not a
    saved capture. A Form generator accepts the same declarations via its
    fixed_elements parameter. This class stores no mutable source or identity.
    """

    fixed_elements: ClassVar[tuple[FixedElements, ...]] = ()

    @classmethod
    def default_input(cls, env: FormEnv) -> FormInput[Self]:
        """Build fresh defaults, rejecting duplicate names with their raw indices."""
        input = super().default_input(env)
        for declaration in cls.fixed_elements:
            declaration.check_declaration(cls)
            declaration.check_unique_names(input.data)
        return input

    @classmethod
    def check_input_change(
        cls, before: dict[str, JsonValue], after: dict[str, JsonValue]
    ) -> None:
        """Apply each declared identity boundary; ordinary value validity is separate."""
        super().check_input_change(before, after)
        for declaration in cls.fixed_elements:
            declaration.check_input_change(before, after)

    @model_validator(mode="after")
    def validate_fixed_elements(self) -> Self:
        """Validate participants without excluding any nonparticipating element."""
        data: dict[str, JsonValue] = self.model_dump(mode="json")
        for declaration in self.fixed_elements:
            declaration.validate_participants(data)
        return self
