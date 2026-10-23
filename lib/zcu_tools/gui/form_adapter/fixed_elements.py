"""Declaration-owned fixed identities and participant value validation.

Core only calls the neutral input-change contract. This adapter capability
assigns domain meaning to named list elements; widgets and transports do not.
"""

from collections.abc import Mapping
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

from ._seeds import selected_model
from ._types import members


def _field(model: type[BaseModel], path: FormWritePath) -> FieldInfo:
    fields = [FieldInfo.from_annotation(model)]
    for part in path:
        children: list[FieldInfo] = []
        for info in fields:
            for annotation in members(info.annotation):
                if (
                    isinstance(part, str)
                    and isinstance(annotation, type)
                    and issubclass(annotation, BaseModel)
                ):
                    if (field := annotation.model_fields.get(part)) is not None:
                        children.append(field)
                elif type(part) is int and get_origin(annotation) is list:
                    children.append(FieldInfo.from_annotation(get_args(annotation)[0]))
        if not children:
            raise ValueError(f"Unknown fixed-elements declaration path: {path!r}")
        fields = children
    if any(info.annotation != fields[0].annotation for info in fields[1:]):
        raise ValueError(f"Ambiguous fixed-elements declaration path: {path!r}")
    return fields[0]


def _at(data: JsonValue, path: FormWritePath) -> JsonValue:
    for part in path:
        if isinstance(data, dict) and isinstance(part, str):
            data = data.get(part)
        elif isinstance(data, list) and isinstance(part, int) and 0 <= part < len(data):
            data = data[part]
        else:
            return None
    return data


def _active_value(
    model: type[BaseModel], data: JsonValue, path: FormWritePath
) -> tuple[bool, JsonValue]:
    """Distinguish an inactive schema branch from an invalid active list draft."""
    annotation: object = model
    value: JsonValue = data
    for part in path:
        current: JsonValue = value
        if current is None and type(None) in members(annotation):
            return False, None
        if isinstance(current, dict) and isinstance(part, str):
            try:
                branch = selected_model(annotation, current)
            except ValueError:
                return True, None
            field = branch.model_fields.get(part)
            if field is None:
                return False, None
            annotation = field.annotation
            value = current.get(part)
        elif isinstance(current, list) and type(part) is int:
            if not 0 <= part < len(current) or get_origin(annotation) is not list:
                return True, None
            annotation = get_args(annotation)[0]
            value = current[part]
        else:
            return True, None
    return True, value


@dataclass(frozen=True)
class FixedElements:
    """Declare a named list/participants policy or fixed editable slots.

    Paths are nonempty root-relative model fields, optionally through indexes
    and active tagged branches. With name_field and participants_path, ordinary
    mutation preserves ordered literal names; participants validate separately.
    Display(element_label=name_field) and Display(readonly=True) supply the
    named presentation, not this policy.

    Omit both optional arguments for fixed slots: only the active list length
    is preserved. Every value, including label, remains editable; this does not
    prevent swapping values or assert component identity. Legal branch switches
    are allowed, and a newly active list establishes its own length boundary.
    Restore/Reset establish new boundaries in either mode. No sources, saved
    identities or secondary ordering are stored.
    Participants is a nonempty, unique list[int].
    """

    list_path: FormWritePath
    name_field: str | None = None
    participants_path: FormWritePath | None = None

    def __post_init__(self) -> None:
        """Reject empty paths/names and incomplete named policies with ValueError."""
        if (
            not self.list_path
            or ((self.name_field is None) != (self.participants_path is None))
            or self.name_field == ""
            or self.participants_path == ()
        ):
            raise ValueError(
                "Fixed elements require a list path and either both name/participants or neither"
            )

    def check_declaration(self, model: type[BaseModel]) -> None:
        """Require list[Model], str names and list[int]."""
        info = _field(model, self.list_path)
        if get_origin(info.annotation) is not list:
            raise ValueError(f"{self.list_path!r}: fixed elements require list[Model]")
        element = get_args(info.annotation)[0]
        if not isinstance(element, type) or not issubclass(element, BaseModel):
            raise ValueError(f"{self.list_path!r}: fixed elements require list[Model]")
        if self.name_field is None or self.participants_path is None:
            return
        name = element.model_fields.get(self.name_field)
        if name is None or members(name.annotation) != (str,):
            raise ValueError(
                f"{self.list_path!r}: {self.name_field!r} must name a str field"
            )
        participants = _field(model, self.participants_path).annotation
        if get_origin(participants) is not list or members(
            get_args(participants)[0]
        ) != (int,):
            raise ValueError(
                f"{self.participants_path!r}: participants require list[int]"
            )

    def names(self, data: dict[str, JsonValue]) -> tuple[JsonValue, ...] | None:
        """Read ordered literal names; slots and missing/non-list return None."""
        if self.name_field is None:
            return None
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
        if self.name_field is None:
            return None
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
        """Reject duplicate capture names; fixed slots have no named identity."""
        if self.name_field is None:
            return
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
        self,
        before: dict[str, JsonValue],
        after: dict[str, JsonValue],
        *,
        model: type[BaseModel] | None = None,
    ) -> None:
        """Refuse identity changes or active slot-count changes atomically.

        Fixed slots require the declaring model to distinguish inactive branches
        from malformed active lists. Named mode compares ordered raw identities.
        """
        if self.name_field is None:
            if model is None:
                raise ValueError("Fixed slots require the declaring model")
            old_active, old_value = _active_value(model, before, self.list_path)
            new_active, new_value = _active_value(model, after, self.list_path)
            if not old_active or not new_active:
                return
            old_size = len(old_value) if isinstance(old_value, list) else None
            new_size = len(new_value) if isinstance(new_value, list) else None
            if old_size != new_size:
                raise FormInputRejected(
                    self.list_path,
                    f"Fixed slot count changed at {self.list_path!r}: {old_size!r} -> {new_size!r}",
                )
            return
        old, new = self.names(before), self.names(after)
        if old != new:
            raise FormInputRejected(
                self.list_path,
                f"Fixed element identity changed at {self.list_path!r}: {old!r} -> {new!r}",
            )

    def validate_participants(self, data: dict[str, JsonValue]) -> None:
        """Raise a directed ordinary Pydantic error for illegal participant values.

        Report empty, duplicate and out-of-range original indices at their input
        position. Never reorder or filter either list. Pydantic handles non-list
        shapes and non-integer entries before this successful-model check.
        """
        if self.participants_path is None:
            return
        names = self.names(data)
        values = _at(data, self.participants_path)
        if names is None or not isinstance(values, list):
            return
        if not values:
            raise field_error(self.participants_path, "Participants must not be empty")
        seen: set[int] = set()
        for index, value in enumerate(values):
            if type(value) is not int:
                continue
            if value in seen:
                raise field_error(
                    (*self.participants_path, index), f"Duplicate participant {value!r}"
                )
            if not 0 <= value < len(names):
                raise field_error(
                    (*self.participants_path, index),
                    f"Participant index {value} outside tones [0, {len(names)})",
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
    def default_input(
        cls, env: FormEnv, *, defaults: Mapping[str, JsonValue] | None = None
    ) -> FormInput[Self]:
        """Build fresh defaults, rejecting duplicate names with their raw indices.

        defaults optionally supplies already-built raw defaults, as in FormModel.
        """
        input = super().default_input(env, defaults=defaults)
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
            declaration.check_input_change(before, after, model=cls)

    @model_validator(mode="before")
    @classmethod
    def validate_participant_types(cls, data: JsonValue) -> JsonValue:
        """Reject non-integer raw participant entries, including booleans."""
        if isinstance(data, dict):
            for declaration in cls.fixed_elements:
                path = declaration.participants_path
                values = _at(data, path) if path is not None else None
                if isinstance(values, list) and path is not None:
                    for index, value in enumerate(values):
                        if type(value) is not int:
                            raise field_error(
                                (*path, index),
                                "Participant index must be an integer, not bool or text",
                            )
        return data

    @model_validator(mode="after")
    def validate_fixed_elements(self) -> Self:
        """Validate original indices without trimming or reordering the full list."""
        data: dict[str, JsonValue] = self.model_dump(mode="json")
        for declaration in self.fixed_elements:
            declaration.validate_participants(data)
        return self
