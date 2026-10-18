"""Restore authored drafts by current names and shapes, without value validation.

Model fields and sequence elements alternate recursively in a JSON input tree.
"""

from dataclasses import dataclass, field
from typing import get_args, get_origin

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from ._json import copy_json, copy_object
from ._selection import literal_values
from ._structure import alternatives, discriminator, model_types, selected, unwrap
from .defaults import static_defaults
from .model import FormEnv, FormModel, FormPath, FormWritePath


def _accepts_selection(info: FieldInfo, value: JsonValue) -> bool:
    """Check only annotation membership, not options hooks or ordinary values."""
    if model_types(info.annotation):
        return _accepts_model_selection(info, value)
    literals = literal_values(info.annotation)
    if literals is not None:
        return any(type(value) is type(item) and value == item for item in literals)
    try:
        TypeAdapter(info.annotation).validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


def _accepts_model_selection(info: FieldInfo, value: JsonValue) -> bool:
    if value is None:
        return type(None) in alternatives(info.annotation)
    if not isinstance(value, dict):
        return False
    branch = selected(info, value)
    if branch is None:
        return discriminator(info) is None
    return all(
        name in branch.model_fields
        and _accepts_selection(branch.model_fields[name], child)
        for name, child in value.items()
    )


@dataclass
class _Restorer:
    env: FormEnv
    selections: dict[FormWritePath, JsonValue]
    retained: dict[FormWritePath, JsonValue] = field(default_factory=dict)
    dropped: list[FormPath] = field(default_factory=list)

    def defaults(self, model: type[BaseModel]) -> dict[str, JsonValue]:
        return (
            model.defaults(self.env)
            if issubclass(model, FormModel)
            else static_defaults(model)
        )

    def model(
        self,
        model: type[BaseModel],
        current: dict[str, JsonValue],
        saved: dict[str, JsonValue],
        prefix: FormPath,
    ) -> dict[str, JsonValue]:
        result = copy_object(current)
        for name, value in saved.items():
            path = (*prefix, name)
            info = model.model_fields.get(name)
            if info is None:
                self.dropped.append(path)
                continue
            restored = self.value(info, result.get(name), value, path)
            if path not in self.dropped:
                result[name] = restored
        # An override may exist even when the authored field is absent.
        for name, info in model.model_fields.items():
            path = (*prefix, name)
            if name not in saved and path in self.selections:
                self.selection(info, path)
        return result

    def selection(self, info: FieldInfo, path: FormPath) -> bool:
        if path not in self.selections:
            return True
        value = self.selections[path]
        if not _accepts_selection(info, value):
            self.dropped.append(path)
            return False
        self.retained[path] = copy_json(value)
        return True

    def value(
        self, info: FieldInfo, current: JsonValue, saved: JsonValue, path: FormPath
    ) -> JsonValue:
        if not self.selection(info, path):
            return current
        members = alternatives(info.annotation)
        models = model_types(info.annotation)
        tag = discriminator(info)
        if (
            len(tuple(member for member in members if member is not type(None))) > 1
            and tag is None
        ):
            # A union without a discriminator is opaque: never infer a branch.
            return copy_json(saved)
        if saved is None and type(None) in members:
            return None
        if models:
            return self.model_value(info, current, saved, path)
        if get_origin(unwrap(info.annotation)) in (list, tuple):
            return self.sequence(info, current, saved, path)
        return self.leaf(info, current, saved, path)

    def model_value(
        self, info: FieldInfo, current: JsonValue, saved: JsonValue, path: FormPath
    ) -> JsonValue:
        tag = discriminator(info)
        child = selected(info, saved)
        if not isinstance(saved, dict) or child is None:
            self.dropped.append(path)
            return current
        if tag is not None and (
            tag not in saved
            or not _accepts_selection(child.model_fields[tag], saved[tag])
        ):
            self.dropped.append(path)
            return current
        prior_branch = selected(info, current)
        seed = (
            current
            if prior_branch is child and isinstance(current, dict)
            else self.defaults(child)
        )
        return self.model(child, seed, saved, path)

    def sequence(
        self, info: FieldInfo, current: JsonValue, saved: JsonValue, path: FormPath
    ) -> JsonValue:
        annotation = unwrap(info.annotation)
        if not isinstance(saved, list):
            self.dropped.append(path)
            return current
        args = get_args(annotation)
        if not args:
            return copy_json(saved)
        result: list[JsonValue] = []
        for index, item in enumerate(saved):
            item_type = (
                args[min(index, len(args) - 1)]
                if get_origin(annotation) is tuple and args[-1] is not Ellipsis
                else args[0]
            )
            item_info = FieldInfo.from_annotation(item_type)
            item_models = model_types(item_type)
            seed: JsonValue = (
                self.defaults(item_models[0]) if len(item_models) == 1 else None
            )
            if isinstance(current, list) and index < len(current):
                prior = current[index]
                seed = (
                    {**seed, **prior}
                    if isinstance(seed, dict) and isinstance(prior, dict)
                    else prior
                )
            result.append(self.value(item_info, seed, item, (*path, index)))
        return result

    def leaf(
        self, info: FieldInfo, current: JsonValue, saved: JsonValue, path: FormPath
    ) -> JsonValue:
        literals = literal_values(info.annotation)
        if literals is not None and not _accepts_selection(info, saved):
            self.dropped.append(path)
            return current
        if (
            isinstance(saved, dict)
            and get_origin(unwrap(info.annotation)) is not dict
            and set(saved) not in ({"expr"}, {"__complex__"})
        ):
            self.dropped.append(path)
            return current
        return copy_json(saved)


def restore(
    model: type[FormModel],
    defaults: dict[str, JsonValue],
    data: dict[str, JsonValue],
    selections: dict[FormWritePath, JsonValue],
    env: FormEnv,
) -> tuple[dict[str, JsonValue], dict[FormWritePath, JsonValue], tuple[FormPath, ...]]:
    """Return restored raw/overrides and dropped paths; no options/converter runs.

    defaults is the host's current captured input. Only defaults for newly
    selected model branches/list elements may be requested from their model.
    Provider failures propagate before the caller installs any candidate.
    """
    restorer = _Restorer(env, selections)
    restored = restorer.model(model, defaults, data, ())
    restorer.dropped.extend(
        path for path in selections if path not in restorer.retained
    )
    retained = {
        path: value
        for path, value in restorer.retained.items()
        if not any(path[: len(parent)] == parent for parent in restorer.dropped)
    }
    return restored, retained, tuple(dict.fromkeys(restorer.dropped))
