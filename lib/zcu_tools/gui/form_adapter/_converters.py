"""Shared name-and-editable-type raw inheritance for generated selectors."""

from collections.abc import Mapping
from functools import partial
from typing import get_args, get_origin

from pydantic import BaseModel, JsonValue

from zcu_tools.cfg_model import (
    ModuleSpec,
    WaveformSpec,
)
from zcu_tools.gui.form import (
    FormCommandContext,
    FormCommandSnapshot,
    FormConversionError,
    FormConversionResult,
    FormEnv,
    FormModel,
    InputConverter,
    static_defaults,
)
from zcu_tools.gui.form.commands import ReadonlyFormInput

from ._seeds import inline_wrapper, selected_model, spec_annotation, supported_input
from ._types import members
from .input_types import KeyRef

_PERSISTENT = tuple(
    item
    for alias in (ModuleSpec, WaveformSpec)
    for item in members(alias)
    if isinstance(item, type) and issubclass(item, BaseModel)
)


def _persistent(model: type[BaseModel]) -> type[BaseModel]:
    return next(item for item in _PERSISTENT if issubclass(model, item))


def _source(
    old: FormCommandSnapshot,
    models: tuple[type[BaseModel], ...],
    env: FormEnv,
    *,
    waveform: bool,
) -> tuple[type[BaseModel], dict[str, JsonValue]] | None:
    raw = old.copy_input()
    if not isinstance(raw, dict):
        return None
    if raw.get("kind") == "ref":
        key = raw.get("name")
        if env.library is None or not isinstance(key, str):
            raise FormConversionError(
                "Missing captured library/key for inline conversion"
            )
        try:
            spec = (
                env.library.waveform_spec(key)
                if waveform
                else env.library.module_spec(key)
            )
        except KeyError as exc:
            raise FormConversionError(f"Missing captured source {key!r}") from exc
        model = next((model for model in models if issubclass(model, type(spec))), None)
        if model is None:
            # A ref family can be narrower than inline families, but never outside them.
            raise FormConversionError(f"Incompatible library family for {key!r}")
        return model, supported_input(
            model, spec.model_dump(mode="json", exclude_none=True)
        )
    module = raw.get("module")
    if raw.get("kind") == "inline" and isinstance(module, dict):
        model = (
            selected_model(
                spec_annotation(old.structure.model_fields["module"].annotation), module
            )
            if old.structure is not None
            else selected_model(models[0], module)
        )
        return model, module
    return None


def _editable_type(annotation: object) -> object:
    """Compare generated declarations without their per-occurrence class identity."""
    alternatives = members(annotation)
    if alternatives != (annotation,):
        if len(alternatives) == 1:
            return _editable_type(alternatives[0])
        return frozenset(_editable_type(item) for item in alternatives)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        # Generated Spec subclasses share a nominal persistent base. Pure tagged
        # wrappers have no such base, so their declared fields define their type.
        persistent = next(
            (
                base
                for base in annotation.__mro__
                if issubclass(base, BaseModel) and not issubclass(base, FormModel)
            ),
            BaseModel,
        )
        if persistent is not BaseModel:
            return persistent
        return tuple(
            (name, _editable_type(info.annotation))
            for name, info in annotation.model_fields.items()
        )
    origin = get_origin(annotation)
    if origin is not None:
        return origin, tuple(_editable_type(arg) for arg in get_args(annotation))
    return annotation


# Traverse the input tree through annotations so nested defaults never fill
# authored missing leaves, while newly declared fields still get initial input.
def _inherit_value(
    source: object, target: object, raw: JsonValue, env: FormEnv
) -> JsonValue:
    if isinstance(raw, list) and get_origin(target) is list:
        return [
            _inherit_value(get_args(source)[0], get_args(target)[0], item, env)
            for item in raw
        ]
    if not isinstance(raw, dict) or "expr" in raw:
        return raw
    if not any(
        isinstance(item, type) and issubclass(item, BaseModel)
        for item in members(target)
    ):
        return raw
    try:
        old_model = selected_model(source, raw)
        new_model = selected_model(target, raw)
    except ValueError:
        # Invalid discriminators are authored input, not permission to repair it.
        return raw
    return _inherit(old_model, new_model, raw, env)


def _inherit(
    source: type[BaseModel],
    target: type[BaseModel],
    raw: Mapping[str, JsonValue],
    env: FormEnv,
) -> dict[str, JsonValue]:
    result = (
        target.defaults(env)
        if issubclass(target, FormModel)
        else static_defaults(target)
    )
    for name, info in target.model_fields.items():
        old_info = source.model_fields.get(name)
        if old_info is None or _editable_type(old_info.annotation) != _editable_type(
            info.annotation
        ):
            continue
        if name not in raw:
            result.pop(name, None)
        else:
            result[name] = _inherit_value(
                old_info.annotation, info.annotation, raw[name], env
            )
    return result


def _convert(
    old: FormCommandSnapshot,
    selection: ReadonlyFormInput,
    _context: FormCommandContext,
    *,
    annotation: object,
    env: FormEnv,
) -> FormConversionResult:
    if selection is None:
        return FormConversionResult(None, None)
    if not isinstance(selection, Mapping):
        raise FormConversionError("Module selection must be a tagged object")
    if selection.get("kind") == "ref":
        ref = next(
            item
            for item in members(annotation)
            if isinstance(item, type) and issubclass(item, KeyRef)
        )
        key = selection.get("name")
        if not isinstance(key, str):
            raise FormConversionError("Module reference needs a literal key")
        return FormConversionResult(ref, {"kind": "ref", "name": key})
    wrapper = inline_wrapper(annotation)
    requested = selection.get("module")
    if wrapper is None or not isinstance(requested, Mapping):
        raise FormConversionError("Inline selection needs a declared module kind")
    tags = {key: value for key, value in requested.items() if isinstance(value, str)}
    target = selected_model(spec_annotation(annotation), tags)
    models = tuple(
        item
        for item in members(spec_annotation(annotation))
        if isinstance(item, type) and issubclass(item, BaseModel)
    )
    source = _source(
        old, models, env, waveform=_persistent(target) in members(WaveformSpec)
    )
    module = (
        _inherit(source[0], target, source[1], env)
        if source is not None
        else target.defaults(env)
        if issubclass(target, FormModel)
        else static_defaults(target)
    )
    return FormConversionResult(wrapper, {"kind": "inline", "module": module})


def module_converter(annotation: object, env: FormEnv) -> InputConverter:
    """Bind one selector's converter to the caller's fixed capture.

    annotation is the generated ref/inline field type. No library is read here;
    the returned converter reads it only during a guarded selection write.
    Each target field inherits raw input (including missing or invalid drafts)
    only from an identically named field with the same editable Form type.
    Generated Spec classes compare by their persistent base, not per-occurrence
    identity; nested content follows the same rule at each level. Other fields
    use target initial input. No expression is evaluated or asset read.
    """
    return InputConverter(partial(_convert, annotation=annotation, env=env))
