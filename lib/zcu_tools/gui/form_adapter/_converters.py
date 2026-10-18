"""Explicit persistent-type inheritance for every generated module selector."""

from collections.abc import Mapping
from functools import partial

from pydantic import BaseModel, JsonValue

from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    GaussWaveformCfg,
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
# These are semantic policies, not a same-name/type compatibility heuristic.
_LENGTH_PAIRS = {
    (ConstWaveformCfg, CosineWaveformCfg),
    (ConstWaveformCfg, GaussWaveformCfg),
    (ConstWaveformCfg, DragWaveformCfg),
    (CosineWaveformCfg, ConstWaveformCfg),
    (CosineWaveformCfg, GaussWaveformCfg),
    (CosineWaveformCfg, DragWaveformCfg),
    (GaussWaveformCfg, ConstWaveformCfg),
    (GaussWaveformCfg, CosineWaveformCfg),
    (GaussWaveformCfg, DragWaveformCfg),
    (DragWaveformCfg, ConstWaveformCfg),
    (DragWaveformCfg, CosineWaveformCfg),
    (DragWaveformCfg, GaussWaveformCfg),
}
_SIGMA_PAIRS = {
    (GaussWaveformCfg, DragWaveformCfg),
    (DragWaveformCfg, GaussWaveformCfg),
}


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
        return type(spec), supported_input(
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
        return _persistent(model), module
    return None


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
    if source is not None and source[0] is _persistent(target):
        module = source[1]
    else:
        module = (
            target.defaults(env)
            if issubclass(target, FormModel)
            else static_defaults(target)
        )
        if source is not None:
            pair = (source[0], _persistent(target))
            for name in ("length", "sigma"):
                permitted = pair in (
                    _LENGTH_PAIRS if name == "length" else _SIGMA_PAIRS
                )
                if permitted and name in source[1]:
                    module[name] = source[1][name]
    return FormConversionResult(wrapper, {"kind": "inline", "module": module})


def module_converter(annotation: object, env: FormEnv) -> InputConverter:
    """Bind one selector's converter to the caller's fixed capture.

    annotation is the generated ref/inline field type. No library is read here;
    the returned converter reads it only during a guarded selection write.
    It retains raw same-type drafts, uses the explicit waveform pair allowlist,
    and otherwise asks the target model for fresh input. No asset is read.
    """
    return InputConverter(partial(_convert, annotation=annotation, env=env))
