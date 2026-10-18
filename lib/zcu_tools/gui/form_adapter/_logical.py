"""Active logical traversal, field-local evaluation and pure candidate preparation."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Annotated, get_args, get_origin

from pydantic import BaseModel, JsonValue, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticCustomError

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.form import (
    FormDescription,
    FormEnv,
    FormErrors,
    FormFieldDescription,
    FormInput,
    FormModel,
    FormPath,
    FormPathError,
    FormState,
    FormValue,
    FormWritePath,
    Option,
)
from zcu_tools.gui.form._describe import describe_field
from zcu_tools.gui.form._selection import Field, Level, root_level, validate_path
from zcu_tools.gui.form._sources import error_path
from zcu_tools.gui.form._structure import StructureError, alternatives

from ._seeds import (
    selected_model,
    spec_annotation,
    supported_input,
)
from .logical import (
    LogicalEdit,
    LogicalEditCandidate,
    LogicalEditError,
    LogicalFieldObservation,
    LogicalSelector,
    LogicalSource,
    LogicalSourceError,
    ModuleSelector,
)


def _marker(info: FieldInfo) -> ModuleSelector | None:
    for item in reversed(info.metadata):
        if isinstance(item, ModuleSelector):
            return item
    annotation = info.annotation
    if annotation is not None and get_origin(annotation) is Annotated:
        return _marker(FieldInfo.from_annotation(annotation))
    for member in get_args(annotation):
        if get_origin(member) is Annotated:
            found = _marker(FieldInfo.from_annotation(member))
            if found is not None:
                return found
    return None


def _contains_selector(info: FieldInfo) -> bool:
    """Whether replacing this declaration could replace a generated selector.

    Inspect metadata across all declared branches, including inactive/nullable
    models and container items. Do not parse payloads or select/evaluate branches;
    selector-bearing containers must be edited through their logical children.
    """
    pending = [info]
    seen: set[type[BaseModel]] = set()
    while pending:
        field = pending.pop()
        if _marker(field) is not None:
            return True
        for member in alternatives(field.annotation):
            if isinstance(member, type) and issubclass(member, BaseModel):
                if member not in seen:
                    seen.add(member)
                    pending.extend(member.model_fields.values())
            else:
                pending.extend(
                    FieldInfo.from_annotation(item) for item in get_args(member)
                )
    return False


def _selector(target: Field, marker: ModuleSelector, env: FormEnv) -> LogicalSelector:
    keys: tuple[str, ...] = ()
    if env.library is not None:
        library = env.library
        available = (
            library.waveform_keys()
            if marker.source_kind == "waveform"
            else library.module_keys()
        )
        read = (
            library.waveform_spec
            if marker.source_kind == "waveform"
            else library.module_spec
        )
        keys = tuple(
            key for key in available if isinstance(read(key), marker.ref_specs)
        )
    if target.name not in target.level.data:
        return LogicalSelector("missing", None, keys)
    raw = target.level.data[target.name]
    if raw is None:
        return LogicalSelector("disabled", None, keys)
    if isinstance(raw, dict):
        if raw.get("kind") == "ref":
            key = raw.get("name")
            return LogicalSelector("key", key if isinstance(key, str) else None, keys)
        if raw.get("kind") == "inline":
            return LogicalSelector("inline", None, keys)
    return LogicalSelector("unresolved", None, keys)


def _source(
    marker: ModuleSelector,
    key: str | None,
    path: FormWritePath,
    selector_path: FormWritePath,
    env: FormEnv,
) -> ConfigBase:
    try:
        if env.library is None:
            raise ValueError("Missing captured library")
        if key is None:
            raise ValueError("Missing literal library key")
        spec = (
            env.library.waveform_spec(key)
            if marker.source_kind == "waveform"
            else env.library.module_spec(key)
        )
        if not isinstance(spec, marker.ref_specs):
            raise ValueError(f"Incompatible captured family for {key!r}")
    except (KeyError, ValueError) as exc:
        raise LogicalSourceError(
            str(exc),
            path=path,
            selector_path=selector_path,
            key=key,
            source_kind=marker.source_kind,
        ) from exc
    return spec


def _shallow(
    model: type[BaseModel], data: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """Project only direct supported fields, deferring all child content."""
    result: dict[str, JsonValue] = {}
    for name, info in model.model_fields.items():
        if name not in data:
            continue
        value: JsonValue = data[name]
        if _marker(info) is not None:
            if isinstance(value, str):
                value = {"kind": "ref", "name": value}
            elif isinstance(value, dict):
                value = {"kind": "inline", "module": value}
        result[name] = value
    return result


@dataclass(frozen=True)
class _Position:
    target: Field
    path: FormWritePath
    source: LogicalSource | None = None

    @property
    def marker(self) -> ModuleSelector | None:
        return _marker(self.target.info)


def _content(
    position: _Position, env: FormEnv, request: FormWritePath, *, writing: bool
) -> tuple[Level | None, LogicalSource | None]:
    target, marker, source = position.target, position.marker, position.source
    if marker is None:
        level = target.children(env)
    else:
        selector = _selector(target, marker, env)
        if selector.mode == "key":
            if writing:
                raise LogicalEditError(
                    "Convert the referenced ancestor to inline first",
                    reason_code="ref_ancestor",
                    path=request,
                    selector_path=position.path,
                )
            spec = _source(marker, selector.key, request, position.path, env)
            model = selected_model(
                spec_annotation(target.info.annotation), spec.model_dump(mode="json")
            )
            if not issubclass(model, FormModel):
                raise TypeError("Generated Spec projection requires FormModel")
            level = root_level(
                model, _shallow(model, spec.model_dump(mode="json")), env
            )
            source = LogicalSource(
                position.path, selector.key or "", marker.source_kind, ()
            )
            return level, source
        if selector.mode != "inline":
            return None, source
        wrapper = target.children(env)
        if wrapper is None or marker.inline_field not in wrapper.model.model_fields:
            return None, source
        level = wrapper.field(marker.inline_field).children(env)
    if level is not None and source is not None:
        level = replace(level, data=_shallow(level.model, level.data))
    return level, source


def _resolve(
    input: FormInput[FormModel],
    path: FormWritePath,
    env: FormEnv,
    *,
    writing: bool = False,
) -> _Position:
    validate_path(path)
    if not path:
        raise FormPathError(path, "Root is not an editable field")
    level = root_level(input.model, input.data, env, input.selections)
    source: LogicalSource | None = None
    for index, name in enumerate(path):
        if name not in level.model.model_fields:
            raise FormPathError(path, f"Unknown logical field: {path!r}")
        if source is not None:
            source = replace(source, spec_path=(*source.spec_path, name))
        position = _Position(level.field(name), path[: index + 1], source)
        if index == len(path) - 1:
            return position
        level, source = _content(position, env, path, writing=writing)
        if level is None:
            raise FormPathError(path, f"Inactive or whole-leaf logical field: {path!r}")
    raise FormPathError(path, f"Unknown logical field: {path!r}")


def _shell(target: Field) -> tuple[type[FormModel], dict[str, JsonValue]]:
    """Preserve field metadata and effective declarations for local core evaluation."""
    choices: dict[FormPath, tuple[JsonValue, ...]] = (
        {("value",): target.choices} if target.choices is not None else {}
    )
    options: dict[FormPath, tuple[Option, ...]] = (
        {("value",): target.options} if target.options is not None else {}
    )

    class LocalField(FormModel):
        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            return choices

        @classmethod
        def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[Option, ...]]:
            return options

    model = create_model(
        "LogicalField",
        __base__=LocalField,
        value=(target.info.annotation, deepcopy(target.info)),
    )
    data = (
        {"value": target.level.data[target.name]}
        if target.name in target.level.data
        else {}
    )
    return model, data


def _metadata(position: _Position, root: type[FormModel]) -> FormFieldDescription:
    description = describe_field(
        position.target, roots=(root, position.target.level.model)
    )
    return replace(description, path=position.path, name=position.path[-1])


def _children(
    level: Level | None, prefix: FormWritePath, root: type[FormModel]
) -> tuple[FormFieldDescription, ...]:
    if level is None:
        return ()
    result: list[FormFieldDescription] = []
    for name in level.model.model_fields:
        target = level.field(name)
        if target.pinned:
            continue
        metadata = _metadata(_Position(target, (*prefix, name)), root)
        if not metadata.hidden:
            result.append(metadata)
    return tuple(result)


def _errors(errors: FormErrors | None, path: FormWritePath) -> FormErrors | None:
    if errors is None:
        return None
    messages = {
        key: value
        for key, value in errors.messages.items()
        if key[: len(path)] == path or path[: len(key)] == key
    }
    return FormErrors(messages) if messages else None


def _local_value(
    position: _Position, env: FormEnv
) -> tuple[bool, FormValue, FormErrors | None]:
    model, data = _shell(position.target)
    preview = FormState(FormInput(model, data)).preview(env)
    values = dict(preview.values)
    errors = (
        FormErrors(
            {
                (*position.path, *key[1:]): text
                for key, text in preview.errors.messages.items()
            }
        )
        if preview.errors
        else None
    )
    if ("value",) in values:
        return errors is None, values[("value",)] if errors is None else None, errors
    subtree: dict[str, FormValue] = {}
    for key, value in values.items():
        current = subtree
        for part in key[1:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = child
        if len(key) > 1:
            current[key[-1]] = value
    return bool(subtree), subtree or None, errors


def _ordinary_level(
    level: Level,
    env: FormEnv,
    source: LogicalSource | None = None,
) -> tuple[dict[str, FormValue], FormErrors | None]:
    """Ordinary supported subtree; child references stay literal keys, not cfgs."""
    result: dict[str, FormValue] = {}
    messages: dict[FormPath, str] = {}
    for name in level.model.model_fields:
        target = level.field(name)
        position = _Position(target, (name,), source)
        marker = position.marker
        if marker is not None:
            selector = _selector(target, marker, env)
            if selector.mode == "key":
                result[name] = selector.key
                continue
        nested, nested_source = _content(position, env, (name,), writing=False)
        if nested is not None:
            value, errors = _ordinary_level(nested, env, nested_source)
            result[name] = value
            if errors:
                messages.update(
                    {(name, *key): text for key, text in errors.messages.items()}
                )
            continue
        present, value, errors = _local_value(position, env)
        if present:
            result[name] = value
        if errors:
            messages.update(errors.messages)
    return result, FormErrors(messages) if messages else None


def _read_reference(
    position: _Position,
    selector: LogicalSelector,
    metadata: FormFieldDescription,
    env: FormEnv,
    root: type[FormModel],
) -> LogicalFieldObservation:
    path = position.path
    target = position.target
    readonly = position.source is not None
    input_present = not readonly and target.name in target.level.data
    raw = target.level.data.get(target.name) if input_present else None
    try:
        level, source = _content(position, env, path, writing=False)
    except LogicalSourceError as exc:
        return LogicalFieldObservation(
            FormDescription(path, replace(metadata, can_expand=False), ()),
            input_present,
            raw,
            "unavailable",
            None,
            selector,
            None,
            readonly,
            FormErrors({path: str(exc)}),
        )
    if level is None:
        raise RuntimeError("Captured Spec has no generated content model")
    value, errors = _ordinary_level(level, env, source)
    errors = (
        FormErrors({(*path, *key): text for key, text in errors.messages.items()})
        if errors
        else None
    )
    return LogicalFieldObservation(
        FormDescription(
            path, replace(metadata, can_expand=True), _children(level, path, root)
        ),
        input_present,
        raw,
        "available",
        value,
        selector,
        source,
        readonly,
        errors,
    )


def read(
    input: FormInput[FormModel], path: FormWritePath, env: FormEnv
) -> LogicalFieldObservation:
    validate_path(path)
    root = input.model
    if not path:
        level = root_level(input.model, input.data, env, input.selections)
        value, errors = _ordinary_level(level, env)
        observed = LogicalFieldObservation(
            FormDescription((), None, _children(level, (), root)),
            True,
            input.data,
            "available",
            value,
            None,
            None,
            False,
            errors,
        )
        return _with_input_errors(input, observed, env)
    position = _resolve(input, path, env)
    target, marker = position.target, position.marker
    metadata = _metadata(position, input.model)
    selector = _selector(target, marker, env) if marker is not None else None
    authored = position.source is None
    input_present = authored and target.name in target.level.data
    raw = target.level.data.get(target.name) if input_present else None
    source = position.source
    if selector is not None and selector.mode == "key":
        observed = _read_reference(position, selector, metadata, env, root)
        return _with_input_errors(input, observed, env)
    level, _ = _content(position, env, path, writing=False)
    if level is not None:
        value, errors = _ordinary_level(level, env, source)
        errors = (
            FormErrors({(*path, *key): text for key, text in errors.messages.items()})
            if errors
            else None
        )
        present = bool(value) or input_present
    else:
        present, value, errors = _local_value(position, env)
    status = (
        "missing"
        if not present and target.name not in target.level.data
        else "unavailable"
        if errors and not present
        else "available"
        if present
        else "missing"
    )
    observed = LogicalFieldObservation(
        FormDescription(path, metadata, _children(level, path, root)),
        input_present,
        raw,
        status,
        value if present else None,
        selector,
        source,
        not authored,
        errors,
    )

    return _with_input_errors(input, observed, env)


def _with_input_errors(
    input: FormInput[FormModel],
    observed: LogicalFieldObservation,
    env: FormEnv,
) -> LogicalFieldObservation:
    validation = logical_errors(input, FormState(input).preview(env).errors, env)
    relevant = _errors(validation, observed.description.path)
    if relevant is None:
        return observed
    return replace(
        observed,
        errors=FormErrors(
            {
                **(observed.errors.messages if observed.errors else {}),
                **relevant.messages,
            }
        ),
    )


def _candidate(
    input: FormInput[FormModel],
    batch: Mapping[FormWritePath, JsonValue],
    env: FormEnv,
    path: FormWritePath,
    index: int | None = None,
    origins: Mapping[FormWritePath, tuple[FormWritePath, int]] | None = None,
) -> LogicalEditCandidate:
    try:
        candidate = FormState(input).prepare_write(batch, env)
    except PydanticCustomError as exc:
        consuming = error_path(exc)
        for canonical, origin in (origins or {}).items():
            if consuming[: len(canonical)] == canonical:
                path, index = origin
                break
        raise LogicalEditError(
            str(exc), reason_code="form_source", path=path, edit_index=index
        ) from exc
    return LogicalEditCandidate(
        candidate,
        candidate.data != input.data or candidate.selections != input.selections,
    )


def edits(
    input: FormInput[FormModel], changes: Sequence[LogicalEdit], env: FormEnv
) -> LogicalEditCandidate:
    batch: dict[FormWritePath, JsonValue] = {}
    previous: list[FormWritePath] = []
    origins: dict[FormWritePath, tuple[FormWritePath, int]] = {}
    for index, change in enumerate(changes):
        for path in previous:
            if (
                change.path[: len(path)] == path
                or path[: len(change.path)] == change.path
            ):
                raise LogicalEditError(
                    "Overlapping logical edits",
                    reason_code="form_structure",
                    path=change.path,
                    edit_index=index,
                )
        previous.append(change.path)
        try:
            position = _resolve(input, change.path, env, writing=True)
        except FormPathError as exc:
            raise LogicalEditError(
                str(exc), reason_code="form_path", path=change.path, edit_index=index
            ) from exc
        except LogicalEditError as exc:
            exc.edit_index = index
            raise
        if _contains_selector(position.target.info):
            raise LogicalEditError(
                "Edit selector-bearing containers through logical children; "
                "use dedicated operations for module selection",
                reason_code="module_selector",
                path=change.path,
                edit_index=index,
                selector_path=change.path if position.marker is not None else None,
            )
        batch[position.target.path] = change.value
        origins[position.target.path] = (change.path, index)
    try:
        return _candidate(
            input,
            batch,
            env,
            changes[-1].path if changes else (),
            len(changes) - 1 if changes else None,
            origins,
        )
    except StructureError as exc:
        path, index = origins[exc.path]
        raise LogicalEditError(
            str(exc),
            reason_code="form_structure",
            path=path,
            edit_index=index,
        ) from exc


def select(
    input: FormInput[FormModel],
    path: FormWritePath,
    env: FormEnv,
    *,
    action: str,
    key: str | None = None,
) -> LogicalEditCandidate:
    try:
        position = _resolve(input, path, env, writing=True)
    except FormPathError as exc:
        raise LogicalEditError(str(exc), reason_code="form_path", path=path) from exc
    marker = position.marker
    if marker is None:
        raise LogicalEditError(
            "Not a generated module selector", reason_code="module_selector", path=path
        )
    description = _metadata(position, input.model)
    selector = _selector(position.target, marker, env)
    if action == "copy" and selector.mode == "inline":
        return LogicalEditCandidate(
            FormInput(input.model, input.data, selections=input.selections), False
        )
    if description.fixed or (action == "library" and description.selection_fixed):
        raise LogicalEditError(
            "Declaration forbids this selection",
            reason_code="module_selector",
            path=path,
            selector_path=path,
        )
    value: JsonValue
    if action == "library":
        if key not in selector.available_keys:
            raise LogicalEditError(
                "Key is not available for this family",
                reason_code="library_key",
                path=path,
                selector_path=path,
                available_keys=selector.available_keys,
            )
        value = {"kind": "ref", "name": key}
    elif action == "disable":
        if not description.nullable:
            raise LogicalEditError(
                "Selector is not nullable",
                reason_code="module_selector",
                path=path,
                selector_path=path,
            )
        value = None
    else:
        if selector.mode != "key":
            raise LogicalEditError(
                "Copy requires a resolved reference",
                reason_code="module_state",
                path=path,
                selector_path=path,
            )
        spec = _source(marker, selector.key, path, path, env)
        model = selected_model(
            spec_annotation(position.target.info.annotation),
            spec.model_dump(mode="json"),
        )
        value = {
            "kind": "inline",
            marker.inline_field: supported_input(
                model, spec.model_dump(mode="json", exclude_none=True)
            ),
        }
    return _candidate(input, {position.target.path: value}, env, path)


def logical_errors(
    input: FormInput[FormModel], errors: FormErrors | None, env: FormEnv
) -> FormErrors | None:
    if errors is None:
        return None
    result: dict[FormPath, str] = {}
    for canonical, message in errors.messages.items():
        level = root_level(input.model, input.data, env, input.selections)
        logical: list[str | int] = []
        index = 0
        while index < len(canonical):
            name = canonical[index]
            if not isinstance(name, str) or name not in level.model.model_fields:
                logical.extend(canonical[index:])
                break
            logical.append(name)
            target = level.field(name)
            marker = _marker(target.info)
            index += 1
            if marker is not None:
                if index < len(canonical) and canonical[index] in ("kind", "name"):
                    break
                if index < len(canonical) and canonical[index] == marker.inline_field:
                    index += 1
                position = _Position(
                    target, tuple(part for part in logical if isinstance(part, str))
                )
                raw = target.level.data.get(name)
                child, _ = (
                    _content(position, env, position.path, writing=False)
                    if isinstance(raw, dict) and raw.get("kind") == "inline"
                    else (None, None)
                )
            else:
                child = target.children(env)
            if child is None:
                logical.extend(canonical[index:])
                break
            level = child
        result[tuple(logical)] = message
    return FormErrors(result) if result else None
