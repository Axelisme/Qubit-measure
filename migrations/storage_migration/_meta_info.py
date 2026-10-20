"""Literal extension storage shared by scalar and blocked-group meta migration."""

from pydantic import JsonValue, TypeAdapter
from zcu_tools.resources.document_store import YamlMap, YamlValue
from zcu_tools.resources.entry import EditView, document_value, snapshot

_YAML = TypeAdapter(YamlValue)


def preserve_meta_field(
    view: EditView, key: str, value: JsonValue, *, legacy: bool
) -> tuple[str, ...]:
    """Store one unconverted source field and return its literal document keys.

    view is the caller's context draft. key/value are the actual source JSON,
    not resolved unit evidence. legacy selects legacy_meta_info for blocked or
    superseded fields. Unknown reserved namespace keys use that container too,
    nested under their own literal names. Existing unrelated ext data survives.
    Owner validation and incompatible extension shapes raise ValueError.
    """
    current = document_value(snapshot(view), "general.ext")
    if not isinstance(current, dict):
        raise ValueError("general.ext must be a mapping")
    ext: YamlMap = dict(current)
    accepted = _YAML.validate_python(value, strict=True)
    if legacy or key in ("legacy_meta_info", "legacy_module_cfg"):
        previous = ext.get("legacy_meta_info", {})
        if not isinstance(previous, dict):
            raise ValueError("general.ext.legacy_meta_info must be a mapping")
        ext["legacy_meta_info"] = {**previous, key: accepted}
        keys = ("general", "ext", "legacy_meta_info", key)
    else:
        ext[key] = accepted
        keys = ("general", "ext", key)
    view.set("general.ext", ext)
    return keys
