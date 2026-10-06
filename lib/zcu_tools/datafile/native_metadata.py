"""Native HDF5 metadata encoding and historical JSON projection."""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import h5py as h5
from pydantic import JsonValue, TypeAdapter, ValidationError

from .native_models import CfgSnapshot, JsonObject, RunMetadata
from .native_nodes import known_group, known_node

_JSON = TypeAdapter(JsonObject)
_METADATA = TypeAdapter(RunMetadata)
_OPTIONAL_EVIDENCE = ("git_commit", "qick_version", "soc_fingerprint", "hostname")


def text_attr(node: h5.Group | h5.Dataset, name: str) -> str:
    """Read a required UTF-8 scalar attr; malformed/missing attrs raise ValueError."""
    if name not in node.attrs:
        raise ValueError(f"{node.name}: missing {name} attr")
    value = node.attrs[name]
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{node.name}: {name} is not UTF-8") from error
    if not isinstance(value, str):
        raise ValueError(f"{node.name}: {name} must be a UTF-8 string")
    return value


def json_object(text: str, location: str) -> JsonObject:
    """Decode a complete JSON object, preserving unknown keys or raising ValueError."""
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
        return _JSON.validate_python(value, strict=True)
    except ValueError as error:
        raise ValueError(f"{location}: invalid JSON object: {error}") from error


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def read_json(node: h5.Group, name: str) -> JsonObject:
    """Read a required scalar UTF-8 JSON dataset under node, without pruning keys."""
    location = f"{str(node.name).rstrip('/')}/{name}"
    dataset = known_node(node, name)
    if dataset is None:
        raise ValueError(f"{location}: missing dataset")
    if not isinstance(dataset, h5.Dataset) or dataset.shape != ():
        raise ValueError(f"{location}: expected scalar JSON dataset")
    dtype = h5.check_string_dtype(dataset.dtype)
    if dtype is None or dtype.encoding != "utf-8":
        raise ValueError(f"{location}: expected UTF-8 JSON dataset")
    text = dataset.asstr()[()]
    if not isinstance(text, str):
        raise ValueError(f"{location}: expected scalar JSON text")
    return json_object(text, location)


def _same_json_value(old: JsonValue, new: JsonValue) -> bool:
    # JSON numbers compare by value, but true must not compare equal to 1.
    if isinstance(old, bool) or isinstance(new, bool):
        return type(old) is type(new) and old == new
    if isinstance(old, dict) and isinstance(new, dict):
        return old.keys() == new.keys() and all(
            _same_json_value(value, new[key]) for key, value in old.items()
        )
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(
            _same_json_value(a, b) for a, b in zip(old, new, strict=True)
        )
    return old == new


def write_json(node: h5.Group, name: str, value: JsonObject) -> h5.Dataset:
    """Write full JSON values while retaining an existing scalar's identity.

    Unchanged JSON retains raw text, including compact number/Unicode spelling.
    Changed fixed UTF-8 strings use compact JSON and must fit their encoded byte
    capacity, otherwise raise a located ValueError before assignment. New datasets
    use vlen UTF-8.
    """
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    dataset = known_node(node, name)
    if dataset is not None:
        old = read_json(node, name)
        if not isinstance(dataset, h5.Dataset):
            raise ValueError(f"{node.name}/{name}: expected scalar JSON dataset")
        if _same_json_value(old, value):
            return dataset
        dtype = h5.check_string_dtype(dataset.dtype)
        if dtype is not None and dtype.length is not None:
            # Fixed storage must not reject a fitting edit for optional whitespace.
            text = json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            if len(text.encode("utf-8")) > dtype.length:
                raise ValueError(
                    f"{str(node.name).rstrip('/')}/{name}: JSON exceeds fixed UTF-8 "
                    f"byte capacity {dtype.length}"
                )
        dataset[()] = text
        return dataset
    return node.create_dataset(name, data=text, dtype=h5.string_dtype("utf-8"))


def validate_metadata(metadata: RunMetadata, cfg: CfgSnapshot) -> None:
    """Validate historical metadata, UTC times and cfg JSON before creating a file."""
    # JSON validation rechecks dataclass fields rather than trusting instance types.
    try:
        checked = _METADATA.validate_json(
            json.dumps(_METADATA.dump_python(metadata, mode="json"), allow_nan=False),
            strict=True,
        )
        _JSON.validate_json(json.dumps(cfg.values, allow_nan=False), strict=True)
    except (ValidationError, ValueError) as error:
        raise ValueError(f"/context or /cfg: {error}") from error
    if re.fullmatch(r"[0-9]+\.[0-9]+", cfg.schema_version) is None:
        raise ValueError("/cfg: cfg_schema_version must be major.minor")
    if not cfg.cfg_type:
        raise ValueError("/cfg: cfg_type must not be empty")
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[a-z0-9]{6}", checked.run_id) is None:
        raise ValueError(
            "/: run_id must be a UTC timestamp plus six-character identity"
        )
    if not checked.experiment:
        raise ValueError("/: experiment must not be empty")
    _validate_utc(checked.started_at, "/: started_at")
    if checked.finished_at is not None:
        _validate_utc(checked.finished_at, "/: finished_at")
    for key, parameter in checked.snapshot.params.items():
        _validate_utc(parameter.source.at, f"/context/params/{key}/source: at")
    for name in _OPTIONAL_EVIDENCE:
        value = getattr(checked.provenance, name)
        if value == "":
            raise ValueError(f"/provenance: {name} must be nonempty or None")
    if checked.labber_path == "":
        raise ValueError("/: labber_path must be nonempty or None")


def _validate_utc(value: str, location: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{location}: expected UTC ISO timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{location}: expected UTC ISO timestamp")


def _merge_context(old: JsonObject, new: JsonObject) -> JsonObject:
    merged = dict(old)
    merged.update(new)
    old_params, new_params = old.get("params"), new.get("params")
    if isinstance(new_params, dict):
        params: JsonObject = {}
        for key, value in new_params.items():
            prior = old_params.get(key) if isinstance(old_params, dict) else None
            if isinstance(prior, dict) and isinstance(value, dict):
                param = dict(prior)
                param.update(value)
                old_source, new_source = prior.get("source"), value.get("source")
                if isinstance(old_source, dict) and isinstance(new_source, dict):
                    source = dict(old_source)
                    source.update(new_source)
                    old_clone = old_source.get("cloned_from")
                    new_clone = new_source.get("cloned_from")
                    if isinstance(old_clone, dict) and isinstance(new_clone, dict):
                        source["cloned_from"] = {**old_clone, **new_clone}
                    param["source"] = source
                params[key] = param
            else:
                params[key] = value
        merged["params"] = params
    return merged


def write_metadata(file: h5.File, metadata: RunMetadata, cfg: CfgSnapshot) -> None:
    """Update known run attrs and JSON, preserving context's unknown fixed siblings.

    Caller validates metadata first. cfg.values and dynamic roles/params are
    complete replacements; retained params preserve unknown fixed-field keys.
    Existing scalar datasets retain their HDF5 object identity.
    """
    raw = json_object(_METADATA.dump_json(metadata).decode(), "/")
    for name in (
        "run_id",
        "experiment",
        "started_at",
        "finished_at",
        "completion",
        "labber_path",
    ):
        file.attrs[name] = raw[name] if raw[name] is not None else ""
    cfg_dataset = write_json(file, "cfg", cfg.values)
    cfg_dataset.attrs["cfg_type"] = cfg.cfg_type
    cfg_dataset.attrs["cfg_schema_version"] = cfg.schema_version
    snapshot = raw["snapshot"]
    if not isinstance(snapshot, dict):
        raise ValueError("/context: snapshot must be a JSON object")
    if "context" in file:
        snapshot = _merge_context(read_json(file, "context"), snapshot)
    write_json(file, "context", snapshot)
    provenance = raw["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("/provenance: expected JSON object")
    group = known_group(file, "provenance")
    group.attrs["software_versions"] = json.dumps(provenance["software_versions"])
    for name in _OPTIONAL_EVIDENCE:
        group.attrs[name] = provenance[name] if provenance[name] is not None else ""
    group.attrs["git_dirty"] = {True: "true", False: "false", None: "unknown"}[
        metadata.provenance.git_dirty
    ]


def read_metadata(file: h5.File, source: Path) -> tuple[RunMetadata, CfgSnapshot]:
    """Decode required metadata/cfg and validate known fields; unknown cfg keys remain.

    Missing/null are distinct. Invalid JSON or wire values raise located
    ValueError; the caller adds source context for other decoding errors.
    """
    raw: JsonObject = {}
    for name in (
        "run_id",
        "experiment",
        "started_at",
        "finished_at",
        "completion",
        "labber_path",
    ):
        value = text_attr(file, name)
        raw[name] = (
            None if name in {"finished_at", "labber_path"} and value == "" else value
        )
    cfg_values = read_json(file, "cfg")
    cfg_node = known_node(file, "cfg")
    if not isinstance(cfg_node, h5.Dataset):
        raise ValueError("/cfg: expected dataset")
    cfg = CfgSnapshot(
        values=cfg_values,
        cfg_type=text_attr(cfg_node, "cfg_type"),
        schema_version=text_attr(cfg_node, "cfg_schema_version"),
    )
    raw["snapshot"] = read_json(file, "context")
    group = known_node(file, "provenance")
    if not isinstance(group, h5.Group):
        raise ValueError("/provenance: missing group")
    evidence: JsonObject = {
        "software_versions": json_object(
            text_attr(group, "software_versions"), "/provenance: software_versions"
        )
    }
    for name in _OPTIONAL_EVIDENCE:
        value = text_attr(group, name)
        evidence[name] = value or None
    dirty = text_attr(group, "git_dirty")
    if dirty not in {"true", "false", "unknown"}:
        raise ValueError("/provenance: git_dirty must be true, false or unknown")
    evidence["git_dirty"] = {"true": True, "false": False, "unknown": None}[dirty]
    raw["provenance"] = evidence
    try:
        metadata = _METADATA.validate_json(json.dumps(raw), strict=True)
    except ValidationError as error:
        raise ValueError(f"{source}: /context or /provenance: {error}") from error
    validate_metadata(metadata, cfg)
    return metadata, cfg
