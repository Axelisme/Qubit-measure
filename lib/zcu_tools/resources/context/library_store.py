"""Library disk admission shared by handles and point cloning, without writes."""

from copy import copy, deepcopy
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from ruamel.yaml import YAML

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue, validate_header

from ..document_store import DocumentStore
from .library import (
    MODULE_LIBRARY_FORMAT,
    MODULE_LIBRARY_VERSION,
    ComponentLibrary,
    ModuleLibraryDocument,
    library_key_kind,
)

_PARTITIONS = ("pulses", "readout", "reset", "waveforms")


class _ComponentContainer(BaseModel):
    """Validate partition shape without attempting to parse individual Specs."""

    model_config = ConfigDict(extra="forbid")
    pulses: dict[str, YamlValue] = Field(default_factory=dict)
    readout: dict[str, YamlValue] = Field(default_factory=dict)
    reset: dict[str, YamlValue] = Field(default_factory=dict)
    waveforms: dict[str, YamlValue] = Field(default_factory=dict)


class _LibraryContainer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: str
    format_version: str
    components: dict[str, _ComponentContainer] = Field(default_factory=dict)


def _read_document(document: YamlMap, version: FormatVersion) -> YamlMap:
    document.pop("provenance", None)
    extra = "allow" if version.minor > MODULE_LIBRARY_VERSION.minor else None
    container = _LibraryContainer.model_validate(document, extra=extra)
    invalid: dict[str, YamlValue] = {}
    components = document.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("Library components must be a mapping")
    for component, partitions in container.components.items():
        if not component or "." in component:
            raise ValueError(f"Invalid component path segment: {component!r}")
        raw_partitions = components[component]
        if not isinstance(raw_partitions, dict):
            raise ValueError("Library component must be a mapping")
        raw_partitions = copy(raw_partitions)
        components[component] = raw_partitions
        for partition in _PARTITIONS:
            entries = copy(getattr(partitions, partition))
            for name, raw in tuple(entries.items()):
                root = f"{component}.{partition}.{name}"
                library_key_kind(root)
                try:
                    if partition == "pulses" and (
                        not isinstance(raw, dict) or "type" not in raw
                    ):
                        raise ValueError("Pulse Spec requires type")
                    ComponentLibrary.model_validate({partition: {name: raw}})
                except ValueError as exc:
                    invalid[root] = str(exc)
                    del entries[name]
            if partition in raw_partitions:
                raw_partitions[partition] = entries
    document["invalid_entries"] = invalid
    return document


def copy_library(source: Path, destination: Path) -> None:
    """Copy admitted Specs to a new file, omitting legacy sources and invalid roots.

    A missing source leaves destination absent. Otherwise destination must not
    exist and its parent must exist. Source bytes stay unchanged. Forward-minor
    fields survive. I/O, header and container errors propagate; no ledger or
    attachment is accessed.
    """
    store = open_library_store(source, allow_absent=True)
    with store.locked():
        store.refresh()
        try:
            original = source.read_bytes()
        except FileNotFoundError:
            return
        document = TypeAdapter(YamlMap).validate_python(
            YAML(typ="rt").load(original.decode("utf-8"))
        )
        version = validate_header(
            document,
            expected_format=MODULE_LIBRARY_FORMAT,
            supported_version=MODULE_LIBRARY_VERSION,
            source=source,
        )
        candidate = _read_document(deepcopy(document), version)
        candidate.pop("invalid_entries", None)
        if candidate == document:
            with destination.open("xb") as stream:
                stream.write(original)
        else:
            with destination.open("x", encoding="utf-8") as stream:
                YAML(typ="rt").dump(candidate, stream)


def open_library_store(
    path: Path, *, allow_absent: bool = False
) -> DocumentStore[ModuleLibraryDocument]:
    """Open module_cfg.yaml without modifying it or consulting a ledger.

    Invalid disk Specs are omitted from the snapshot and recorded by root in
    invalid_entries. Header/container errors propagate. Strict refresh rejects
    invalid Specs instead of omitting them. Draft operations remain strict.
    Legacy provenance is ignored; a nonempty commit drops it from disk.
    allow_absent supplies an empty snapshot with revision None for a missing
    file. Nonempty edits then create it. No references materialize and reads
    never create missing files or directories.
    """
    return DocumentStore(
        path,
        ModuleLibraryDocument,
        format=MODULE_LIBRARY_FORMAT,
        supported_version=MODULE_LIBRARY_VERSION,
        validate=lambda document: document.validate_content(),
        read_normalize=_read_document,
        absent_document=(
            ModuleLibraryDocument(format=MODULE_LIBRARY_FORMAT, format_version="2.0")
            if allow_absent
            else None
        ),
    )
