"""Library disk admission and entry-local cleanup, shared by handles and clone."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue

from ..document_store import DocumentStore
from ..entry.provenance import MissingSourceError, ProvenanceData, validate_source
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
    provenance: dict[str, YamlValue] = Field(default_factory=dict)


def _roots(container: _LibraryContainer) -> dict[str, tuple[str, str, str]]:
    roots: dict[str, tuple[str, str, str]] = {}
    for component, partitions in container.components.items():
        if not component or "." in component:
            raise ValueError(f"Invalid component path segment: {component!r}")
        for partition in _PARTITIONS:
            for name in getattr(partitions, partition):
                root = f"{component}.{partition}.{name}"
                library_key_kind(root)
                roots[root] = (component, partition, name)
    return roots


def _sources_by_root(
    container: _LibraryContainer, roots: dict[str, tuple[str, str, str]]
) -> dict[str, dict[str, YamlValue]]:
    sources: dict[str, dict[str, YamlValue]] = {root: {} for root in roots}
    for path, record in container.provenance.items():
        segments = path.split(".")
        root = ".".join(segments[:3])
        library_key_kind(root)
        if not all(segments) or root not in roots:
            raise ValueError(f"Source path has no library entry: {path!r}")
        sources[root][path] = record
    return sources


def _valid_entry(
    partition: str,
    name: str,
    raw: YamlValue,
    sources: dict[str, YamlValue],
    *,
    ledger: Path,
    entry_id: str,
) -> bool:
    # Current entry extra rules apply even inside a forward-minor container.
    try:
        component = ComponentLibrary.model_validate({partition: {name: raw}})
    except ValidationError:
        return False
    spec = component.model_dump(mode="python")[partition][name]
    for path, record in sources.items():
        try:
            provenance = ProvenanceData.model_validate(record)
        except ValidationError:
            return False
        value = spec
        for segment in path.split(".")[3:]:
            if not isinstance(value, dict) or segment not in value:
                return False
            value = value[segment]
        try:
            validate_source(provenance.to_provenance(), ledger, entry_id)
        except MissingSourceError:
            return False
    return True


def _clean_disk_entries(
    document: YamlMap, version: FormatVersion, *, ledger: Path, entry_id: str
) -> tuple[str, ...]:
    extra = "allow" if version.minor > MODULE_LIBRARY_VERSION.minor else None
    container = _LibraryContainer.model_validate(document, extra=extra)
    roots = _roots(container)
    sources = _sources_by_root(container, roots)
    removed: list[str] = []
    components = document.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("Library components must be a mapping")
    for root, (component, partition, name) in roots.items():
        partitions = components[component]
        if not isinstance(partitions, dict):
            raise ValueError("Library component must be a mapping")
        entries = partitions[partition]
        if not isinstance(entries, dict):
            raise ValueError("Library partition must be a mapping")
        if not _valid_entry(
            partition,
            name,
            entries[name],
            sources[root],
            ledger=ledger,
            entry_id=entry_id,
        ):
            removed.append(root)
            del entries[name]
    provenance = document.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Library provenance must be a mapping")
    for root in removed:
        for path in sources[root]:
            del provenance[path]
    return tuple(removed)


def open_library_store(
    path: Path, *, ledger: Path, entry_id: str
) -> DocumentStore[ModuleLibraryDocument]:
    """Open an existing library with locked backup-before-clean disk reads.

    path is module_cfg.yaml; ledger is its owner's records/ledger.jsonl and
    entry_id its UUID. Container/header and unassignable provenance errors reject
    the whole file. Invalid Specs or entry-local provenance remove their root.
    Ledger I/O/schema/attachment and unexpected failures propagate. The store
    validates the entire cleaned candidate before backup and atomic replacement.
    Drafts remain strict; this does not create files or materialize references.
    """

    def validate(document: ModuleLibraryDocument) -> None:
        document.validate_content(ledger=ledger, entry_id=entry_id)

    def cleanup(document: YamlMap, version: FormatVersion) -> tuple[str, ...]:
        return _clean_disk_entries(document, version, ledger=ledger, entry_id=entry_id)

    return DocumentStore(
        path,
        ModuleLibraryDocument,
        format=MODULE_LIBRARY_FORMAT,
        supported_version=MODULE_LIBRARY_VERSION,
        validate=validate,
        read_cleanup=cleanup,
    )
