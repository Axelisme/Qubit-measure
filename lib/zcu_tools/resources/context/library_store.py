"""Library disk admission and entry-local cleanup, shared by handles and clone."""

from copy import copy
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue

from ..document_store import DocumentStore
from ..entry.ledger import RecordsLedger
from ..entry.provenance import Provenance, ProvenanceData, missing_sources
from .library import (
    MODULE_LIBRARY_FORMAT,
    MODULE_LIBRARY_VERSION,
    ComponentLibrary,
    ModuleLibraryDocument,
    has_persistent_source_suffix,
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


def _entry_sources(
    partition: str,
    name: str,
    raw: YamlValue,
    sources: dict[str, YamlValue],
) -> tuple[Provenance, ...] | None:
    # PulseSpec defaults are for authored values, not missing disk discriminators.
    if partition == "pulses" and (not isinstance(raw, dict) or "type" not in raw):
        return None
    # Current entry extra rules apply even inside a forward-minor container.
    try:
        component = ComponentLibrary.model_validate({partition: {name: raw}})
    except ValidationError:
        return None
    spec = component.model_dump(mode="python")[partition][name]
    provenances: list[Provenance] = []
    for path, record in sources.items():
        try:
            provenance = ProvenanceData.model_validate(record)
        except ValidationError:
            return None
        if not has_persistent_source_suffix(spec, tuple(path.split(".")[3:])):
            return None
        provenances.append(provenance.to_provenance())
    return tuple(provenances)


def _clean_disk_entries(
    document: YamlMap,
    version: FormatVersion,
    *,
    ledger: Path | RecordsLedger,
    entry_id: str,
) -> tuple[str, ...]:
    extra = "allow" if version.minor > MODULE_LIBRARY_VERSION.minor else None
    container = _LibraryContainer.model_validate(document, extra=extra)
    roots = _roots(container)
    sources = _sources_by_root(container, roots)
    entry_sources = {
        root: _entry_sources(
            partition,
            name,
            getattr(container.components[component], partition)[name],
            sources[root],
        )
        for root, (component, partition, name) in roots.items()
    }
    missing = missing_sources(
        (
            source
            for provenances in entry_sources.values()
            if provenances is not None
            for source in provenances
        ),
        ledger,
        entry_id,
    )
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
        provenances = entry_sources[root]
        if provenances is None or any(
            source.source in missing for source in provenances
        ):
            removed.append(root)
            # YAML aliases may share either ancestor. Detach this root's path
            # before removal so siblings keep their own Spec/source admission.
            partitions = copy(partitions)
            entries = copy(entries)
            del entries[name]
            partitions[partition] = entries
            components[component] = partitions
    provenance = document.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Library provenance must be a mapping")
    for root in removed:
        for path in sources[root]:
            del provenance[path]
    return tuple(removed)


def open_library_store(
    path: Path, *, ledger: Path | RecordsLedger, entry_id: str
) -> DocumentStore[ModuleLibraryDocument]:
    """Open an existing library with locked backup-before-clean disk reads.

    path is module_cfg.yaml; ledger is its owner's records/ledger.jsonl path or
    scoped RecordsLedger. entry_id opens path inputs; supplied handles must belong
    to this entry and already own their scope. The store retains one ledger for
    dependency checks, cleanup and full validation, reusing only the validated
    event index, never attachment contents. Container/header and unassignable
    provenance errors reject the whole file. Invalid Specs or entry-local provenance remove their root.
    Ledger I/O/schema/attachment and unexpected failures propagate. The store
    validates the entire cleaned candidate before backup and atomic replacement.
    Unchanged library bytes reuse validation after checking every nonmanual
    source in a retained ledger index and stat-ing its local attachment. Missing
    dependencies trigger full admission/cleanup; other dependency errors propagate.
    Attachment contents are revalidated on changed library bytes or nonempty edits,
    not on unchanged reads. Drafts remain strict; this does not create files or
    materialize references.
    """
    records = (
        ledger
        if isinstance(ledger, RecordsLedger)
        else RecordsLedger(ledger.parent, entry_id=entry_id)
    )

    def check_dependencies(document: ModuleLibraryDocument) -> bool:
        sources = tuple(
            dict.fromkeys(
                record.source
                for record in document.provenance.values()
                if record.source != "manual"
            )
        )
        return records.dependencies_exist(sources)

    def validate(document: ModuleLibraryDocument) -> None:
        document.validate_content(ledger=records, entry_id=entry_id)

    def cleanup(document: YamlMap, version: FormatVersion) -> tuple[str, ...]:
        return _clean_disk_entries(document, version, ledger=records, entry_id=entry_id)

    return DocumentStore(
        path,
        ModuleLibraryDocument,
        format=MODULE_LIBRARY_FORMAT,
        supported_version=MODULE_LIBRARY_VERSION,
        validate=validate,
        check_dependencies=check_dependencies,
        read_cleanup=cleanup,
    )
