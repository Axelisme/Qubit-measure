"""Point-local persistent Specs, isolated snapshots and single-file edits.

Library paths are concrete component.partition.name roots. Partitions are
pulses/readout/reset for modules and waveforms for envelopes. No role expansion,
legacy flat-file bridge, implicit synchronization or reference cascade is
provided. DocumentStore owns version handling, conflicts and atomic replacement.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from zcu_tools.cfg_model import (
    CfgValidationContext,
    ModuleCfg,
    ModuleCfgFactory,
    ModuleSpec,
    PulseSpec,
    ReadoutSpec,
    ResetSpec,
    WaveformAssetInfo,
    WaveformCfg,
    WaveformCfgFactory,
    WaveformSpec,
    module_spec_from_raw,
    waveform_spec_from_raw,
)
from zcu_tools.format_version import FormatVersion, YamlMap, YamlValue

from ..document_store import DocumentChange, DocumentStore, DocumentValue
from ..entry.provenance import Provenance, ProvenanceData, validate_source

MODULE_LIBRARY_FORMAT = "zcu.module-library"
MODULE_LIBRARY_VERSION = FormatVersion(1, 0)
_YAML_MAP = TypeAdapter(YamlMap)
_YAML_VALUE = TypeAdapter(YamlValue)
_READOUT_SPEC = TypeAdapter(ReadoutSpec)
_RESET_SPEC = TypeAdapter(ResetSpec)


class ComponentLibrary(BaseModel):
    """One component's persistent, named Specs, not materialized runtime cfgs.

    pulses contains PulseSpecs, readout the readout families, reset the reset
    families, and waveforms envelope Specs. Each mapping key is a nonempty
    dot-free name; omitted partitions are empty, not executable defaults.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    pulses: dict[str, PulseSpec] = Field(default_factory=dict)
    readout: dict[str, ReadoutSpec] = Field(default_factory=dict)
    reset: dict[str, ResetSpec] = Field(default_factory=dict)
    waveforms: dict[str, WaveformSpec] = Field(default_factory=dict)


class ModuleLibraryDocument(BaseModel):
    """Full working-unit YAML shape for a point's module_cfg.yaml.

    format is zcu.module-library; format_version is the major.minor header
    interpreted by DocumentStore. components maps nonempty dot-free component
    names to their partitions. provenance maps concrete dotted leaf paths,
    without a components prefix, to source records. Empty mappings are legal.
    Unknown current-minor fields are rejected; Store preserves future fields.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    format: Literal["zcu.module-library"]
    format_version: str
    components: dict[str, ComponentLibrary] = Field(default_factory=dict)
    provenance: dict[str, ProvenanceData] = Field(default_factory=dict)

    def validate_content(self, *, ledger: Path, entry_id: str) -> None:
        """Validate persistent keys, Specs and local source evidence, without mutation.

        ledger is the owning entry's records/ledger.jsonl; entry_id is its UUID.
        Reject invalid names, non-YAML values, invalid Spec source paths or missing
        local events. Ledger schema/attachment/I/O errors propagate. Manual sources
        do not consult the ledger. References and assets are not materialized.
        DocumentStore owns format/version checks; this method does not read YAML.
        """
        _validate_document(self, ledger=ledger, entry_id=entry_id)


@dataclass(frozen=True)
class LibraryEntryState:
    """An independent entry projection for fixed-proposal comparisons.

    value is its working-unit Spec mapping, preserving child reference strings.
    provenance maps relative leaf-segment tuples to independent source records.
    The empty tuple represents an existing root record, never a generated one.
    Missing sources remain missing; referenced children are not flattened.
    """

    value: YamlMap
    provenance: Mapping[tuple[str, ...], Provenance]


def _root_parts(
    path: str, *, kind: Literal["module", "waveform"]
) -> tuple[str, str, str]:
    parts = path.split(".")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid library root path: {path!r}")
    component, partition, name = parts
    partitions = ("pulses", "readout", "reset") if kind == "module" else ("waveforms",)
    if kind not in ("module", "waveform") or partition not in partitions:
        raise ValueError(f"Library kind {kind!r} does not match partition: {path!r}")
    return component, partition, name


def _path_root(path: str) -> tuple[str, Literal["module", "waveform"], tuple[str, ...]]:
    parts = path.split(".")
    if len(parts) < 3 or not all(parts):
        raise ValueError(f"Invalid library path: {path!r}")
    kind: Literal["module", "waveform"] = (
        "waveform" if parts[1] == "waveforms" else "module"
    )
    root = ".".join(parts[:3])
    _root_parts(root, kind=kind)
    return root, kind, tuple(parts[3:])


def library_key_kind(path: str) -> Literal["module", "waveform"]:
    """Validate a concrete library root and return its module/waveform family.

    path is exactly component.partition.name. pulses/readout/reset are module
    partitions; waveforms is the waveform partition. Empty segments, leaf paths
    and unknown partitions raise ValueError. No document, lookup or I/O is used.
    """
    _, kind, suffix = _path_root(path)
    if suffix:
        raise ValueError(f"Expected a library root, not a leaf: {path!r}")
    return kind


def _validate_names(document: ModuleLibraryDocument) -> None:
    for component, library in document.components.items():
        if not component or "." in component:
            raise ValueError(f"Invalid component path segment: {component!r}")
        for name in (
            *library.pulses,
            *library.readout,
            *library.reset,
            *library.waveforms,
        ):
            if not name or "." in name:
                raise ValueError(f"Invalid library name path segment: {name!r}")


def _put_module(document: ModuleLibraryDocument, path: str, value: ModuleSpec) -> None:
    component, partition, name = _root_parts(path, kind="module")
    spec = module_spec_from_raw(value)
    library = document.components.setdefault(component, ComponentLibrary())
    match partition:
        case "pulses":
            if not isinstance(spec, PulseSpec):
                raise ValueError(f"Wrong module family for partition: {path!r}")
            library.pulses[name] = spec
        case "readout":
            if not spec.type.startswith("readout/"):
                raise ValueError(f"Wrong module family for partition: {path!r}")
            library.readout[name] = _READOUT_SPEC.validate_python(spec)
        case "reset":
            if not spec.type.startswith("reset/"):
                raise ValueError(f"Wrong module family for partition: {path!r}")
            library.reset[name] = _RESET_SPEC.validate_python(spec)
        case _:
            raise ValueError(f"Invalid module partition: {path!r}")


def _put_waveform(
    document: ModuleLibraryDocument, path: str, value: WaveformSpec
) -> None:
    component, _, name = _root_parts(path, kind="waveform")
    spec = waveform_spec_from_raw(value)
    document.components.setdefault(component, ComponentLibrary()).waveforms[name] = spec


def _remove_sources(document: ModuleLibraryDocument, path: str) -> None:
    prefix = path + "."
    document.provenance = {
        key: source
        for key, source in document.provenance.items()
        if key != path and not key.startswith(prefix)
    }


def _leaf_paths(value: YamlValue, prefix: str) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaf_paths(child, prefix + "." + key)
    else:
        # Lists are a single field, not index-addressable provenance paths.
        yield prefix


def _accept_sources(
    document: ModuleLibraryDocument,
    path: str,
    value: YamlMap,
    source: Provenance,
) -> None:
    record = ProvenanceData.from_provenance(source)
    _remove_sources(document, path)
    for leaf in _leaf_paths(value, path):
        document.provenance[leaf] = record.model_copy(deep=True)


def _manual_source() -> Provenance:
    return Provenance(
        "manual", None, None, datetime.now(timezone.utc).isoformat(), None
    )


def _delete_entry(
    document: ModuleLibraryDocument,
    path: str,
    *,
    kind: Literal["module", "waveform"],
) -> None:
    component, partition, name = _root_parts(path, kind=kind)
    library = document.components[component]
    match partition:
        case "pulses":
            del library.pulses[name]
        case "readout":
            del library.readout[name]
        case "reset":
            del library.reset[name]
        case "waveforms":
            del library.waveforms[name]
        case _:
            raise ValueError(f"Invalid library partition: {path!r}")
    _remove_sources(document, path)


class LibrarySnapshot:
    """An independent memory-only document and fixed ModuleResolver.

    Queries never read files or ledger. Spec queries preserve references; cfg
    queries materialize through core factories and the bound asset capability.
    Returned documents, models, YAML values and provenance are independent.
    """

    def __init__(
        self,
        document: ModuleLibraryDocument,
        *,
        assets: WaveformAssetInfo | None = None,
    ) -> None:
        """Copy a validated document; assets may be absent until arb is used."""
        _validate_names(document)
        self._document = document.model_copy(deep=True)
        self._assets = assets

    def document(self) -> ModuleLibraryDocument:
        """Copy the complete working-unit document without I/O."""
        return self._document.model_copy(deep=True)

    def module_keys(self) -> tuple[str, ...]:
        """Return sorted concrete roots in the three module partitions."""
        return tuple(
            sorted(
                f"{component}.{partition}.{name}"
                for component, library in self._document.components.items()
                for partition, names in (
                    ("pulses", library.pulses.keys()),
                    ("readout", library.readout.keys()),
                    ("reset", library.reset.keys()),
                )
                for name in names
            )
        )

    def waveform_keys(self) -> tuple[str, ...]:
        """Return sorted concrete roots in the waveforms partition."""
        return tuple(
            sorted(
                f"{component}.waveforms.{name}"
                for component, library in self._document.components.items()
                for name in library.waveforms
            )
        )

    def validation_context(self) -> CfgValidationContext:
        """Return a fresh context bound to this resolver and its asset capability."""
        return CfgValidationContext(modules=self, assets=self._assets)

    def module_spec(self, path: str) -> ModuleSpec:
        """Copy a module Spec; invalid root/kind raises ValueError, missing KeyError."""
        component, partition, name = _root_parts(path, kind="module")
        library = self._document.components[component]
        match partition:
            case "pulses":
                spec = library.pulses[name]
            case "readout":
                spec = library.readout[name]
            case "reset":
                spec = library.reset[name]
            case _:
                raise ValueError(f"Invalid module partition: {path!r}")
        return spec.model_copy(deep=True)

    def waveform_spec(self, path: str) -> WaveformSpec:
        """Copy a waveform Spec; invalid root/kind raises ValueError, missing KeyError."""
        component, _, name = _root_parts(path, kind="waveform")
        return (
            self._document.components[component].waveforms[name].model_copy(deep=True)
        )

    def get_module(self, path: str) -> ModuleCfg:
        """Materialize a concrete module root; core reference/asset failures propagate."""
        _root_parts(path, kind="module")
        return ModuleCfgFactory.from_raw(path, context=self.validation_context())

    def get_waveform(self, path: str) -> WaveformCfg:
        """Materialize a concrete waveform root; core reference/asset failures propagate."""
        _root_parts(path, kind="waveform")
        return WaveformCfgFactory.from_raw(path, context=self.validation_context())

    def value(self, path: str) -> YamlValue:
        """Read a materialized root's declared leaf; missing raises AttributeError.

        path is root plus dotted data suffix, never a role. Core resolution
        failures propagate; non-YAML runtime values raise TypeError.
        """
        result = self.lookup(path)
        if not result.present:
            raise AttributeError(path)
        return result.value

    def lookup(self, path: str) -> DocumentValue:
        """Read a materialized leaf, preserving missing versus null.

        Missing requested roots/leaves return present=False. Invalid syntax,
        nested reference, core validation and asset failures propagate. Runtime
        leaves without a YAML representation raise TypeError.
        """
        root, kind, suffix = _path_root(path)
        if not suffix:
            raise ValueError(f"Library value path requires a leaf: {path!r}")
        try:
            if kind == "module":
                self.module_spec(root)
            else:
                self.waveform_spec(root)
        except KeyError:
            # Only the requested root is absent; materialization is outside this catch.
            return DocumentValue(False, None)
        value: object = (
            self.get_module(root) if kind == "module" else self.get_waveform(root)
        )
        for segment in suffix:
            if (
                not isinstance(value, BaseModel)
                or segment not in type(value).model_fields
            ):
                return DocumentValue(False, None)
            value = getattr(value, segment)
        try:
            return DocumentValue(True, _YAML_VALUE.validate_python(value))
        except ValidationError as exc:
            raise TypeError(
                f"Library leaf is not YAML-representable: {path!r}"
            ) from exc

    def entry_value(self, path: str, *, kind: Literal["module", "waveform"]) -> YamlMap:
        """Copy a root's Spec mapping without materialization; ValueError/KeyError on bad/missing root."""
        _root_parts(path, kind=kind)
        spec = self.module_spec(path) if kind == "module" else self.waveform_spec(path)
        return _YAML_MAP.validate_python(spec.model_dump(mode="python"))

    def meta(self, path: str) -> Provenance | None:
        """Copy a recorded concrete path's source, or None; illegal paths raise ValueError."""
        _path_root(path)
        source = self._document.provenance.get(path)
        return source.to_provenance() if source is not None else None

    def entry_state(
        self, path: str, *, kind: Literal["module", "waveform"]
    ) -> LibraryEntryState:
        """Copy a complete Spec and its relative recorded sources; errors follow entry_value."""
        value = self.entry_value(path, kind=kind)
        prefix = path + "."
        sources = {
            (
                ()
                if source_path == path
                else tuple(source_path[len(prefix) :].split("."))
            ): source.to_provenance()
            for source_path, source in self._document.provenance.items()
            if source_path == path or source_path.startswith(prefix)
        }
        return LibraryEntryState(value, sources)

    def renamed(
        self, from_path: str, to_path: str, *, kind: Literal["module", "waveform"]
    ) -> LibrarySnapshot:
        """Pure-relocate Spec and sources in a copy, without evaluating children.

        Missing source raises KeyError; identical/existing destination or wrong
        family raises ValueError. No other references or ledger events change.
        """
        _root_parts(from_path, kind=kind)
        _root_parts(to_path, kind=kind)
        if from_path == to_path:
            raise ValueError(f"Rename destination equals source: {to_path!r}")
        self.entry_value(from_path, kind=kind)
        try:
            self.entry_value(to_path, kind=kind)
        except KeyError:
            pass  # An absent destination is the required rename precondition.
        else:
            raise ValueError(f"Rename destination already exists: {to_path!r}")
        candidate = self.document()
        if kind == "module":
            _put_module(candidate, to_path, self.module_spec(from_path))
        else:
            _put_waveform(candidate, to_path, self.waveform_spec(from_path))
        _delete_entry(candidate, from_path, kind=kind)
        prefix = from_path + "."
        for source_path, source in self._document.provenance.items():
            if source_path == from_path or source_path.startswith(prefix):
                candidate.provenance[to_path + source_path[len(from_path) :]] = (
                    source.model_copy(deep=True)
                )
        return LibrarySnapshot(candidate, assets=self._assets)

    def replaced_module(self, path: str, value: ModuleSpec) -> LibrarySnapshot:
        """Copy and set a validated module Spec with manual leaf sources; wrong path/family raises ValueError."""
        candidate = self.document()
        _put_module(candidate, path, value)
        projection = LibrarySnapshot(candidate, assets=self._assets)
        _accept_sources(
            candidate,
            path,
            projection.entry_value(path, kind="module"),
            _manual_source(),
        )
        return LibrarySnapshot(candidate, assets=self._assets)

    def replaced_waveform(self, path: str, value: WaveformSpec) -> LibrarySnapshot:
        """Copy and set a validated envelope Spec with manual leaf sources; wrong path/kind raises ValueError."""
        candidate = self.document()
        _put_waveform(candidate, path, value)
        projection = LibrarySnapshot(candidate, assets=self._assets)
        _accept_sources(
            candidate,
            path,
            projection.entry_value(path, kind="waveform"),
            _manual_source(),
        )
        return LibrarySnapshot(candidate, assets=self._assets)

    def deleted(
        self, path: str, *, kind: Literal["module", "waveform"]
    ) -> LibrarySnapshot:
        """Copy and remove one entry and sources without cascading; errors follow entry_value."""
        candidate = self.document()
        _delete_entry(candidate, path, kind=kind)
        return LibrarySnapshot(candidate, assets=self._assets)


def _validate_document(
    document: ModuleLibraryDocument,
    *,
    ledger: Path,
    entry_id: str,
) -> None:
    """Validate persistent Specs, addressable sources and local source evidence."""
    _validate_names(document)
    _YAML_MAP.validate_python(document.model_dump(mode="python"))
    for library in document.components.values():
        for spec in (
            *library.pulses.values(),
            *library.readout.values(),
            *library.reset.values(),
        ):
            module_spec_from_raw(spec)
        for waveform in library.waveforms.values():
            waveform_spec_from_raw(waveform)
    snapshot = LibrarySnapshot(document)
    for path, record in document.provenance.items():
        root, kind, suffix = _path_root(path)
        value: YamlValue = snapshot.entry_value(root, kind=kind)
        for segment in suffix:
            if not isinstance(value, dict) or segment not in value:
                raise ValueError(f"Source path is not a persistent Spec leaf: {path!r}")
            value = value[segment]
        validate_source(record.to_provenance(), ledger, entry_id)


class LibraryEdit:
    """One DocumentStore draft; every operation outside its body raises RuntimeError."""

    def __init__(
        self,
        document: ModuleLibraryDocument,
        *,
        assets: WaveformAssetInfo | None,
        ledger: Path,
        entry_id: str,
    ) -> None:
        """Bind a Store-owned draft and its scope; callers receive this from edit().

        document is the mutable draft, not the published snapshot. assets,
        ledger and entry_id are the owning handle's fixed capabilities.
        ModuleLibrary expires this object when its with body exits.
        """
        self._document = document
        self._assets = assets
        self._ledger = ledger
        self._entry_id = entry_id
        self._active = True

    def _ensure_active(self) -> None:
        """Reject access once the owning with body has exited."""
        if not self._active:
            raise RuntimeError("LibraryEdit is outside its edit body")

    def __enter__(self) -> LibraryEdit:
        """Enter this live draft's body; expired drafts raise RuntimeError."""
        self._ensure_active()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Expire the draft; propagate any body failure without commit or suppression.

        The three arguments describe the body's exception, or are all None on
        normal exit. The enclosing DocumentStore context owns commit.
        """
        self._active = False

    def _install(self, candidate: ModuleLibraryDocument) -> None:
        """Validate a complete candidate before replacing the two draft mappings."""
        self._ensure_active()
        candidate.validate_content(ledger=self._ledger, entry_id=self._entry_id)
        self._document.components = candidate.components
        self._document.provenance = candidate.provenance

    def snapshot(self) -> LibrarySnapshot:
        """Copy this draft's current Specs and sources for subsequent materialization."""
        self._ensure_active()
        return LibrarySnapshot(self._document, assets=self._assets)

    def set_module(
        self, path: str, value: ModuleSpec, *, provenance: Provenance | None = None
    ) -> None:
        """Set a validated module root and leaf sources; omit provenance for manual.

        Invalid root/family/source raises without changing the draft. Explicit
        non-manual sources must exist in this entry's ledger and attachments.
        """
        candidate = self.snapshot().document()
        _put_module(candidate, path, value)
        projection = LibrarySnapshot(candidate, assets=self._assets)
        _accept_sources(
            candidate,
            path,
            projection.entry_value(path, kind="module"),
            provenance or _manual_source(),
        )
        self._install(candidate)

    def set_waveform(
        self, path: str, value: WaveformSpec, *, provenance: Provenance | None = None
    ) -> None:
        """Set a validated waveform root and leaf sources, with set_module's source policy."""
        candidate = self.snapshot().document()
        _put_waveform(candidate, path, value)
        projection = LibrarySnapshot(candidate, assets=self._assets)
        _accept_sources(
            candidate,
            path,
            projection.entry_value(path, kind="waveform"),
            provenance or _manual_source(),
        )
        self._install(candidate)

    def accept_source(self, path: str, *, provenance: Provenance) -> None:
        """Attach a validated source to an existing entry's leaves, without changing its Spec.

        Missing roots raise KeyError; invalid roots/sources raise ValueError.
        Ledger schema, attachment and I/O failures propagate.
        """
        snapshot = self.snapshot()
        root, kind, suffix = _path_root(path)
        if suffix:
            raise ValueError(f"Source acceptance requires a root path: {path!r}")
        value = snapshot.entry_value(root, kind=kind)
        candidate = snapshot.document()
        _accept_sources(candidate, root, value, provenance)
        self._install(candidate)

    def rename(
        self, from_path: str, to_path: str, *, kind: Literal["module", "waveform"]
    ) -> None:
        """Pure-relocate this draft's Spec and all recorded sources; policy follows renamed."""
        candidate = self.snapshot().renamed(from_path, to_path, kind=kind)
        self._install(candidate.document())

    def delete(self, path: str, *, kind: Literal["module", "waveform"]) -> None:
        """Delete only this root and sources; errors follow entry_value, no reference cascade."""
        candidate = self.snapshot().deleted(path, kind=kind)
        self._install(candidate.document())


class ModuleLibrary:
    """Point-local DocumentStore handle; no live store or mutable cfg is exposed.

    Snapshot is memory-only. edit/refresh use the existing single-file
    optimistic transaction and publication contract, never multi-file atomicity.
    """

    def __init__(
        self,
        path: Path,
        *,
        assets: WaveformAssetInfo | None = None,
        ledger: Path,
        entry_id: str,
    ) -> None:
        """Open existing YAML with this entry's source scope, without creation.

        path is the point-local module_cfg.yaml; assets is its fixed capability,
        or None for Spec-only operations. Materializing Arb requires a provider.
        ledger names the entry's records/ledger.jsonl; entry_id is its UUID.
        Header/schema/source/I/O errors propagate. Opening does not read assets.
        """
        self._path = path.resolve()
        self._assets = assets
        self._ledger = ledger.resolve()
        self._entry_id = entry_id
        self._store = DocumentStore(
            self._path,
            ModuleLibraryDocument,
            format=MODULE_LIBRARY_FORMAT,
            supported_version=MODULE_LIBRARY_VERSION,
            validate=self._validate,
        )

    def _validate(self, document: ModuleLibraryDocument) -> None:
        """Validate this handle's persistent document and entry-scoped source evidence."""
        document.validate_content(ledger=self._ledger, entry_id=self._entry_id)

    @property
    def source_path(self) -> Path:
        """Return the absolute module_cfg.yaml path for external watchers."""
        return self._path

    def snapshot(self) -> LibrarySnapshot:
        """Return an isolated copy of the published document, without file I/O or notice."""
        return LibrarySnapshot(self._store.snapshot(), assets=self._assets)

    @contextmanager
    def edit(self) -> Generator[LibraryEdit]:
        """Open a draft and commit on successful exit; body/schema/conflict/I/O failures do not publish.

        Nested edits raise RuntimeError. Draft operations expire on exit.
        Subscribers observe only the committed single-file snapshot.
        """
        with (
            self._store.edit() as document,
            LibraryEdit(
                document,
                assets=self._assets,
                ledger=self._ledger,
                entry_id=self._entry_id,
            ) as draft,
        ):
            yield draft

    def refresh(self) -> bool:
        """Reload and publish changed valid content; return whether it changed.

        Source/schema/header/I/O failures propagate and retain published state.
        """
        return self._store.refresh()

    def subscribe(
        self, callback: Callable[[DocumentChange], None]
    ) -> Callable[[], None]:
        """Observe published changes; return an idempotent unsubscribe function.

        Callback errors are isolated by DocumentStore and do not undo commits.
        """
        return self._store.subscribe(callback)
