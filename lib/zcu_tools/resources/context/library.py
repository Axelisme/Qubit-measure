"""Point-local persistent Specs, isolated snapshots and single-file edits.

Library paths are concrete component.partition.name roots. Partitions are
pulses/readout/reset for modules and waveforms for envelopes. No role expansion,
legacy flat-file bridge, implicit synchronization or reference cascade is
provided. DocumentStore owns version handling, conflicts and atomic replacement.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import TracebackType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    ValidationError,
    model_validator,
)

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

from ..document_store import DocumentChange, DocumentRevision, DocumentValue
from ..projection import typed_projection, typed_summary

MODULE_LIBRARY_FORMAT = "zcu.module-library"
MODULE_LIBRARY_VERSION = FormatVersion(2, 0)
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
    names to their partitions. invalid_entries holds omitted disk roots and their
    validation diagnostics, and is never serialized. Empty mappings are legal.
    Unknown current-minor fields are rejected; Store preserves future fields.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    format: Literal["zcu.module-library"]
    format_version: str
    components: dict[str, ComponentLibrary] = Field(default_factory=dict)
    invalid_entries: dict[str, str] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def ignore_legacy_provenance(cls, value: object) -> object:
        """Ignore retired source records without interpreting their contents."""
        if isinstance(value, dict):
            return {key: child for key, child in value.items() if key != "provenance"}
        return value

    def validate_content(self) -> None:
        """Validate persistent keys and Specs without I/O or materializing references."""
        _validate_document(self)


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


class LibraryContent(RootModel[YamlValue]):
    """Opaque captured library sections; requested roots are decoded by this owner.

    root may be missing (None) or an older unused shape. Construction never
    validates Specs. Accessing a malformed used section raises ValueError.
    """

    root: YamlValue = None

    def keys(self, *, waveform: bool) -> tuple[str, ...]:
        """List recorded roots in the requested family, without decoding Specs.

        Missing library content or malformed used containers raise ValueError;
        absent optional partitions contribute no keys. Other families stay opaque.
        """
        if not isinstance(self.root, dict) or not isinstance(
            components := self.root.get("components"), dict
        ):
            raise ValueError("library components must be a mapping")
        partitions = ("waveforms",) if waveform else ("pulses", "readout", "reset")
        roots: list[str] = []
        for component, library in components.items():
            if not isinstance(library, dict):
                raise ValueError(f"library {component}: expected mapping")
            for partition in partitions:
                names = library.get(partition, {})
                if not isinstance(names, dict):
                    raise ValueError(
                        f"library {component}.{partition}: expected mapping"
                    )
                for name in names:
                    root = f"{component}.{partition}.{name}"
                    _root_parts(root, kind="waveform" if waveform else "module")
                    roots.append(root)
        return tuple(sorted(roots))

    def spec(self, path: str) -> YamlValue:
        """Copy one concrete Spec; missing roots raise KeyError, bad sections ValueError."""
        node = self.root
        for part in ("components", *path.split(".")):
            if not isinstance(node, dict):
                raise ValueError(f"library {path}: expected mapping at {part}")
            if part not in node:
                raise KeyError(f"library {path}: missing {part}")
            node = node[part]
        return deepcopy(node)


class LibrarySnapshot:
    """An independent memory-only document and fixed ModuleResolver.

    Queries never read files or ledger. Spec queries preserve references; cfg
    queries materialize through core factories and the bound asset capability.
    Returned documents, models and YAML values are independent.
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
        self._content: LibraryContent | None = None

    @classmethod
    def from_content(cls, content: LibraryContent) -> LibrarySnapshot:
        """Restore opaque captured content; validate only requested roots on use."""
        result = cls(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        )
        result._content = content.model_copy(deep=True)
        return result

    def content(self) -> LibraryContent:
        """Copy captured content without decoding unused sections or Specs."""
        if self._content is not None:
            return self._content.model_copy(deep=True)
        return LibraryContent(
            _YAML_VALUE.validate_python(self._document.model_dump(mode="python"))
        )

    def document(self) -> ModuleLibraryDocument:
        """Copy the complete working-unit document without I/O."""
        if self._content is not None:
            return ModuleLibraryDocument.model_validate(self._content.root)
        return self._document.model_copy(deep=True)

    def projection(self, segments: tuple[str, ...]) -> object:
        """Project a native document subtree without decoding other Specs.

        Empty segments explicitly validate the complete library. Selected
        components, partitions and Specs retain their whole projection checks;
        selected leaves retain field checks. No references or assets are loaded.
        Invalid paths raise ProjectionPathError; used schema failures propagate.
        """
        value: object = self._document if self._content is None else self._content.root
        return typed_projection(value, ModuleLibraryDocument, segments)

    def summary(self, describe: Callable[[object], object]) -> object:
        """Describe sequence leaves and validate only fields emitted by a summary.

        describe formats a sequence's shape without projecting its elements.
        Used scalar fields keep their schema checks. No materialization, I/O or
        empty-on-error substitution occurs; malformed used content raises.
        """
        value: object = self._document if self._content is None else self._content.root
        return typed_summary(value, ModuleLibraryDocument, describe)

    def module_keys(self) -> tuple[str, ...]:
        """Return sorted concrete roots in the three module partitions."""
        if self._content is not None:
            return self._content.keys(waveform=False)
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
        if self._content is not None:
            return self._content.keys(waveform=True)
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
        if self._content is not None:
            return TypeAdapter(ModuleSpec).validate_python(
                self._content.spec(path), extra="ignore"
            )
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
        if self._content is not None:
            return TypeAdapter(WaveformSpec).validate_python(
                self._content.spec(path), extra="ignore"
            )
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

    def invalid_entries(self) -> dict[str, str]:
        """Return omitted disk roots and their Spec validation diagnostics."""
        return dict(self._document.invalid_entries)

    def renamed(
        self, from_path: str, to_path: str, *, kind: Literal["module", "waveform"]
    ) -> LibrarySnapshot:
        """Pure-relocate Spec in a copy, without evaluating children.

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
        return LibrarySnapshot(candidate, assets=self._assets)

    def replaced_module(self, path: str, value: ModuleSpec) -> LibrarySnapshot:
        """Return an isolated module replacement; invalid root/family raises ValueError."""
        candidate = self.document()
        _put_module(candidate, path, value)
        return LibrarySnapshot(candidate, assets=self._assets)

    def replaced_waveform(self, path: str, value: WaveformSpec) -> LibrarySnapshot:
        """Return an isolated waveform replacement; invalid root/family raises ValueError."""
        candidate = self.document()
        _put_waveform(candidate, path, value)
        return LibrarySnapshot(candidate, assets=self._assets)

    def deleted(
        self, path: str, *, kind: Literal["module", "waveform"]
    ) -> LibrarySnapshot:
        """Copy and remove one entry without cascading; errors follow entry_value."""
        candidate = self.document()
        _delete_entry(candidate, path, kind=kind)
        return LibrarySnapshot(candidate, assets=self._assets)


def _validate_document(document: ModuleLibraryDocument) -> None:
    """Validate persistent keys and Specs without external dependencies."""
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


class LibraryEdit:
    """One DocumentStore draft; every operation outside its body raises RuntimeError."""

    def __init__(
        self,
        document: ModuleLibraryDocument,
        *,
        assets: WaveformAssetInfo | None,
    ) -> None:
        """Bind a Store-owned draft and optional asset capability until edit exits."""
        self._document = document
        self._assets = assets
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
        """Validate a complete candidate before replacing the draft components."""
        self._ensure_active()
        candidate.validate_content()
        self._document.components = candidate.components

    def snapshot(self) -> LibrarySnapshot:
        """Copy this draft's current Specs for subsequent materialization."""
        self._ensure_active()
        return LibrarySnapshot(self._document, assets=self._assets)

    def set_module(self, path: str, value: ModuleSpec) -> None:
        """Set one validated module; invalid root/family leaves the draft unchanged."""
        candidate = self.snapshot().document()
        _put_module(candidate, path, value)
        self._install(candidate)

    def set_waveform(self, path: str, value: WaveformSpec) -> None:
        """Set one validated waveform; invalid root/family leaves the draft unchanged."""
        candidate = self.snapshot().document()
        _put_waveform(candidate, path, value)
        self._install(candidate)

    def replace_contents(self, candidate: LibrarySnapshot) -> None:
        """Install a complete isolated candidate into this active transaction.

        Validate schema before changing the transaction.
        Candidate may combine the snapshot's pure rename/delete/replacement
        operations; no references materialize and no files commit until edit
        exits successfully. Inactive draft, schema errors propagate.
        Subsequent snapshot reads see the replacement; body/conflict/I/O errors
        still leave the owning ModuleLibrary unpublished.
        """
        self._install(candidate.document())

    def rename(
        self, from_path: str, to_path: str, *, kind: Literal["module", "waveform"]
    ) -> None:
        """Pure-relocate this draft's Spec; policy follows renamed."""
        candidate = self.snapshot().renamed(from_path, to_path, kind=kind)
        self._install(candidate.document())

    def delete(self, path: str, *, kind: Literal["module", "waveform"]) -> None:
        """Delete only this root; errors follow entry_value, no reference cascade."""
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
    ) -> None:
        """Open existing module_cfg.yaml without writes or ledger dependencies.

        assets is optional for Spec-only operations; Arb materialization needs a
        provider. Invalid disk Specs are skipped with invalid_entries diagnostics.
        Header/container/I/O errors propagate. Draft edits remain strict.
        """
        self._path = path.resolve()
        self._assets = assets
        from .library_store import open_library_store

        self._store = open_library_store(self._path)

    @property
    def source_path(self) -> Path:
        """Return the absolute module_cfg.yaml path for external watchers."""
        return self._path

    @property
    def document_revision(self) -> DocumentRevision:
        """Return the installed library bytes' version without disk I/O."""
        return self._store.document_revision

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
            ) as draft,
        ):
            yield draft

    def refresh(self, *, strict: bool = False) -> bool:
        """Reload and publish changed valid content; return whether it changed.

        By default invalid disk Specs are skipped and reported in invalid_entries.
        strict=True validates the whole external candidate without skipping.
        Validation and I/O failures retain the installed snapshot and revision.
        Refresh never writes the document.
        """
        return self._store.refresh(strict=strict)

    def subscribe(
        self, callback: Callable[[DocumentChange], None]
    ) -> Callable[[], None]:
        """Observe published changes; return an idempotent unsubscribe function.

        Callback errors are isolated by DocumentStore and do not undo commits.
        """
        return self._store.subscribe(callback)
