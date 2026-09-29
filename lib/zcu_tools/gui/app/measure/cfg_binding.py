from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Protocol, cast

from zcu_tools.experiment.cfg_editing import ProgramCfgKind, program_shape_for_input
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
)
from zcu_tools.gui.cfg.binding import CfgDraft, ResolvedReference
from zcu_tools.gui.cfg.resource import (
    CfgPreconditionError,
    CfgPreconditionReason,
    CfgResolution,
    CfgRevision,
    SourceBasis,
    SourceRevision,
)
from zcu_tools.gui.session.expression import evaluate_scalar_expr, validate_scalar_expr
from zcu_tools.gui.session.state import DEVICE_SET_VERSION_KEY, SessionState
from zcu_tools.gui.session.value_lookup import (
    ScalarValue as LookupScalarValue,
)
from zcu_tools.gui.session.value_lookup import (
    ValueInfo,
    ValueRef,
    ValueTypeError,
    name_from_type,
)
from zcu_tools.resources.context import MetaDict, ModuleLibrary
from zcu_tools.resources.context.content import snapshot_context_contents

from .cfg_schemas import module_cfg_to_value, waveform_cfg_to_value

_DEVICES_SOURCE = "devices"
_ARB_WAVEFORMS_SOURCE = "arb_waveforms"

_ReferenceConverter = Callable[[object], tuple[CfgSectionSpec, CfgSectionValue]]


class MeasureCfgBindingHost(Protocol):
    def get_current_md(self) -> MetaDict: ...

    def get_current_ml(self) -> ModuleLibrary: ...

    def list_device_names(self) -> list[str]: ...

    def list_arb_waveforms(self) -> list[str]: ...

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, LookupScalarValue]: ...


class MeasureCfgBindings:
    """Measure-app policy adapter for the shared mutable cfg binding."""

    def __init__(self, host: MeasureCfgBindingHost) -> None:
        self._host = host

    def snapshot(
        self,
        source_basis: SourceBasis,
        *,
        captured_values: Mapping[str, object],
    ) -> CfgResolution:
        """Freeze published local data on the owner sequence, never read hardware.

        The caller supplies matching provenance and cached dotted capture values.
        Bare capture names use the same metadata snapshot as dynamic expressions.
        No live provider is retained by the returned resolution.
        """
        md, ml = snapshot_context_contents(
            self._host.get_current_md(), self._host.get_current_ml()
        )
        references = _SnapshotReferences(ml)
        options = {
            _DEVICES_SOURCE: tuple(self._host.list_device_names()),
            _ARB_WAVEFORMS_SOURCE: tuple(self._host.list_arb_waveforms()),
        }
        captures = deepcopy(dict(captured_values))
        metadata = dict(md.items())

        def evaluate(expression: str) -> int | float | complex:
            return evaluate_scalar_expr(expression, md)

        def provide_options(source_id: str) -> Sequence[object]:
            if source_id not in options:
                raise RuntimeError(
                    f"Unsupported measure cfg option source {source_id!r}"
                )
            return options[source_id]

        def read_capture(name: str) -> object:
            if "." not in name and name in metadata:
                return deepcopy(metadata[name])
            if "." in name and name in captures:
                return deepcopy(captures[name])
            raise CfgPreconditionError(
                CfgPreconditionReason.CAPTURE_UNAVAILABLE,
                f"Capture source {name!r} is unavailable",
            )

        return CfgResolution(
            source_basis,
            evaluate,
            provide_options,
            references,
            read_capture,
            validate_scalar_expr,
        )

    def snapshot_from_state(
        self,
        state: SessionState,
        *,
        captured_values: Mapping[str, object],
    ) -> CfgResolution:
        """Bind the content snapshot to context and device-set provenance.

        The set key disambiguates removal and re-creation when a per-device
        revision starts over. Captures must already be published cache values;
        this method never queries a value provider or live instrument.
        """
        versions = state.version.snapshot()
        basis: SourceBasis = (
            SourceRevision("context", CfgRevision(versions.get("context", 0))),
            SourceRevision(
                DEVICE_SET_VERSION_KEY,
                CfgRevision(versions.get(DEVICE_SET_VERSION_KEY, 0)),
            ),
            *(
                SourceRevision(
                    f"device:{device.name}",
                    CfgRevision(versions.get(f"device:{device.name}", 0)),
                )
                for device in state.list_devices()
            ),
        )
        return self.snapshot(basis, captured_values=captured_values)

    def new_draft(self, schema: CfgSchema) -> CfgDraft:
        return CfgDraft(
            schema,
            evaluate_expression=self.evaluate_expression,
            provide_options=self.provide_options,
            references=self,
        )

    def evaluate_expression(self, expression: str) -> int | float | complex:
        return evaluate_scalar_expr(expression, self._host.get_current_md())

    def provide_options(self, source_id: str) -> Sequence[object]:
        if source_id == _DEVICES_SOURCE:
            return self._host.list_device_names()
        if source_id == _ARB_WAVEFORMS_SOURCE:
            return self._host.list_arb_waveforms()
        raise RuntimeError(f"Unsupported measure cfg option source {source_id!r}")

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return _reference_keys(self._host.get_current_ml(), kind, allowed_labels)

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return _resolve_reference(self._host.get_current_ml(), kind, key)

    def resolve_value_ref(self, ref: ValueRef, target_type: type) -> DirectValue:
        try:
            target_type_name = name_from_type(target_type)  # type: ignore[arg-type]
        except AssertionError as exc:
            raise ValueTypeError(
                ref.key,
                f"Value source {ref.key!r} cannot target unsupported scalar field type "
                f"{target_type.__name__!r}; only int, float, str, and bool fields are supported",
            ) from exc
        if ref.type_name is not None and ref.type_name != target_type_name:
            raise ValueTypeError(
                ref.key,
                f"Value source {ref.key!r} requested as {ref.type_name!r} but "
                f"target field expects {target_type_name!r}",
            )
        _, value = self._host.read_value_source(ref.key, target_type_name)
        return DirectValue(value)


class _SnapshotReferences:
    def __init__(self, library: ModuleLibrary) -> None:
        self._library = library

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return _reference_keys(self._library, kind, allowed_labels)

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return _resolve_reference(self._library, kind, key)


def _reference_store(
    library: ModuleLibrary, kind: str
) -> tuple[Mapping[str, object], ProgramCfgKind]:
    if kind == "module":
        return library.modules, "module"
    if kind == "waveform":
        return library.waveforms, "waveform"
    raise RuntimeError(f"Unsupported measure cfg reference kind {kind!r}")


def _reference_keys(
    library: ModuleLibrary, kind: str, allowed_labels: frozenset[str]
) -> tuple[str, ...]:
    store, catalog_kind = _reference_store(library, kind)
    return tuple(
        sorted(
            key
            for key, value in store.items()
            if program_shape_for_input(catalog_kind, value).label in allowed_labels
        )
    )


def _resolve_reference(
    library: ModuleLibrary, kind: str, key: str
) -> ResolvedReference | None:
    store, _ = _reference_store(library, kind)
    if key not in store:
        return None
    converter = cast(
        _ReferenceConverter,
        module_cfg_to_value if kind == "module" else waveform_cfg_to_value,
    )
    spec, value = converter(deepcopy(store[key]))
    return ResolvedReference(label=spec.label, value=value)
