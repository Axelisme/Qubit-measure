from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Protocol

from zcu_tools.experiment.cfg_editing import program_shape_for_input
from zcu_tools.gui.cfg import (
    CfgSchema,
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
from zcu_tools.gui.session.types import ParameterContext, RoleBinding
from zcu_tools.gui.session.value_lookup import (
    ScalarValue as LookupScalarValue,
)
from zcu_tools.gui.session.value_lookup import (
    ValueInfo,
    ValueRef,
    ValueTypeError,
    name_from_type,
)
from zcu_tools.resources.context.library import LibrarySnapshot

from .arb_waveform import ArbWaveformKeys
from .cfg_schemas import module_cfg_to_value, waveform_cfg_to_value

_DEVICES_SOURCE = "devices"
_ARB_WAVEFORMS_SOURCE = "arb_waveforms"


class MeasureCfgBindingHost(Protocol):
    def read_context(self, *, roles: RoleBinding | None = None) -> ParameterContext:
        """Return the published fixed point/library capture; no hardware reads."""
        ...

    def list_device_names(self) -> list[str]: ...

    @property
    def arb_waveforms(self) -> ArbWaveformKeys: ...

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
        parameters: ParameterContext | None,
    ) -> CfgResolution:
        """Freeze published local data on the owner sequence, never read hardware.

        The caller supplies matching provenance and cached dotted capture values.
        Bare capture names use the same metadata snapshot as dynamic expressions.
        No live provider is retained by the returned resolution.
        """
        references = _SnapshotReferences(parameters)
        options = {
            _DEVICES_SOURCE: tuple(self._host.list_device_names()),
            _ARB_WAVEFORMS_SOURCE: tuple(self._host.arb_waveforms.list_data_keys()),
        }
        captures = deepcopy(dict(captured_values))

        def evaluate(expression: str) -> int | float | complex:
            return evaluate_scalar_expr(expression, parameters)

        def provide_options(source_id: str) -> Sequence[object]:
            if source_id not in options:
                raise RuntimeError(
                    f"Unsupported measure cfg option source {source_id!r}"
                )
            return options[source_id]

        def read_capture(name: str) -> object:
            if name in captures:
                return deepcopy(captures[name])
            if parameters is not None:
                captured = parameters.lookup(name)
                if captured.present:
                    return captured.value
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
        parameters: ParameterContext | None,
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
        return self.snapshot(
            basis, captured_values=captured_values, parameters=parameters
        )

    def new_draft(
        self, schema: CfgSchema, *, parameters: ParameterContext | None
    ) -> CfgDraft:
        """Bind an independent editor to this fixed capture, never live selection.

        None allows pure math but parameter/library references fail. Device and
        asset options use the published host catalog. Data-source errors propagate.
        """
        return CfgDraft(
            schema,
            evaluate_expression=lambda expression: evaluate_scalar_expr(
                expression, parameters
            ),
            provide_options=self.provide_options,
            references=_SnapshotReferences(parameters),
        )

    def evaluate_expression(self, expression: str) -> int | float | complex:
        return evaluate_scalar_expr(expression, self._host.read_context())

    def provide_options(self, source_id: str) -> Sequence[object]:
        if source_id == _DEVICES_SOURCE:
            return self._host.list_device_names()
        if source_id == _ARB_WAVEFORMS_SOURCE:
            return self._host.arb_waveforms.list_data_keys()
        raise RuntimeError(f"Unsupported measure cfg option source {source_id!r}")

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        return _reference_keys(self._host.read_context().library, kind, allowed_labels)

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        return _resolve_reference(self._host.read_context(), kind, key)

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
    def __init__(self, parameters: ParameterContext | None) -> None:
        self._parameters = parameters

    def keys(self, kind: str, allowed_labels: frozenset[str]) -> Sequence[str]:
        if self._parameters is None:
            return ()
        return _reference_keys(self._parameters.library, kind, allowed_labels)

    def resolve(self, kind: str, key: str) -> ResolvedReference | None:
        if self._parameters is None:
            return None
        return _resolve_reference(self._parameters, kind, key)


def _reference_keys(
    library: LibrarySnapshot, kind: str, allowed_labels: frozenset[str]
) -> tuple[str, ...]:
    if kind == "module":
        keys = library.module_keys()
        return tuple(
            sorted(
                key
                for key in keys
                if program_shape_for_input(
                    "module", library.entry_value(key, kind="module")
                ).label
                in allowed_labels
            )
        )
    if kind == "waveform":
        keys = library.waveform_keys()
        return tuple(
            sorted(
                key
                for key in keys
                if program_shape_for_input(
                    "waveform", library.entry_value(key, kind="waveform")
                ).label
                in allowed_labels
            )
        )
    raise RuntimeError(f"Unsupported measure cfg reference kind {kind!r}")


def _resolve_reference(
    parameters: ParameterContext, kind: str, key: str
) -> ResolvedReference | None:
    library = parameters.library
    root = parameters.library_key(key)
    if kind == "module":
        if root not in library.module_keys():
            return None
        spec, value = module_cfg_to_value(library.get_module(root))
    elif kind == "waveform":
        if root not in library.waveform_keys():
            return None
        spec, value = waveform_cfg_to_value(library.get_waveform(root))
    else:
        raise RuntimeError(f"Unsupported measure cfg reference kind {kind!r}")
    return ResolvedReference(label=spec.label, value=value)
