"""LiveModel — reactive data layer for CfgSchema.

Uses LiveModelEnv for dependency injection, fetches md/ml through the
controller boundary, and tracks unset Scalar fields explicitly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, Optional, Protocol, Union, cast

from .adapter import (
    CfgNodeSpec,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
    DeviceRefSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    MultiSweepSpec,
    MultiSweepValue,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    default_value_for_type,
)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    from .event_bus import EventBus

logger = logging.getLogger(__name__)


class ControllerProtocol(Protocol):
    """Minimal interface needed by LiveFields to fetch environment state."""

    def get_bus(self) -> EventBus: ...
    def get_current_md(self) -> MetaDict: ...
    def get_current_ml(self) -> ModuleLibrary: ...
    def has_soc(self) -> bool: ...
    def list_device_names(self) -> list[str]: ...


@dataclass(frozen=True)
class LiveModelEnv:
    """Environment container for LiveFields."""

    ctrl: ControllerProtocol

    @property
    def bus(self) -> EventBus:
        return self.ctrl.get_bus()


class CallbackList:
    """Simple callback container for reactivity."""

    def __init__(self) -> None:
        self._cbs: list[Callable[..., None]] = []

    def connect(self, cb: Callable[..., None]) -> None:
        if cb not in self._cbs:
            self._cbs.append(cb)

    def disconnect(self, cb: Callable[..., None]) -> None:
        try:
            self._cbs.remove(cb)
        except ValueError:
            pass

    def clear(self) -> None:
        self._cbs.clear()

    def emit(self, *args: object, **kwargs: object) -> None:
        for cb in list(self._cbs):
            cb(*args, **kwargs)


class LiveField(ABC):
    """Base class for a reactive field."""

    spec: CfgNodeSpec

    def __init__(self, spec: CfgNodeSpec, env: LiveModelEnv) -> None:
        self.spec = spec
        self.env = env
        self.on_change = CallbackList()
        self.on_validity_changed = CallbackList()
        self._valid = True

    @abstractmethod
    def get_value(self) -> object:
        """Return the current value (as a CfgNodeValue or subtype)."""
        ...

    @abstractmethod
    def set_value(self, val: object) -> None:
        """Update the current value and emit on_change."""
        ...

    def is_valid(self) -> bool:
        return self._valid

    def _set_valid(self, valid: bool) -> None:
        if valid != self._valid:
            self._valid = valid
            logger.debug(
                "%s._set_valid: spec=%r valid=%r",
                type(self).__name__,
                getattr(self.spec, "label", None) or type(self.spec).__name__,
                valid,
            )
            self.on_validity_changed.emit(valid)

    def teardown(self) -> None:
        """Cleanup subscriptions."""
        pass

    def refresh_external(self, event: object) -> None:
        """Refresh values derived from external context."""
        del event


class ScalarLiveField(LiveField):
    spec: ScalarSpec

    def __init__(
        self, spec: ScalarSpec, env: LiveModelEnv, initial_val: object = None
    ) -> None:
        super().__init__(spec, env)

        if isinstance(initial_val, (DirectValue, EvalValue)):
            self._value: ScalarValue = initial_val
        else:
            self._value = self._make_direct_value(initial_val, initial_val is None)

        if isinstance(self._value, EvalValue):
            self._resolve_expression(emit_change=False)
        self._refresh_validity()

    def get_value(self) -> ScalarValue:
        return self._value

    def set_value(self, val: object) -> None:
        if isinstance(val, (DirectValue, EvalValue)):
            new_value = val
        else:
            new_value = self._make_direct_value(val, val is None)

        if isinstance(new_value, EvalValue):
            new_value = self._resolved_eval_value(new_value)

        if new_value != self._value:
            self._value = new_value
            self._refresh_validity()
            self.on_change.emit(self.get_value())
        else:
            self._refresh_validity()

    def refresh_external(self, event: object) -> None:
        del event
        if isinstance(self._value, EvalValue):
            self._resolve_expression(emit_change=True)

    def _make_direct_value(self, value: object, is_unset: bool) -> DirectValue:
        if is_unset:
            value = default_value_for_type(self.spec.type)
        return DirectValue(value=value, is_unset=is_unset)

    def _resolved_eval_value(self, value: EvalValue) -> EvalValue:
        from dataclasses import replace

        from .expression import coerce_eval_result, evaluate_numeric_expr

        try:
            resolved = coerce_eval_result(
                evaluate_numeric_expr(value.expr, self.env.ctrl.get_current_md()),
                self.spec.type,
            )
        except Exception as exc:
            return replace(value, resolved=None, error=str(exc))
        return replace(value, resolved=resolved, error=None)

    def _resolve_expression(self, *, emit_change: bool) -> None:
        assert isinstance(self._value, EvalValue)
        new_value = self._resolved_eval_value(self._value)
        if new_value != self._value:
            self._value = new_value
            self._refresh_validity()
            if emit_change:
                self.on_change.emit(self.get_value())
        else:
            self._refresh_validity()

    def _refresh_validity(self) -> None:
        if isinstance(self._value, DirectValue):
            valid = not self._value.is_unset
            if not valid:
                logger.debug(
                    "ScalarLiveField: label=%r is unset → invalid", self.spec.label
                )
            self._set_valid(valid)
        else:
            valid = self._value.resolved is not None
            if not valid:
                logger.debug(
                    "ScalarLiveField: label=%r expr=%r unresolved (error=%r) → invalid",
                    self.spec.label,
                    self._value.expr,
                    self._value.error,
                )
            self._set_valid(valid)


class LiteralLiveField(LiveField):
    spec: LiteralSpec

    def __init__(
        self, spec: LiteralSpec, env: LiveModelEnv, initial_val: object = None
    ) -> None:
        super().__init__(spec, env)

    def get_value(self) -> ScalarValue:
        return DirectValue(value=self.spec.value, is_unset=False)

    def set_value(self, val: object) -> None:
        pass


class SweepLiveField(LiveField):
    spec: SweepSpec

    def __init__(
        self, spec: SweepSpec, env: LiveModelEnv, initial_val: object = None
    ) -> None:
        super().__init__(spec, env)

        if isinstance(initial_val, SweepValue):
            self._value = initial_val
        else:
            self._value = SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)

    def get_value(self) -> SweepValue:
        return self._value

    def set_value(self, val: object) -> None:
        if isinstance(val, SweepValue):
            self._value = val
            self.on_change.emit(val)
            return
        raise TypeError(f"SweepLiveField expects SweepValue, got {type(val).__name__}")


class MultiSweepLiveField(LiveField):
    spec: MultiSweepSpec

    def __init__(
        self, spec: MultiSweepSpec, env: LiveModelEnv, initial_val: object = None
    ) -> None:
        super().__init__(spec, env)

        self.fields: Dict[str, SweepLiveField] = {}

        initial_axes = {}
        if isinstance(initial_val, MultiSweepValue):
            initial_axes = initial_val.axes

        for axis, axis_spec in spec.axes.items():
            sv = initial_axes.get(axis, SweepValue(0.0, 1.0, 11, 0.1))
            field = SweepLiveField(axis_spec, env, initial_val=sv)
            self.fields[axis] = field
            field.on_change.connect(self._on_child_change)

    def _on_child_change(self, *_: object) -> None:
        self.on_change.emit(self.get_value())

    def get_value(self) -> MultiSweepValue:
        return MultiSweepValue(axes={k: f.get_value() for k, f in self.fields.items()})

    def set_value(self, val: object) -> None:
        if not isinstance(val, MultiSweepValue):
            raise TypeError(
                f"MultiSweepLiveField expects MultiSweepValue, got {type(val).__name__}"
            )
        for k, field in self.fields.items():
            if k in val.axes:
                field.set_value(val.axes[k])


class SectionLiveField(LiveField):
    """Container for a group of fields."""

    spec: CfgSectionSpec

    def __init__(
        self,
        spec: CfgSectionSpec,
        env: LiveModelEnv,
        initial_val: Optional[CfgSectionValue] = None,
    ) -> None:
        super().__init__(spec, env)
        self.fields: dict[str, LiveField] = {}

        from .adapter import make_default_value

        default_val = make_default_value(spec)
        provided_val = initial_val if initial_val is not None else default_val

        # Build child fields; fall back to spec default for keys missing from provided_val.
        # Optional ModuleRef/WaveformRef missing from provided_val → pass None so the
        # field initialises as disabled (is_enabled=False).
        for key, node_spec in spec.fields.items():
            child_val = provided_val.fields.get(key)
            if child_val is None:
                if (
                    isinstance(node_spec, (ModuleRefSpec, WaveformRefSpec))
                    and node_spec.optional
                ):
                    child_val = None  # intentionally disabled
                else:
                    child_val = default_val.fields.get(key)
            field = create_live_field(node_spec, env, child_val)
            self.fields[key] = field
            field.on_change.connect(self._on_child_change)
            field.on_validity_changed.connect(self._on_child_validity_change)

        self._refresh_validity()

    def _on_child_change(self, *_: object) -> None:
        self.on_change.emit(self.get_value())

    def _on_child_validity_change(self, *_: object) -> None:
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        self._set_valid(all(f.is_valid() for f in self.fields.values()))

    def get_value(self) -> CfgSectionValue:
        fields: dict[str, CfgNodeValue] = {}
        for k, f in self.fields.items():
            if (
                isinstance(f, ModuleRefLiveField)
                and f.spec.optional
                and not f.is_enabled
            ):
                continue  # disabled optional ModuleRef → omit key
            fields[k] = cast(CfgNodeValue, f.get_value())
        return CfgSectionValue(fields=fields)

    def set_value(self, val: object) -> None:
        if not isinstance(val, CfgSectionValue):
            raise TypeError(
                f"SectionLiveField expects CfgSectionValue, got {type(val).__name__}"
            )
        for k, field in self.fields.items():
            if k in val.fields:
                field.set_value(val.fields[k])

    def teardown(self) -> None:
        for f in self.fields.values():
            f.on_change.disconnect(self._on_child_change)
            f.on_validity_changed.disconnect(self._on_child_validity_change)
            f.teardown()

    def refresh_external(self, event: object) -> None:
        for f in self.fields.values():
            f.refresh_external(event)
        self._refresh_validity()


class DeviceRefLiveField(LiveField):
    """Reactive field for selecting a registered device by name."""

    spec: DeviceRefSpec

    def __init__(
        self, spec: DeviceRefSpec, env: LiveModelEnv, initial_val: object = None
    ) -> None:
        super().__init__(spec, env)
        chosen = initial_val.value if isinstance(initial_val, DirectValue) else None
        self._chosen_name: str = chosen if isinstance(chosen, str) else ""
        self._refresh_validity()

    def get_value(self) -> DirectValue:
        return DirectValue(self._chosen_name)

    def set_value(self, val: object) -> None:
        if isinstance(val, DirectValue) and isinstance(val.value, str):
            self._chosen_name = val.value
        elif isinstance(val, str):
            self._chosen_name = val
        else:
            raise TypeError(
                f"DeviceRefLiveField expects str or DirectValue(str), got {type(val).__name__}"
            )
        self._refresh_validity()
        self.on_change.emit(self.get_value())

    def get_chosen_name(self) -> str:
        return self._chosen_name

    def set_chosen_name(self, name: str) -> None:
        if name != self._chosen_name:
            self._chosen_name = name
            self._refresh_validity()
            self.on_change.emit(self.get_value())

    def _refresh_validity(self) -> None:
        names = self.env.ctrl.list_device_names()
        self._set_valid(self._chosen_name in names)

    def refresh_external(self, event: object) -> None:
        from .event_bus import GuiEvent

        if event is GuiEvent.DEVICE_CHANGED:
            self._refresh_validity()
            self.on_change.emit(self.get_value())

    def teardown(self) -> None:
        pass


class LibraryBindingState(Enum):
    LINKED = "linked"
    MODIFIED = "modified"
    CUSTOM = "custom"


def _binding_state_for_key(chosen_key: str) -> LibraryBindingState:
    if chosen_key.startswith("<Custom:"):
        return LibraryBindingState.CUSTOM
    return LibraryBindingState.LINKED


class ModuleRefLiveField(LiveField):
    """Reactive field for Module/Waveform references with dynamic sub-sections."""

    spec: Union[ModuleRefSpec, WaveformRefSpec]

    def __init__(
        self,
        spec: Union[ModuleRefSpec, WaveformRefSpec],
        env: LiveModelEnv,
        initial_val: object = None,
    ) -> None:
        super().__init__(spec, env)

        if isinstance(initial_val, (ModuleRefValue, WaveformRefValue)):
            self._chosen_key = initial_val.chosen_key
            init_sub: Optional[CfgSectionValue] = initial_val.value
        else:
            # Default to first allowed
            self._chosen_key = (
                f"<Custom:{spec.allowed[0].label}>" if spec.allowed else ""
            )
            init_sub = None

        self._binding_state = _binding_state_for_key(self._chosen_key)
        self.sub_field: Optional[SectionLiveField] = None
        self.is_enabled: bool = not (spec.optional and initial_val is None)
        self.on_enabled_changed = CallbackList()
        self._rebuild_sub_field(hint=init_sub)

    def is_modified(self) -> bool:
        return self._binding_state is LibraryBindingState.MODIFIED

    def _rebuild_sub_field(self, hint: Optional[CfgSectionValue] = None) -> None:
        """Rebuild the sub-field for the current chosen_key.

        hint: explicit initial CfgSectionValue to seed the sub-field (takes
        priority over both library value and inherit_from inheritance).
        """
        old_spec = self.sub_field.spec if self.sub_field else None
        old_val = self.sub_field.get_value() if self.sub_field else None
        if self.sub_field:
            self.sub_field.on_change.disconnect(self._on_sub_change)
            self.sub_field.on_validity_changed.disconnect(self._on_sub_validity_change)
            self.sub_field.teardown()

        from .ui.fields.utils import _spec_value_for_chosen

        ml = self.env.ctrl.get_current_ml()
        chosen_spec, lib_val = _spec_value_for_chosen(
            self._chosen_key, self.spec.allowed, ml
        )
        if chosen_spec:
            if hint is not None:
                val: Optional[CfgSectionValue] = hint
            elif lib_val is not None:
                val = lib_val
            elif isinstance(old_spec, CfgSectionSpec) and isinstance(
                old_val, CfgSectionValue
            ):
                from .adapter import inherit_from

                val = inherit_from(old_val, old_spec, chosen_spec)
            else:
                val = None
            self.sub_field = SectionLiveField(chosen_spec, self.env, val)
            self.sub_field.on_change.connect(self._on_sub_change)
            self.sub_field.on_validity_changed.connect(self._on_sub_validity_change)
        else:
            self.sub_field = None

        self._refresh_validity()

    def _on_sub_change(self, *_: object) -> None:
        if self._binding_state is LibraryBindingState.LINKED:
            self._binding_state = LibraryBindingState.MODIFIED
        self.on_change.emit(self.get_value())

    def _on_sub_validity_change(self, *_: object) -> None:
        self._refresh_validity()

    def set_enabled(self, enabled: bool) -> None:
        if not self.spec.optional:
            return
        if enabled != self.is_enabled:
            self.is_enabled = enabled
            self._refresh_validity()
            self.on_enabled_changed.emit(enabled)
            self.on_change.emit(self.get_value())

    def _refresh_validity(self) -> None:
        if self.spec.optional and not self.is_enabled:
            self._set_valid(True)
            return
        if self.sub_field is None:
            logger.debug(
                "ModuleRefLiveField._refresh_validity: key=%r sub_field=None → valid=True",
                self._chosen_key,
            )
            self._set_valid(True)
        else:
            valid = self.sub_field.is_valid()
            if not valid:
                logger.debug(
                    "ModuleRefLiveField._refresh_validity: key=%r sub_field invalid",
                    self._chosen_key,
                )
            self._set_valid(valid)

    def _refresh_library_binding(self) -> None:
        if self._binding_state is not LibraryBindingState.LINKED:
            return
        self._rebuild_sub_field()
        self.on_change.emit(self.get_value())

    def get_chosen_key(self) -> str:
        return self._chosen_key

    def set_chosen_key(self, key: str) -> None:
        if key != self._chosen_key or self.is_modified():
            self._chosen_key = key
            self._binding_state = _binding_state_for_key(key)
            self._rebuild_sub_field(hint=None)
            self.on_change.emit(self.get_value())

    def get_value(self) -> Union[ModuleRefValue, WaveformRefValue]:
        klass = (
            ModuleRefValue if isinstance(self.spec, ModuleRefSpec) else WaveformRefValue
        )
        sub_val = self.sub_field.get_value() if self.sub_field else CfgSectionValue()
        return klass(chosen_key=self._chosen_key, value=sub_val)

    def set_value(self, val: object) -> None:
        if not isinstance(val, (ModuleRefValue, WaveformRefValue)):
            raise TypeError(
                "ModuleRefLiveField expects ModuleRefValue or WaveformRefValue, "
                f"got {type(val).__name__}"
            )
        if val.chosen_key != self._chosen_key or self.is_modified():
            self._chosen_key = val.chosen_key
            self._binding_state = _binding_state_for_key(val.chosen_key)
            self._rebuild_sub_field(hint=val.value)
        elif self.sub_field:
            self.sub_field.set_value(val.value)
        self.on_change.emit(self.get_value())

    def teardown(self) -> None:
        if self.spec.optional:
            self.on_enabled_changed.clear()
        if self.sub_field:
            self.sub_field.on_change.disconnect(self._on_sub_change)
            self.sub_field.on_validity_changed.disconnect(self._on_sub_validity_change)
            self.sub_field.teardown()

    def refresh_external(self, event: object) -> None:
        from .event_bus import GuiEvent

        if event in {GuiEvent.CONTEXT_SWITCHED, GuiEvent.ML_CHANGED}:
            self._refresh_library_binding()
            if self._binding_state is LibraryBindingState.CUSTOM and self.sub_field:
                self.sub_field.refresh_external(event)
                self._refresh_validity()
            return
        if self.sub_field:
            self.sub_field.refresh_external(event)
            self._refresh_validity()


def create_live_field(
    spec: CfgNodeSpec,
    env: LiveModelEnv,
    initial_val: object = None,
) -> LiveField:
    """Factory to create the appropriate LiveField from a Spec."""
    if isinstance(spec, ScalarSpec):
        return ScalarLiveField(spec, env, initial_val)
    if isinstance(spec, LiteralSpec):
        return LiteralLiveField(spec, env, initial_val)
    if isinstance(spec, SweepSpec):
        return SweepLiveField(spec, env, initial_val)
    if isinstance(spec, MultiSweepSpec):
        return MultiSweepLiveField(spec, env, initial_val)
    if isinstance(spec, (ModuleRefSpec, WaveformRefSpec)):
        return ModuleRefLiveField(spec, env, initial_val)
    if isinstance(spec, DeviceRefSpec):
        return DeviceRefLiveField(spec, env, initial_val)
    if isinstance(spec, CfgSectionSpec):
        return SectionLiveField(
            spec,
            env,
            initial_val if isinstance(initial_val, CfgSectionValue) else None,
        )

    raise TypeError(f"Unknown spec type: {type(spec)}")
