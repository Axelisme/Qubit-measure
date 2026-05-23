"""LiveModel — Reactive data layer for CfgSchema.

REFACTORED (Phase 36/36.5):
- Uses LiveModelEnv for dependency injection (SSOT enforcement).
- Fields dynamically fetch md/ml from ControllerProtocol.
- Supports 'is_unset' flag for Scalar fields to distinguish missing data.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Protocol, Union, cast

from .adapter import (
    CfgNodeSpec,
    CfgNodeValue,
    CfgSectionSpec,
    CfgSectionValue,
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
    def is_running(self) -> bool: ...
    def has_soc(self) -> bool: ...


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

    def emit(self, *args: object, **kwargs: object) -> None:
        for cb in list(self._cbs):
            try:
                cb(*args, **kwargs)
            except Exception:
                logger.warning("CallbackList: error in callback", exc_info=True)


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

    @abstractmethod
    def to_dict(self) -> object:
        """Return the value in a format suitable for experiment config dicts."""
        ...

    def is_valid(self) -> bool:
        return self._valid

    def _set_valid(self, valid: bool) -> None:
        if valid != self._valid:
            self._valid = valid
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

    def to_dict(self) -> object:
        if isinstance(self._value, DirectValue):
            return self._value.value
        return self._value.resolved

    def refresh_external(self, event: object) -> None:
        del event
        if isinstance(self._value, EvalValue):
            self._resolve_expression(emit_change=True)

    def _make_direct_value(self, value: object, is_unset: bool) -> DirectValue:
        if is_unset:
            defaults: dict[type, object] = {int: 0, float: 0.0, bool: False, str: ""}
            value = defaults.get(self.spec.type)
        return DirectValue(value=value, is_unset=is_unset)

    def _resolved_eval_value(self, value: EvalValue) -> EvalValue:
        from dataclasses import replace

        from .expression import coerce_eval_result, evaluate_numeric_expr

        try:
            resolved = coerce_eval_result(
                evaluate_numeric_expr(value.expr, self.env.ctrl.get_current_md()),
                self.spec.type,
            )
        except Exception:
            resolved = None
        return replace(value, resolved=resolved)

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
            self._set_valid(not self._value.is_unset)
        else:
            self._set_valid(self._value.resolved is not None)


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

    def to_dict(self) -> object:
        return self.spec.value


class SweepLiveField(LiveField):
    spec: SweepSpec

    def __init__(
        self, spec: SweepSpec, env: LiveModelEnv, initial_val: object = None
    ) -> None:
        super().__init__(spec, env)

        if isinstance(initial_val, SweepValue):
            self._value = initial_val
        else:
            self._value = SweepValue(start=0.0, stop=1.0, expts=11)

    def get_value(self) -> SweepValue:
        return self._value

    def set_value(self, val: object) -> None:
        if isinstance(val, SweepValue):
            self._value = val
            self.on_change.emit(val)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "start": self._value.start,
            "stop": self._value.stop,
            "expts": self._value.expts,
            "step": self._value.step,
        }


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
            sv = initial_axes.get(axis, SweepValue(0.0, 1.0, 11))
            field = SweepLiveField(axis_spec, env, initial_val=sv)
            self.fields[axis] = field
            field.on_change.connect(self._on_child_change)

    def _on_child_change(self, *_: object) -> None:
        self.on_change.emit(self.get_value())

    def get_value(self) -> MultiSweepValue:
        return MultiSweepValue(axes={k: f.get_value() for k, f in self.fields.items()})

    def set_value(self, val: object) -> None:
        if isinstance(val, MultiSweepValue):
            for k, field in self.fields.items():
                if k in val.axes:
                    field.set_value(val.axes[k])

    def to_dict(self) -> dict[str, dict[str, float | int | None]]:
        return {k: f.to_dict() for k, f in self.fields.items()}


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

        # If no initial value provided, use Spec defaults to avoid is_unset=True for all children
        from .adapter import make_default_value

        val = initial_val if initial_val is not None else make_default_value(spec)

        # Build child fields
        for key, node_spec in spec.fields.items():
            child_val = val.fields.get(key)
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
        return CfgSectionValue(
            fields={
                k: cast(CfgNodeValue, f.get_value()) for k, f in self.fields.items()
            }
        )

    def set_value(self, val: object) -> None:
        # val should be CfgSectionValue
        if isinstance(val, CfgSectionValue):
            for k, field in self.fields.items():
                if k in val.fields:
                    field.set_value(val.fields[k])

    def to_dict(self) -> dict[str, object]:
        return {k: f.to_dict() for k, f in self.fields.items()}

    def teardown(self) -> None:
        for f in self.fields.values():
            f.on_change.disconnect(self._on_child_change)
            f.on_validity_changed.disconnect(self._on_child_validity_change)
            f.teardown()

    def refresh_external(self, event: object) -> None:
        for f in self.fields.values():
            f.refresh_external(event)
        self._refresh_validity()


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
            self._sub_value = initial_val.value
        else:
            # Default to first allowed
            self._chosen_key = (
                f"<Custom:{spec.allowed[0].label}>" if spec.allowed else ""
            )
            self._sub_value = None

        self._is_modified = False
        self.sub_field: Optional[SectionLiveField] = None
        self._rebuild_sub_field()

    def is_modified(self) -> bool:
        return self._is_modified

    def _rebuild_sub_field(self) -> None:
        old_spec = self.sub_field.spec if self.sub_field else None
        old_val = self.sub_field.get_value() if self.sub_field else None
        if self.sub_field:
            self.sub_field.on_change.disconnect(self._on_sub_change)
            self.sub_field.on_validity_changed.disconnect(self._on_sub_validity_change)
            self.sub_field.teardown()

        from .ui.fields.utils import _spec_value_for_chosen

        ml = self.env.ctrl.get_current_ml()
        chosen_spec, initial_val = _spec_value_for_chosen(
            self._chosen_key, self.spec.allowed, ml
        )
        if chosen_spec:
            # Use initial_val from library if available, otherwise fallback to self._sub_value
            if initial_val is not None:
                val = initial_val
            elif isinstance(old_spec, CfgSectionSpec) and isinstance(
                old_val, CfgSectionValue
            ):
                from .adapter import inherit_from

                val = inherit_from(old_val, old_spec, chosen_spec)
            else:
                val = self._sub_value
            self.sub_field = SectionLiveField(chosen_spec, self.env, val)
            self.sub_field.on_change.connect(self._on_sub_change)
            self.sub_field.on_validity_changed.connect(self._on_sub_validity_change)
        else:
            self.sub_field = None

        self._refresh_validity()

    def _on_sub_change(self, *_: object) -> None:
        if not self._chosen_key.startswith("<Custom:"):
            self._is_modified = True
        self.on_change.emit(self.get_value())

    def _on_sub_validity_change(self, *_: object) -> None:
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        valid = self.sub_field.is_valid() if self.sub_field else True
        self._set_valid(valid)

    def _refresh_library_binding(self) -> None:
        if self._chosen_key.startswith("<Custom:"):
            return
        if self._is_modified:
            # Do not overwrite user-modified fields
            return
        self._sub_value = (
            self.sub_field.get_value() if self.sub_field else self._sub_value
        )
        self._rebuild_sub_field()
        self.on_change.emit(self.get_value())

    def get_chosen_key(self) -> str:
        return self._chosen_key

    def set_chosen_key(self, key: str) -> None:
        if key != self._chosen_key:
            self._chosen_key = key
            self._is_modified = False
            self._sub_value = (
                self.sub_field.get_value() if self.sub_field is not None else None
            )
            self._rebuild_sub_field()
            self.on_change.emit(self.get_value())

    def get_value(self) -> Union[ModuleRefValue, WaveformRefValue]:
        klass = (
            ModuleRefValue if isinstance(self.spec, ModuleRefSpec) else WaveformRefValue
        )
        sub_val = self.sub_field.get_value() if self.sub_field else CfgSectionValue()
        return klass(chosen_key=self._chosen_key, value=sub_val)

    def set_value(self, val: object) -> None:
        if not isinstance(val, (ModuleRefValue, WaveformRefValue)):
            return
        if val.chosen_key != self._chosen_key:
            self._chosen_key = val.chosen_key
            self._is_modified = False
            self._sub_value = val.value
            self._rebuild_sub_field()
        elif self.sub_field:
            self.sub_field.set_value(val.value)
        self.on_change.emit(self.get_value())

    def to_dict(self) -> dict[str, object]:
        if self.sub_field:
            return self.sub_field.to_dict()
        return {}

    def teardown(self) -> None:
        if self.sub_field:
            self.sub_field.on_change.disconnect(self._on_sub_change)
            self.sub_field.on_validity_changed.disconnect(self._on_sub_validity_change)
            self.sub_field.teardown()

    def refresh_external(self, event: object) -> None:
        from .event_bus import GuiEvent

        if event in {GuiEvent.CONTEXT_CHANGED, GuiEvent.INSPECT_CHANGED}:
            self._refresh_library_binding()
            if self._chosen_key.startswith("<Custom:") and self.sub_field:
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
    if isinstance(spec, CfgSectionSpec):
        return SectionLiveField(
            spec,
            env,
            initial_val if isinstance(initial_val, CfgSectionValue) else None,
        )

    raise TypeError(f"Unknown spec type: {type(spec)}")
