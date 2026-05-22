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
    ChannelSpec,
    ChannelValue,
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
from .event_bus import GuiEvent

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


class ScalarLiveField(LiveField):
    spec: ScalarSpec

    def __init__(
        self, spec: ScalarSpec, env: LiveModelEnv, initial_val: object = None
    ) -> None:
        super().__init__(spec, env)

        if isinstance(initial_val, ScalarValue):
            self._value = initial_val.value
            self._is_unset = initial_val.is_unset
        else:
            self._value = initial_val
            self._is_unset = initial_val is None

        if self._is_unset:
            # Type-appropriate zero values, but flag remains is_unset=True
            defaults: dict[type, object] = {int: 0, float: 0.0, bool: False, str: ""}
            self._value = defaults.get(spec.type)
        self._set_valid(not self._is_unset)

    def get_value(self) -> ScalarValue:
        return ScalarValue(value=self._value, is_unset=self._is_unset)

    def set_value(self, val: object) -> None:
        if isinstance(val, ScalarValue):
            new_val = val.value
            new_unset = val.is_unset
        else:
            new_val = val
            new_unset = val is None

        if new_unset:
            defaults: dict[type, object] = {int: 0, float: 0.0, bool: False, str: ""}
            new_val = defaults.get(self.spec.type)

        if new_val != self._value or new_unset != self._is_unset:
            self._value = new_val
            self._is_unset = new_unset
            self.on_change.emit(self.get_value())
        self._set_valid(not self._is_unset)

    def to_dict(self) -> object:
        return self._value


class LiteralLiveField(LiveField):
    spec: LiteralSpec

    def __init__(
        self, spec: LiteralSpec, env: LiveModelEnv, initial_val: object = None
    ) -> None:
        super().__init__(spec, env)

    def get_value(self) -> ScalarValue:
        return ScalarValue(value=self.spec.value, is_unset=False)

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


class ChannelLiveField(LiveField):
    """Reactive Channel field that resolves names via MetaDict from Controller."""

    spec: ChannelSpec

    def __init__(
        self,
        spec: ChannelSpec,
        env: LiveModelEnv,
        initial_val: object = None,
    ) -> None:
        super().__init__(spec, env)

        if isinstance(initial_val, ChannelValue):
            self._chosen = initial_val.chosen
        else:
            self._chosen = initial_val if isinstance(initial_val, (int, str)) else 0

        self._resolved_id: Optional[int] = None
        self.env.bus.subscribe(GuiEvent.MD_CHANGED, self._on_md_changed)
        self.env.bus.subscribe(GuiEvent.CONTEXT_CHANGED, self._on_context_changed)
        self._refresh_resolve()

    def _on_md_changed(self, md: object) -> None:
        self._refresh_resolve()

    def _on_context_changed(self, md: object, ml: object) -> None:
        self._refresh_resolve()

    def _refresh_resolve(self) -> None:
        from .ui.fields.utils import _resolve_channel

        md = self.env.ctrl.get_current_md()
        new_id = _resolve_channel(str(self._chosen), md)
        if new_id != self._resolved_id:
            self._resolved_id = new_id
            self._set_valid(new_id is not None)
            self.on_change.emit(self.get_value())

    def get_value(self) -> ChannelValue:
        return ChannelValue(chosen=self._chosen, resolved=self._resolved_id)

    def set_value(self, val: object) -> None:
        if isinstance(val, ChannelValue):
            new_chosen = val.chosen
        elif isinstance(val, (int, str)):
            new_chosen = val
        else:
            return

        if new_chosen != self._chosen:
            self._chosen = new_chosen
            self._refresh_resolve()
            self.on_change.emit(self.get_value())

    def to_dict(self) -> Optional[int]:
        return self._resolved_id

    def teardown(self) -> None:
        self.env.bus.unsubscribe(GuiEvent.MD_CHANGED, self._on_md_changed)
        self.env.bus.unsubscribe(GuiEvent.CONTEXT_CHANGED, self._on_context_changed)


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
            f.teardown()


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

        self.sub_field: Optional[SectionLiveField] = None
        self._rebuild_sub_field()

    def _rebuild_sub_field(self) -> None:
        old_spec = self.sub_field.spec if self.sub_field else None
        old_val = self.sub_field.get_value() if self.sub_field else None
        if self.sub_field:
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
        self.on_change.emit(self.get_value())

    def _on_sub_validity_change(self, *_: object) -> None:
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        valid = self.sub_field.is_valid() if self.sub_field else True
        self._set_valid(valid)

    def get_chosen_key(self) -> str:
        return self._chosen_key

    def set_chosen_key(self, key: str) -> None:
        if key != self._chosen_key:
            self._chosen_key = key
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
            self.sub_field.teardown()


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
    if isinstance(spec, ChannelSpec):
        return ChannelLiveField(spec, env, initial_val)
    if isinstance(spec, (ModuleRefSpec, WaveformRefSpec)):
        return ModuleRefLiveField(spec, env, initial_val)
    if isinstance(spec, CfgSectionSpec):
        return SectionLiveField(
            spec,
            env,
            initial_val if isinstance(initial_val, CfgSectionValue) else None,
        )

    raise TypeError(f"Unknown spec type: {type(spec)}")
