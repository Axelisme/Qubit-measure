"""LiveModel — Reactive data layer for CfgSchema.

This layer sits between Spec (static definition) and Widget (UI rendering).
It holds the active state, handles validation, and emits signals on change.
No Qt dependency; pure Python.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from .event_bus import GuiEvent

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict
    from .adapter import (
        CfgNodeSpec,
        CfgSectionSpec,
        CfgSectionValue,
        ChannelSpec,
        ModuleRefSpec,
        MultiSweepSpec,
        ScalarSpec,
        SweepSpec,
        WaveformRefSpec,
    )
    from .event_bus import EventBus

logger = logging.getLogger(__name__)


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

    def emit(self, *args: Any, **kwargs: Any) -> None:
        for cb in list(self._cbs):
            try:
                cb(*args, **kwargs)
            except Exception:
                logger.warning("CallbackList: error in callback", exc_info=True)


class LiveField(ABC):
    """Base class for a reactive field."""

    def __init__(self, spec: Any) -> None:
        self.spec = spec
        self.on_change = CallbackList()
        self.on_validity_changed = CallbackList()
        self._valid = True

    @abstractmethod
    def get_value(self) -> Any:
        """Return the current value (as a Python object)."""
        ...

    @abstractmethod
    def set_value(self, val: Any) -> None:
        """Update the current value and emit on_change."""
        ...

    @abstractmethod
    def to_dict(self) -> Any:
        """Return the value in a format suitable for experiment config dicts."""
        ...

    def is_valid(self) -> bool:
        return self._valid

    def _set_valid(self, valid: bool) -> None:
        if valid != self._valid:
            self._valid = valid
            self.on_validity_changed.emit(valid)


class ScalarLiveField(LiveField):
    def __init__(self, spec: ScalarSpec, initial_val: Any = None) -> None:
        super().__init__(spec)
        from .adapter import ScalarValue

        if isinstance(initial_val, ScalarValue):
            self._value = initial_val.value
        else:
            self._value = initial_val

    def get_value(self) -> Any:
        from .adapter import ScalarValue

        return ScalarValue(value=self._value)

    def set_value(self, val: Any) -> None:
        from .adapter import ScalarValue

        new_val = val.value if isinstance(val, ScalarValue) else val
        if new_val != self._value:
            self._value = new_val
            self.on_change.emit(self.get_value())

    def to_dict(self) -> Any:
        return self._value


class LiteralLiveField(LiveField):
    def __init__(self, spec: LiteralSpec, initial_val: Any = None) -> None:
        super().__init__(spec)

    def get_value(self) -> Any:
        from .adapter import ScalarValue

        return ScalarValue(value=self.spec.value)

    def set_value(self, val: Any) -> None:
        # Literal values cannot be changed
        pass

    def to_dict(self) -> Any:
        return self.spec.value


class SweepLiveField(LiveField):
    def __init__(self, spec: SweepSpec, initial_val: Any = None) -> None:
        super().__init__(spec)
        from .adapter import SweepValue

        if isinstance(initial_val, SweepValue):
            self._value = initial_val
        else:
            self._value = SweepValue(start=0.0, stop=1.0, expts=11)

    def get_value(self) -> Any:
        return self._value

    def set_value(self, val: Any) -> None:
        # Expecting SweepValue
        self._value = val
        self.on_change.emit(val)

    def to_dict(self) -> Any:
        return self._value.to_dict()


class MultiSweepLiveField(LiveField):
    def __init__(self, spec: MultiSweepSpec, initial_val: Any = None) -> None:
        super().__init__(spec)
        from .adapter import MultiSweepValue, SweepValue

        if isinstance(initial_val, MultiSweepValue):
            self._value = initial_val
        else:
            axes = {k: SweepValue(0.0, 1.0, 11) for k in spec.axes}
            self._value = MultiSweepValue(axes=axes)

    def get_value(self) -> Any:
        return self._value

    def set_value(self, val: Any) -> None:
        self._value = val
        self.on_change.emit(val)

    def to_dict(self) -> Any:
        return self._value.to_dict()


class ChannelLiveField(LiveField):
    """Reactive Channel field that resolves names via MetaDict."""

    def __init__(
        self,
        spec: ChannelSpec,
        bus: EventBus,
        md: Optional[MetaDict] = None,
        initial_val: Any = None,
    ) -> None:
        super().__init__(spec)
        self._bus = bus
        self._md = md

        from .adapter import ChannelValue

        if isinstance(initial_val, ChannelValue):
            self._chosen = initial_val.chosen
        else:
            self._chosen = initial_val if initial_val is not None else 0

        self._resolved_id: Optional[int] = None
        self._bus.subscribe(GuiEvent.MD_CHANGED, self._refresh_resolve)
        self._bus.subscribe(GuiEvent.CONTEXT_CHANGED, self._on_context_changed)
        self._refresh_resolve()

    def _on_context_changed(self, md: Optional[MetaDict] = None) -> None:
        # Event might pass new md
        if md is not None:
            self._md = md
        self._refresh_resolve()

    def _refresh_resolve(self) -> None:
        from .ui.fields.utils import _resolve_channel

        new_id = _resolve_channel(str(self._chosen), self._md)
        if new_id != self._resolved_id:
            self._resolved_id = new_id
            self._set_valid(new_id is not None)
            self.on_change.emit(self.get_value())

    def get_value(self) -> Any:
        from .adapter import ChannelValue

        return ChannelValue(chosen=self._chosen, resolved=self._resolved_id)

    def set_value(self, val: Any) -> None:
        # UI usually sets the 'chosen' part (str or int)
        if val != self._chosen:
            self._chosen = val
            self._refresh_resolve()
            self.on_change.emit(self.get_value())

    def to_dict(self) -> Any:
        return self._resolved_id

    def teardown(self) -> None:
        self._bus.unsubscribe(GuiEvent.MD_CHANGED, self._refresh_resolve)
        self._bus.unsubscribe(GuiEvent.CONTEXT_CHANGED, self._on_context_changed)


class SectionLiveField(LiveField):
    """Container for a group of fields."""

    def __init__(
        self,
        spec: CfgSectionSpec,
        bus: EventBus,
        ml: Optional[Any] = None,
        md: Optional[Any] = None,
        initial_val: Optional[CfgSectionValue] = None,
    ) -> None:
        super().__init__(spec)
        self.fields: dict[str, LiveField] = {}
        self._bus = bus
        self._ml = ml
        self._md = md

        # Build child fields
        val_map = initial_val.fields if initial_val else {}
        for key, node_spec in spec.fields.items():
            child_val = val_map.get(key)
            field = create_live_field(node_spec, bus, ml, md, child_val)
            self.fields[key] = field
            field.on_change.connect(self._on_child_change)
            field.on_validity_changed.connect(self._on_child_validity_change)

        self._refresh_validity()

    def _on_child_change(self, *_: Any) -> None:
        self.on_change.emit(self.get_value())

    def _on_child_validity_change(self, *_: Any) -> None:
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        self._set_valid(all(f.is_valid() for f in self.fields.values()))

    def get_value(self) -> Any:
        from .adapter import CfgSectionValue

        return CfgSectionValue(fields={k: f.get_value() for k, f in self.fields.items()})

    def set_value(self, val: Any) -> None:
        # val should be CfgSectionValue
        for k, field in self.fields.items():
            if k in val.fields:
                field.set_value(val.fields[k])

    def to_dict(self) -> dict[str, Any]:
        return {k: f.to_dict() for k, f in self.fields.items()}

    def teardown(self) -> None:
        for f in self.fields.values():
            if hasattr(f, "teardown"):
                f.teardown()


class ModuleRefLiveField(LiveField):
    """Reactive field for Module/Waveform references with dynamic sub-sections."""

    def __init__(
        self,
        spec: Union[ModuleRefSpec, WaveformRefSpec],
        bus: EventBus,
        ml: Optional[Any] = None,
        md: Optional[Any] = None,
        initial_val: Any = None,
    ) -> None:
        super().__init__(spec)
        self._bus = bus
        self._ml = ml
        self._md = md

        from .adapter import ModuleRefValue, WaveformRefValue

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
        if self.sub_field:
            self.sub_field.teardown()

        from .ui.fields.utils import _spec_for_chosen

        chosen_spec = _spec_for_chosen(self._chosen_key, self.spec.allowed, self._ml)
        if chosen_spec:
            self.sub_field = SectionLiveField(
                chosen_spec, self._bus, self._ml, self._md, self._sub_value
            )
            self.sub_field.on_change.connect(self._on_sub_change)
            self.sub_field.on_validity_changed.connect(self._on_sub_validity_change)
        else:
            self.sub_field = None

        self._refresh_validity()

    def _on_sub_change(self, *_: Any) -> None:
        self.on_change.emit(self.get_value())

    def _on_sub_validity_change(self, *_: Any) -> None:
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        valid = self.sub_field.is_valid() if self.sub_field else True
        self._set_valid(valid)

    def get_chosen_key(self) -> str:
        return self._chosen_key

    def set_chosen_key(self, key: str) -> None:
        if key != self._chosen_key:
            self._chosen_key = key
            self._sub_value = None  # Reset sub-value on key change
            self._rebuild_sub_field()
            self.on_change.emit(self.get_value())

    def get_value(self) -> Any:
        from .adapter import ModuleRefValue, WaveformRefValue, ModuleRefSpec

        klass = ModuleRefValue if isinstance(self.spec, ModuleRefSpec) else WaveformRefValue
        sub_val = self.sub_field.get_value() if self.sub_field else None
        return klass(chosen_key=self._chosen_key, value=sub_val)

    def set_value(self, val: Any) -> None:
        # val is ModuleRefValue or WaveformRefValue
        if val.chosen_key != self._chosen_key:
            self._chosen_key = val.chosen_key
            self._sub_value = val.value
            self._rebuild_sub_field()
        elif self.sub_field:
            self.sub_field.set_value(val.value)
        self.on_change.emit(self.get_value())

    def to_dict(self) -> Any:
        if self.sub_field:
            return self.sub_field.to_dict()
        return {}

    def teardown(self) -> None:
        if self.sub_field:
            self.sub_field.teardown()


def create_live_field(
    spec: CfgNodeSpec,
    bus: EventBus,
    ml: Optional[Any] = None,
    md: Optional[Any] = None,
    initial_val: Any = None,
) -> LiveField:
    """Factory to create the appropriate LiveField from a Spec."""
    from .adapter import (
        CfgSectionSpec,
        ChannelSpec,
        LiteralSpec,
        ModuleRefSpec,
        MultiSweepSpec,
        ScalarSpec,
        SweepSpec,
        WaveformRefSpec,
    )

    if isinstance(spec, ScalarSpec):
        return ScalarLiveField(spec, initial_val)
    if isinstance(spec, LiteralSpec):
        return LiteralLiveField(spec, initial_val)
    if isinstance(spec, SweepSpec):
        return SweepLiveField(spec, initial_val)
    if isinstance(spec, MultiSweepSpec):
        return MultiSweepLiveField(spec, initial_val)
    if isinstance(spec, ChannelSpec):
        return ChannelLiveField(spec, bus, md, initial_val)
    if isinstance(spec, (ModuleRefSpec, WaveformRefSpec)):
        return ModuleRefLiveField(spec, bus, ml, md, initial_val)
    if isinstance(spec, CfgSectionSpec):
        return SectionLiveField(spec, bus, ml, md, initial_val)

    raise TypeError(f"Unknown spec type: {type(spec)}")
