"""LiveModel — runtime *draft* SSOT for one CfgSchema editing surface.

The ``SectionLiveField`` tree built by ``CfgFormWidget.populate`` is the
authoritative live state while a form is being edited: it owns
``DirectValue``/``EvalValue`` resolution, validity, dirty bubbling, ModuleRef
binding and SweepEditor canonicalization. There are two distinct draft surfaces
using this same machinery:

- **Tab form (auto-commit draft)** — every ``on_change`` flows through
  ``CfgFormWidget._emit_schema_changed`` → ``Controller.update_tab_cfg`` and
  is committed into ``State.cfg_schema`` immediately. The tab's LiveModel and
  the committed State value are kept in lockstep.
- **Dialog / writeback (local draft)** — ``inspect_dialog.py`` and
  ``writeback_widget.py`` each construct their own ``CfgFormWidget`` /
  LiveModel; their drafts never reach ``State.cfg_schema`` until an explicit
  Apply path writes to ``ModuleLibrary`` or to a ``WritebackItem``.

LiveModel itself is not the persisted truth. ``State.cfg_schema`` is the
committed SSOT used by run / save / session persistence; this module produces
the draft that auto-commits (tab) or stays local (dialog).

Uses LiveModelEnv for dependency injection, fetches md/ml through the
controller boundary, and tracks unset Scalar fields explicitly.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Union, cast

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
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    default_value_for_type,
)
from .sweep_model import SweepEditor

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
        self._updating = False
        if isinstance(initial_val, SweepValue):
            initial = SweepEditor.canonicalize(initial_val)
        else:
            initial = SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)
        start_init = initial.start
        stop_init = initial.stop
        self._expts = initial.expts
        self._step = initial.step

        edge_spec = ScalarSpec(
            label=spec.label,
            type=float,
            decimals=spec.decimals,
            editable=spec.editable,
        )
        self.start_field = ScalarLiveField(
            edge_spec, env, initial_val=self._coerce_edge(start_init)
        )
        self.stop_field = ScalarLiveField(
            edge_spec, env, initial_val=self._coerce_edge(stop_init)
        )
        self.start_field.on_change.connect(self._on_child_change)
        self.stop_field.on_change.connect(self._on_child_change)
        self.start_field.on_validity_changed.connect(self._on_child_validity_changed)
        self.stop_field.on_validity_changed.connect(self._on_child_validity_changed)
        self._refresh_validity()

    def get_value(self) -> SweepValue:
        return SweepValue(
            start=self._edge_value(self.start_field.get_value()),
            stop=self._edge_value(self.stop_field.get_value()),
            expts=self._expts,
            step=self._step,
        )

    def set_value(self, val: object) -> None:
        if isinstance(val, SweepValue):
            canonical = SweepEditor.canonicalize(val)
            self._updating = True
            try:
                self.start_field.set_value(self._coerce_edge(canonical.start))
                self.stop_field.set_value(self._coerce_edge(canonical.stop))
                self._expts = canonical.expts
                self._step = canonical.step
            finally:
                self._updating = False
            self._refresh_validity()
            self.on_change.emit(self.get_value())
            return
        raise TypeError(f"SweepLiveField expects SweepValue, got {type(val).__name__}")

    def update_expts(self, expts: int) -> None:
        self.set_value(SweepEditor.update_expts(self.get_value(), expts))

    def update_step(self, step: float) -> None:
        self.set_value(SweepEditor.update_step(self.get_value(), step))

    def teardown(self) -> None:
        self.start_field.teardown()
        self.stop_field.teardown()

    def refresh_external(self, event: object) -> None:
        self.start_field.refresh_external(event)
        self.stop_field.refresh_external(event)
        self._refresh_validity()

    def _coerce_edge(self, value: object) -> ScalarValue:
        if isinstance(value, EvalValue):
            return value
        if isinstance(value, (int, float)):
            return DirectValue(value=float(value), is_unset=False)
        raise TypeError(
            f"Sweep edge expects float or EvalValue, got {type(value).__name__}"
        )

    def _edge_value(self, value: ScalarValue) -> Union[float, EvalValue]:
        if isinstance(value, EvalValue):
            return value
        return float(value.value)

    def _on_child_change(self, *_: object) -> None:
        if self._updating:
            return
        canonical = SweepEditor.canonicalize(self.get_value())
        self._expts = canonical.expts
        self._step = canonical.step
        self._refresh_validity()
        self.on_change.emit(canonical)

    def _on_child_validity_changed(self, *_: object) -> None:
        self._refresh_validity()

    def _refresh_validity(self) -> None:
        self._set_valid(self.start_field.is_valid() and self.stop_field.is_valid())


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

        init_overridden = False
        if isinstance(initial_val, (ModuleRefValue, WaveformRefValue)):
            self._chosen_key = initial_val.chosen_key
            init_sub: Optional[CfgSectionValue] = initial_val.value
            init_overridden = initial_val.is_overridden
        else:
            # Default to first allowed
            self._chosen_key = (
                f"<Custom:{spec.allowed[0].label}>" if spec.allowed else ""
            )
            init_sub = None

        self._binding_state = _binding_state_for_key(self._chosen_key)
        # Restore a persisted override: a library ref whose value was edited away
        # from the snapshot reloads as MODIFIED, not LINKED. (<Custom:> refs stay
        # CUSTOM; is_overridden is meaningless there.)
        if init_overridden and self._binding_state is LibraryBindingState.LINKED:
            self._binding_state = LibraryBindingState.MODIFIED
        self.sub_field: Optional[SectionLiveField] = None
        # True when a LINKED ref's chosen_key names a library entry that does not
        # exist (deleted/renamed). Kept LINKED so re-adding the name re-links it;
        # surfaced as invalid + a red "missing library reference" badge.
        self._missing_library_ref: bool = False
        self.is_enabled: bool = not (spec.optional and initial_val is None)
        self.on_enabled_changed = CallbackList()
        self._rebuild_sub_field(hint=init_sub)

    def is_modified(self) -> bool:
        return self._binding_state is LibraryBindingState.MODIFIED

    def has_missing_library_ref(self) -> bool:
        return self._missing_library_ref

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
        self._missing_library_ref = False
        try:
            chosen_spec, lib_val = _spec_value_for_chosen(
                self._chosen_key, self.spec.allowed, ml
            )
        except RuntimeError as exc:
            # The referenced library entry no longer exists (deleted/renamed, or
            # a persisted session referencing an absent entry). Two behaviours by
            # binding state (deliberately asymmetric — see CONTEXT.md):
            #   - MODIFIED (user edited the value away from the snapshot): heal to
            #     an inline <Custom:…> KEEPING the edits (the override would
            #     otherwise be lost; it has already left the library).
            #   - LINKED (pure library ref, value = snapshot, no edits to lose):
            #     keep chosen_key, mark missing+invalid (red badge). The user
            #     sees the broken ref, and re-adding an entry of the same name
            #     auto-relinks it to LINKED (the recoverable path).
            if "Unknown library reference" in str(
                exc
            ) and not self._chosen_key.startswith("<Custom:"):
                if self._binding_state is LibraryBindingState.MODIFIED:
                    kept_val = old_val if old_val is not None else hint
                    label = self._custom_label_for_value(old_spec, kept_val)
                    logger.warning(
                        "ML ref %r (modified) no longer exists; converting to "
                        "inline <Custom:%s>, keeping edits",
                        self._chosen_key,
                        label,
                    )
                    self._chosen_key = f"<Custom:{label}>"
                    self._binding_state = LibraryBindingState.CUSTOM
                    chosen_spec, lib_val = _spec_value_for_chosen(
                        self._chosen_key, self.spec.allowed, ml
                    )
                    if hint is None:
                        hint = kept_val
                else:
                    logger.warning(
                        "ML ref %r no longer exists in the library "
                        "(missing reference)",
                        self._chosen_key,
                    )
                    self._missing_library_ref = True
                    chosen_spec = None
                    lib_val = None
            else:
                raise
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

    def _custom_label_for_value(
        self,
        old_spec: Optional[CfgNodeSpec],
        value: Optional[CfgSectionValue],
    ) -> str:
        """Pick the allowed <Custom:label> shape for a dangling ref's heal.

        Prefer the prior sub_field's spec label (already an allowed label). Else
        match the value's type/style discriminator against an allowed spec's
        label by lowering each allowed shape. Fall back to the first allowed.
        """
        if isinstance(old_spec, CfgSectionSpec) and any(
            s.label == old_spec.label for s in self.spec.allowed
        ):
            return old_spec.label
        if isinstance(value, CfgSectionValue):
            disc_field = value.fields.get("type") or value.fields.get("style")
            disc = getattr(disc_field, "value", None)
            if isinstance(disc, str):
                for spec in self.spec.allowed:
                    lit = spec.fields.get("type") or spec.fields.get("style")
                    if getattr(lit, "value", None) == disc and spec.label:
                        return spec.label
        return self.spec.allowed[0].label if self.spec.allowed else "Custom"

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
        if self._missing_library_ref:
            logger.debug(
                "ModuleRefLiveField._refresh_validity: key=%r missing in library",
                self._chosen_key,
            )
            self._set_valid(False)
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
        # Custom refs never track the library.
        if self._chosen_key.startswith("<Custom:"):
            return
        # A present MODIFIED ref is left alone (rebuilding would discard edits).
        # Everything else rebuilds — _rebuild_sub_field then decides:
        #   LINKED-present  → re-sync from the library snapshot
        #   LINKED-absent   → mark missing + invalid (recoverable: re-adding the
        #                     name re-links here on the next ML_CHANGED)
        #   MODIFIED-absent → heal to inline Custom, keeping the edits
        if self._library_key_present() and self._binding_state is not (
            LibraryBindingState.LINKED
        ):
            return
        self._rebuild_sub_field()
        self.on_change.emit(self.get_value())

    def _library_key_present(self) -> bool:
        ml = self.env.ctrl.get_current_ml()
        if ml is None:
            return False
        store = ml.modules if isinstance(self.spec, ModuleRefSpec) else ml.waveforms
        return self._chosen_key in store

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
        return klass(
            chosen_key=self._chosen_key,
            value=sub_val,
            is_overridden=self.is_modified(),
        )

    def set_value(self, val: object) -> None:
        if not isinstance(val, (ModuleRefValue, WaveformRefValue)):
            raise TypeError(
                "ModuleRefLiveField expects ModuleRefValue or WaveformRefValue, "
                f"got {type(val).__name__}"
            )
        if val.chosen_key != self._chosen_key or self.is_modified():
            self._chosen_key = val.chosen_key
            self._binding_state = _binding_state_for_key(val.chosen_key)
            if val.is_overridden and self._binding_state is LibraryBindingState.LINKED:
                self._binding_state = LibraryBindingState.MODIFIED
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
