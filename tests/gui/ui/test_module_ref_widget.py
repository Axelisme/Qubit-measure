"""Tests — ModuleRefWidget combo auto-refresh on library change.

Bug: adding a module to the library after the widget was shown did not update
the combo — the widget only subscribed to model on_change (which only fires when
*this field's* referenced entry changes, not when the library set grows).

Fix: the widget now also subscribes to MlChangedPayload on the shared EventBus
so that any library-set mutation triggers a combo rebuild.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.gui.app.main.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
)
from zcu_tools.gui.app.main.live_model import LiveModelEnv, ModuleRefLiveField
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.events import MlChangedPayload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INNER_LABEL = "readout_rf"


def _inner_spec() -> CfgSectionSpec:
    """Minimal module spec whose label matches _INNER_LABEL."""
    return CfgSectionSpec(
        label=_INNER_LABEL,
        fields={"freq": ScalarSpec(label="Freq", type=float)},
    )


def _inner_value() -> CfgSectionValue:
    return CfgSectionValue(fields={"freq": DirectValue(1000.0)})


def _ref_spec() -> ModuleRefSpec:
    """ModuleRefSpec allowing one custom shape whose label is _INNER_LABEL."""
    return ModuleRefSpec(allowed=[_inner_spec()])


def _ctrl(bus: EventBus, ml: MagicMock | None = None) -> MagicMock:
    c = MagicMock()
    c.get_bus.return_value = bus
    c.get_current_md.return_value = MagicMock()
    c.get_current_ml.return_value = ml
    return c


def _make_field(ctrl: MagicMock) -> ModuleRefLiveField:
    spec = _ref_spec()
    value = ModuleRefValue(
        chosen_key=f"<Custom:{_INNER_LABEL}>",
        value=_inner_value(),
        is_overridden=False,
    )
    return ModuleRefLiveField(spec, LiveModelEnv(ctrl=ctrl), value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_module_ref_widget_combo_refreshes_on_ml_changed(qapp):
    """Emitting MlChangedPayload causes the combo to include the new library entry."""
    bus = EventBus()
    ml = MagicMock()
    # Start with an empty library — the widget shows only the custom-spec entry.
    ml.modules = {}
    ctrl = _ctrl(bus, ml)
    field = _make_field(ctrl)

    from zcu_tools.gui.app.main.ui.fields.containers import ModuleRefWidget

    w = ModuleRefWidget(field)
    count_before = w._combo.count()

    # Simulate adding "my_module" to the library: the ctrl now returns a ml
    # with one entry.  We patch module_cfg_to_value so the test does not need
    # a real ModuleCfg — the function is an internal implementation detail of
    # _refresh_combo_items and is tested separately.
    fake_cfg = object()
    ml.modules = {"my_module": fake_cfg}

    with patch(
        "zcu_tools.gui.app.main.cfg_schemas.module_cfg_to_value",
        return_value=(_inner_spec(), _inner_value()),
    ):
        # Dispatch the library-change event that ContextService emits.
        bus.emit(MlChangedPayload(ml=ml))

    count_after = w._combo.count()
    # The separator + the new library item should have been added.
    assert count_after > count_before, (
        f"Expected combo to grow after MlChangedPayload, "
        f"before={count_before} after={count_after}"
    )
    # The new entry must actually appear in the combo.
    texts = [w._combo.itemText(i) for i in range(w._combo.count())]
    assert any("my_module" in t for t in texts), (
        f"'my_module' not found in combo items: {texts}"
    )

    w.teardown()


def test_module_ref_widget_teardown_removes_bus_subscription(qapp):
    """After teardown, MlChangedPayload no longer triggers _refresh_combo_items."""
    bus = EventBus()
    ml = MagicMock()
    ml.modules = {}
    ctrl = _ctrl(bus, ml)
    field = _make_field(ctrl)

    from zcu_tools.gui.app.main.ui.fields.containers import ModuleRefWidget

    w = ModuleRefWidget(field)
    w.teardown()

    # After teardown: adding to the bus and emitting should not raise (stale
    # callback would cause AttributeError on a partially-torn-down widget or
    # silently succeed — we only need to confirm the subscriber count is zero).
    subscribers_after = bus._subs.get(MlChangedPayload, [])
    assert w._on_ml_changed not in subscribers_after, (
        "Stale subscription found after teardown — unsubscribe was not called"
    )


def test_module_ref_widget_initial_combo_without_ml(qapp):
    """When ml is None, combo shows only the custom-spec entry (no library section)."""
    bus = EventBus()
    # ctrl returns None for ml → no library section in combo
    ctrl = _ctrl(bus, ml=None)
    field = _make_field(ctrl)

    from zcu_tools.gui.app.main.ui.fields.containers import ModuleRefWidget

    w = ModuleRefWidget(field)

    texts = [w._combo.itemText(i) for i in range(w._combo.count())]
    # Only the single custom spec entry should be present.
    assert texts == [_INNER_LABEL], f"Unexpected combo items: {texts}"

    w.teardown()
