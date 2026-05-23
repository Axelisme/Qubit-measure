"""Tests for LiveModel reactive data layer."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    ChannelSpec,
    ChannelValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
    ScalarValue,
)
from zcu_tools.gui.event_bus import EventBus, GuiEvent
from zcu_tools.gui.live_model import (
    ChannelLiveField,
    LiveModelEnv,
    ScalarLiveField,
    SectionLiveField,
)


@pytest.fixture()
def bus():
    return EventBus()


@pytest.fixture()
def ctrl(bus):
    c = MagicMock()
    c.get_bus.return_value = bus
    c.get_current_md.return_value = MagicMock()
    c.get_current_ml.return_value = MagicMock()
    return c


@pytest.fixture()
def env(bus, ctrl):
    return LiveModelEnv(ctrl=ctrl)


def test_scalar_field_reactivity(env):
    spec = ScalarSpec(label="Test", type=int)
    field = ScalarLiveField(spec, env, initial_val=DirectValue(10))

    cb = MagicMock()
    field.on_change.connect(cb)

    field.set_value(20)
    value = field.get_value()
    assert isinstance(value, DirectValue)
    assert value.value == 20
    assert field.to_dict() == 20
    assert cb.called


def test_section_field_propagation(env):
    spec = CfgSectionSpec(
        fields={
            "f1": ScalarSpec(label="F1", type=int),
            "f2": ScalarSpec(label="F2", type=float),
        }
    )
    initial_val = CfgSectionValue(
        fields={
            "f1": DirectValue(1),
            "f2": DirectValue(0.5),
        }
    )
    section = SectionLiveField(spec, env, initial_val=initial_val)

    cb = MagicMock()
    section.on_change.connect(cb)

    section.fields["f1"].set_value(10)

    # Check that section emitted on_change
    assert cb.called
    val = section.get_value()
    f1_val = cast(ScalarValue, val.fields["f1"])
    f2_val = cast(ScalarValue, val.fields["f2"])
    assert isinstance(f1_val, DirectValue)
    assert isinstance(f2_val, DirectValue)
    assert f1_val.value == 10
    assert f2_val.value == 0.5

    assert section.to_dict() == {"f1": 10, "f2": 0.5}


def test_channel_field_resolution(env):
    spec = ChannelSpec(label="Ch")

    # Setup mock md
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.qub_ch = 7
    env.ctrl.get_current_md.return_value = md

    field = ChannelLiveField(spec, env, initial_val="qub_ch")
    assert field.get_value().resolved == 7
    assert field.is_valid() is True

    # Update md
    md.qub_ch = 9
    field.refresh_external(GuiEvent.MD_CHANGED)
    assert field.get_value().resolved == 9

    # Change to unknown
    field.set_value("unknown_ch")
    assert field.get_value().resolved is None
    assert field.is_valid() is False


def test_channel_field_does_not_subscribe_to_bus(env, bus):
    spec = ChannelSpec(label="Ch")
    field = ChannelLiveField(spec, env, initial_val="qub_ch")

    assert bus._subs == {}
    field.teardown()


def test_scalar_eval_field_resolves_from_md(env):
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    md.rf_w = 2.0
    env.ctrl.get_current_md.return_value = md

    spec = ScalarSpec(label="Freq", type=float)
    field = ScalarLiveField(spec, env, initial_val=EvalValue("r_f - 1.5 * rf_w"))

    val = field.get_value()
    assert isinstance(val, EvalValue)
    assert val.resolved == 5997.0
    assert field.is_valid() is True


def test_scalar_eval_field_refresh_updates_snapshot(env):
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    env.ctrl.get_current_md.return_value = md

    spec = ScalarSpec(label="Freq", type=float)
    field = ScalarLiveField(spec, env, initial_val=EvalValue("r_f"))
    cb = MagicMock()
    field.on_change.connect(cb)

    md.r_f = 6100.0
    field.refresh_external(GuiEvent.MD_CHANGED)

    val = field.get_value()
    assert isinstance(val, EvalValue)
    assert val.resolved == 6100.0
    assert cb.called


def test_scalar_eval_field_invalid_expression_marks_invalid(env):
    from zcu_tools.meta_tool import MetaDict

    env.ctrl.get_current_md.return_value = MetaDict()
    field = ScalarLiveField(
        ScalarSpec(label="Freq", type=float),
        env,
        initial_val=EvalValue("missing_name"),
    )

    val = field.get_value()
    assert isinstance(val, EvalValue)
    assert val.resolved is None
    assert field.is_valid() is False
