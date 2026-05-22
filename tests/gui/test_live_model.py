"""Tests for LiveModel reactive data layer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    ChannelSpec,
    ChannelValue,
    ScalarSpec,
    ScalarValue,
)
from zcu_tools.gui.event_bus import EventBus, GuiEvent
from zcu_tools.gui.live_model import (
    ChannelLiveField,
    ScalarLiveField,
    SectionLiveField,
    create_live_field,
)


def test_scalar_field_reactivity():
    spec = ScalarSpec(label="Test", type=int)
    field = ScalarLiveField(spec, initial_val=ScalarValue(10))
    
    cb = MagicMock()
    field.on_change.connect(cb)
    
    field.set_value(20)
    assert field.get_value().value == 20
    assert field.to_dict() == 20
    assert cb.called


def test_section_field_propagation():
    spec = CfgSectionSpec(
        fields={
            "f1": ScalarSpec(label="F1", type=int),
            "f2": ScalarSpec(label="F2", type=float),
        }
    )
    initial_val = CfgSectionValue(
        fields={
            "f1": ScalarValue(1),
            "f2": ScalarValue(0.5),
        }
    )
    bus = EventBus()
    section = SectionLiveField(spec, bus, initial_val=initial_val)
    
    cb = MagicMock()
    section.on_change.connect(cb)
    
    section.fields["f1"].set_value(10)
    
    # Check that section emitted on_change
    assert cb.called
    val = section.get_value()
    assert val.fields["f1"].value == 10
    assert val.fields["f2"].value == 0.5
    
    assert section.to_dict() == {"f1": 10, "f2": 0.5}

def test_channel_field_resolution():
    spec = ChannelSpec(label="Ch")
    bus = EventBus()
    md = MagicMock()
    md.qub_ch = 5
    
    # We need to mock _resolve_channel or use a real MetaDict if possible.
    # Since it's imported inside the method, we can patch it.
    with MagicMock() as mock_resolve:
        import zcu_tools.gui.live_model as lm
        # ChannelLiveField imports from .ui.cfg_form._resolve_channel
        # For test, let's just assume it works or mock the import.
        pass

    # A simpler way to test ChannelLiveField without complex patching:
    # Use a dummy MetaDict that works with the real _resolve_channel.
    from zcu_tools.meta_tool import MetaDict
    md = MetaDict()
    md.qub_ch = 7
    
    field = ChannelLiveField(spec, bus, md, initial_val="qub_ch")
    assert field.get_value().resolved == 7
    assert field.is_valid() is True
    
    # Update md
    md.qub_ch = 9
    bus.emit(GuiEvent.MD_CHANGED)
    assert field.get_value().resolved == 9
    
    # Change to unknown
    field.set_value("unknown_ch")
    assert field.get_value().resolved is None
    assert field.is_valid() is False
