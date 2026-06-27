from __future__ import annotations

from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.fake.stub import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import ADAPTERS
from zcu_tools.gui.app.main.adapter import AdapterGuide


def test_registered_adapters_define_local_guide_text() -> None:
    missing = [
        name for name, cls in ADAPTERS.items() if "guide_text" not in cls.__dict__
    ]

    assert not missing, f"registered adapters without local guide_text: {missing}"


def test_registered_adapters_use_base_guide_method() -> None:
    overrides = [name for name, cls in ADAPTERS.items() if "guide" in cls.__dict__]

    assert not overrides, f"registered adapters overriding guide(): {overrides}"


def test_registered_adapter_guides_are_written_adapter_guides() -> None:
    missing: list[str] = []
    wrong_type: list[str] = []

    for name, cls in ADAPTERS.items():
        guide = cls.guide()
        if not isinstance(guide, AdapterGuide):
            wrong_type.append(name)
        if guide.behavior == BaseAdapter.guide_text.behavior:
            missing.append(name)

    assert not wrong_type, (
        f"registered adapters with non-AdapterGuide guide: {wrong_type}"
    )
    assert not missing, f"registered adapters without written guide text: {missing}"


def test_unregistered_adapter_uses_honest_default_guide() -> None:
    guide = FakeAdapter.guide()

    assert guide is BaseAdapter.guide_text
    assert guide.behavior == "(no guide written yet)"
    assert guide.expects_md == ""
    assert guide.expects_ml == ""
    assert guide.typical_writeback == ""
    assert guide.recommended == ""
