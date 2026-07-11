from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import zcu_tools.gui.app.main.cfg_binding as binding_module
from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.measure_cfg import ProgramShape, UnknownProgramShapeError
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _bindings(ml: ModuleLibrary | None = None) -> tuple[MeasureCfgBindings, MagicMock]:
    host = MagicMock()
    host.get_current_md.return_value = MetaDict()
    host.get_current_ml.return_value = ml or ModuleLibrary()
    host.list_device_names.return_value = ["flux"]
    host.list_arb_waveforms.return_value = ["asset"]
    return MeasureCfgBindings(host), host


def _pulse() -> dict[str, object]:
    return {
        "type": "pulse",
        "waveform": {"style": "const", "length": 0.1},
        "ch": 0,
        "freq": 5000.0,
        "gain": 0.2,
        "phase": 0.0,
        "pre_delay": 0.0,
        "post_delay": 0.0,
    }


def test_measure_option_provider_owns_device_and_arb_catalogs() -> None:
    bindings, _ = _bindings()

    assert bindings.provide_options("devices") == ["flux"]
    assert bindings.provide_options("arb_waveforms") == ["asset"]
    with pytest.raises(RuntimeError, match="Unsupported measure cfg option source"):
        bindings.provide_options("unknown")


def test_measure_catalog_filters_by_shape_and_materializes_resolution() -> None:
    ml = ModuleLibrary()
    ml.modules["drive"] = cast(Any, _pulse())
    ml.modules["direct"] = cast(
        Any,
        {
            "type": "readout/direct",
            "ro_ch": 0,
            "ro_freq": 6000.0,
            "ro_length": 1.0,
            "trig_offset": 0.1,
        },
    )
    bindings, _ = _bindings(ml)

    assert bindings.keys("module", frozenset({"Pulse"})) == ("drive",)
    resolved = bindings.resolve("module", "drive")
    assert resolved is not None
    assert resolved.label == "Pulse"
    assert resolved.value is not None
    assert bindings.resolve("module", "missing") is None


def test_measure_catalog_corrupt_entry_fast_fails_during_enumeration() -> None:
    ml = ModuleLibrary()
    ml.modules["corrupt"] = cast(Any, {"type": "not-a-module"})
    bindings, _ = _bindings(ml)

    with pytest.raises(UnknownProgramShapeError, match="Unknown module program shape"):
        bindings.keys("module", frozenset({"Pulse"}))


def test_measure_keys_inspects_shapes_without_converter_or_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TypedModule:
        def __init__(self, discriminator: str) -> None:
            self.type = discriminator

        def to_dict(self):
            raise AssertionError("keys must not normalize typed cfg")

    ml = ModuleLibrary()
    ml.modules["drive"] = cast(Any, TypedModule("pulse"))
    ml.modules["readout"] = cast(Any, TypedModule("readout/direct"))
    bindings, _ = _bindings(ml)
    shape_lookup = MagicMock(side_effect=binding_module.program_shape_for_input)
    converter = MagicMock(side_effect=binding_module.module_cfg_to_value)
    monkeypatch.setattr(binding_module, "program_shape_for_input", shape_lookup)
    monkeypatch.setattr(binding_module, "module_cfg_to_value", converter)
    monkeypatch.setattr(
        ProgramShape,
        "make_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("keys must not construct specs")
        ),
    )

    assert bindings.keys("module", frozenset({"Pulse", "Direct Readout"})) == (
        "drive",
        "readout",
    )
    assert shape_lookup.call_count == 2
    converter.assert_not_called()


def test_measure_resolve_calls_app_converter_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ml = ModuleLibrary()
    ml.modules["drive"] = cast(Any, _pulse())
    bindings, _ = _bindings(ml)
    shape_lookup = MagicMock(side_effect=binding_module.program_shape_for_input)
    converter = MagicMock(side_effect=binding_module.module_cfg_to_value)
    monkeypatch.setattr(binding_module, "program_shape_for_input", shape_lookup)
    monkeypatch.setattr(binding_module, "module_cfg_to_value", converter)

    assert bindings.resolve("module", "missing") is None
    shape_lookup.assert_not_called()
    converter.assert_not_called()
    assert bindings.resolve("module", "drive") is not None
    shape_lookup.assert_not_called()
    assert converter.call_count == 1


def test_measure_resolve_materializes_missing_waveform_style_as_const(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ml = ModuleLibrary()
    ml.waveforms["legacy"] = cast(Any, {})
    bindings, _ = _bindings(ml)
    shape_lookup = MagicMock(side_effect=binding_module.program_shape_for_input)
    converter = MagicMock(side_effect=binding_module.waveform_cfg_to_value)
    monkeypatch.setattr(binding_module, "program_shape_for_input", shape_lookup)
    monkeypatch.setattr(binding_module, "waveform_cfg_to_value", converter)

    resolved = bindings.resolve("waveform", "legacy")

    assert resolved is not None
    assert resolved.label == "Const"
    assert resolved.value is not None
    shape_lookup.assert_not_called()
    assert converter.call_count == 1
