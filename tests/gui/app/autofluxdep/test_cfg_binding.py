from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import zcu_tools.gui.app.autofluxdep.cfg.binding as binding_module
from zcu_tools.gui.app.autofluxdep.cfg.binding import AutofluxCfgBindings
from zcu_tools.gui.app.autofluxdep.cfg.module_adapter import waveform_cfg_to_value
from zcu_tools.gui.cfg import ScalarSpec
from zcu_tools.gui.measure_cfg import ProgramShape, UnknownProgramShapeError
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _bindings(ml: ModuleLibrary | None = None) -> tuple[AutofluxCfgBindings, MagicMock]:
    host = MagicMock()
    host.get_current_md.return_value = MetaDict()
    host.get_current_ml.return_value = ml or ModuleLibrary()
    host.list_device_names.return_value = ["flux"]
    return AutofluxCfgBindings(host), host


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


def test_autoflux_arb_data_is_free_form_without_measure_catalog() -> None:
    spec, _ = waveform_cfg_to_value({"style": "arb", "data": "legacy_asset"})
    data_spec = spec.fields["data"]

    assert isinstance(data_spec, ScalarSpec)
    assert data_spec.choices_source == ""


def test_autoflux_option_provider_does_not_impersonate_arb_catalog() -> None:
    bindings, _ = _bindings()

    assert bindings.provide_options("devices") == ["flux"]
    with pytest.raises(RuntimeError, match="Unsupported autoflux cfg option source"):
        bindings.provide_options("arb_waveforms")


def test_autoflux_catalog_separates_compatible_and_valid_unsupported_shapes() -> None:
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

    assert bindings.keys("module", frozenset({"Pulse", "Direct Readout"})) == ("drive",)
    supported = bindings.resolve("module", "drive")
    assert supported is not None
    assert supported.label == "Pulse"
    assert supported.value is not None
    unsupported = bindings.resolve("module", "direct")
    assert unsupported is not None
    assert unsupported.label == "Direct Readout"
    assert unsupported.value is None
    assert bindings.resolve("module", "missing") is None


def test_autoflux_catalog_corrupt_entry_fast_fails_during_enumeration() -> None:
    ml = ModuleLibrary()
    ml.modules["corrupt"] = cast(Any, {"type": "not-a-module"})
    bindings, _ = _bindings(ml)

    with pytest.raises(UnknownProgramShapeError, match="Unknown module program shape"):
        bindings.keys("module", frozenset({"Pulse"}))


def test_autoflux_keys_never_materializes_supported_or_legal_unsupported_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ml = ModuleLibrary()
    ml.modules["drive"] = cast(Any, _pulse())
    ml.modules["direct"] = cast(Any, {"type": "readout/direct"})
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

    assert bindings.keys("module", frozenset({"Pulse", "Direct Readout"})) == ("drive",)
    assert shape_lookup.call_count == 2
    converter.assert_not_called()


def test_autoflux_resolve_uses_discriminator_capability_and_converter_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ml = ModuleLibrary()
    ml.modules["drive"] = cast(Any, _pulse())
    ml.modules["direct"] = cast(Any, {"type": "readout/direct"})
    bindings, _ = _bindings(ml)
    shape_lookup = MagicMock(side_effect=binding_module.program_shape_for_input)
    converter = MagicMock(side_effect=binding_module.module_cfg_to_value)
    monkeypatch.setattr(binding_module, "program_shape_for_input", shape_lookup)
    monkeypatch.setattr(binding_module, "module_cfg_to_value", converter)

    assert bindings.resolve("module", "missing") is None
    shape_lookup.assert_not_called()
    converter.assert_not_called()

    unsupported = bindings.resolve("module", "direct")
    assert unsupported is not None
    assert unsupported.label == "Direct Readout"
    assert unsupported.value is None
    assert shape_lookup.call_count == 1
    converter.assert_not_called()

    assert bindings.resolve("module", "drive") is not None
    assert shape_lookup.call_count == 2
    assert converter.call_count == 1
