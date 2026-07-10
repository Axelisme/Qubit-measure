from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.autofluxdep.cfg.binding import AutofluxCfgBindings
from zcu_tools.gui.app.autofluxdep.cfg.module_adapter import waveform_cfg_to_value
from zcu_tools.gui.cfg import ScalarSpec
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

    with pytest.raises(RuntimeError, match="Unsupported module type"):
        bindings.keys("module", frozenset({"Pulse"}))
