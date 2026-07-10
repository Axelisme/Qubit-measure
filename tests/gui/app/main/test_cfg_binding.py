from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
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

    with pytest.raises(RuntimeError, match="Unsupported module type"):
        bindings.keys("module", frozenset({"Pulse"}))
