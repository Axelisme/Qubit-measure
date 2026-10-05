"""Registry-wide contract tests for context-free measure cfg definitions."""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    ExpAdapterProtocol,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.program.v2 import ModuleCfgFactory
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from zcu_lab.definitions import register_all


def _registry() -> Registry:
    registry = Registry()
    register_all(registry)
    return registry


def _pulse_raw(*, freq: float, gain: float) -> dict[str, object]:
    return {
        "type": "pulse",
        "waveform": {"style": "const", "length": 0.05},
        "ch": 3,
        "nqz": 2,
        "freq": freq,
        "gain": gain,
    }


def _readout_raw() -> dict[str, object]:
    return {
        "type": "readout/pulse",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 1.0},
            "ch": 1,
            "nqz": 2,
            "freq": 6100.0,
            "gain": 0.2,
        },
        "ro_cfg": {
            "ro_ch": 2,
            "ro_freq": 6100.0,
            "ro_length": 1.0,
            "trig_offset": 0.5,
        },
    }


def _ctx(*, rich: bool) -> SessionEnv:
    md = MetaDict()
    ml = ModuleLibrary()
    if rich:
        for key, value in {
            "t1": 12.0,
            "q_f": 4200.0,
            "r_f": 6000.0,
            "rf_w": 20.0,
            "rabi_f": 10.0,
            "res_probe_len": 1.0,
            "qub_ch": 3,
            "res_ch": 1,
            "ro_ch": 2,
            "timeFly": 0.45,
        }.items():
            setattr(md, key, value)
        ml.register_module(
            pi_amp=ModuleCfgFactory.from_raw(
                _pulse_raw(freq=4200.0, gain=0.4),
                ml=ml,
            ),
            pi2_amp=ModuleCfgFactory.from_raw(
                _pulse_raw(freq=4200.0, gain=0.2),
                ml=ml,
            ),
            readout_rf=ModuleCfgFactory.from_raw(_readout_raw(), ml=ml),
        )
    return SessionEnv(md=md, ml=ml, soc=None, soccfg=None)


@pytest.mark.parametrize("name", _registry().list_names())
def test_registered_adapter_conforms_to_framework(name: str) -> None:
    adapter = _registry().create(name)

    assert isinstance(adapter, ExpAdapterProtocol)
    assert isinstance(adapter.capabilities, AdapterCapabilities)


@pytest.mark.parametrize("name", _registry().list_names())
def test_registered_adapter_definition_has_context_free_shape(name: str) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)

    definition = type(adapter).cfg_definition()
    empty_cfg = adapter.make_default_cfg(_ctx(rich=False))
    rich_cfg = adapter.make_default_cfg(_ctx(rich=True))

    assert empty_cfg.spec == definition.spec
    assert rich_cfg.spec == definition.spec
    assert rich_cfg.spec == empty_cfg.spec


@pytest.mark.parametrize("name", _registry().list_names())
def test_registered_adapter_definition_is_reusable(name: str) -> None:
    adapter = _registry().create(name)
    assert isinstance(adapter, BaseAdapter)

    definition = type(adapter).cfg_definition()
    ctx = _ctx(rich=True)

    first = definition.instantiate(ctx)
    second = definition.instantiate(ctx)
    expected = definition.instantiate(ctx)

    assert first == second == expected
    assert first.spec is not second.spec
    assert first.spec.fields is not second.spec.fields
    assert first.value is not second.value
    assert first.value.fields is not second.value.fields

    first.spec.fields.clear()
    first.value.fields.clear()
    assert second == expected
    assert definition.instantiate(ctx) == expected

    exposed_spec = definition.spec
    exposed_spec.fields.clear()
    assert definition.spec == expected.spec
