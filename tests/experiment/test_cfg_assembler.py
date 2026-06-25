from __future__ import annotations

from typing import Any

import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.cfg_assembler import assemble_experiment_cfg, make_cfg
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.program.v2 import PulseCfg


class _DeviceOnlyCfg(ExpCfgModel):
    pass


class _PulseModules(ConfigBase):
    drive: PulseCfg


class _PulseExperimentCfg(ExpCfgModel):
    modules: _PulseModules


def _device(value: float) -> FakeDeviceInfo:
    return FakeDeviceInfo(address="fake", value=value)


def _assert_fake_device_value(cfg: _DeviceOnlyCfg, name: str, value: float) -> None:
    assert cfg.dev is not None
    dev = cfg.dev[name]
    assert isinstance(dev, FakeDeviceInfo)
    assert dev.value == pytest.approx(value)


def _pulse_raw(freq: float) -> dict[str, Any]:
    return {
        "type": "pulse",
        "waveform": {"style": "const", "length": 1.0},
        "ch": 0,
        "nqz": 1,
        "freq": freq,
        "gain": 0.1,
    }


def _ml_with_drive(freq: float) -> ModuleLibrary:
    ml = ModuleLibrary()
    ml.register_module(drive=_pulse_raw(freq))
    return ml


def test_assemble_experiment_cfg_uses_explicit_device_snapshot() -> None:
    snapshot = {"flux": _device(1.0)}

    cfg = assemble_experiment_cfg(
        {"dev": {"flux": {"value": 2.5, "output": "on"}}},
        _DeviceOnlyCfg,
        ml=ModuleLibrary(),
        device_snapshot=snapshot,
    )

    _assert_fake_device_value(cfg, "flux", 2.5)
    assert cfg.dev is not None
    assert cfg.dev["flux"].output == "on"
    assert snapshot["flux"].value == pytest.approx(1.0)


def test_make_cfg_with_explicit_snapshot_does_not_read_global_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get_all_info() -> dict[str, FakeDeviceInfo]:
        raise AssertionError("GlobalDeviceManager must not be read")

    monkeypatch.setattr(
        "zcu_tools.experiment.cfg_assembler.GlobalDeviceManager.get_all_info",
        fail_get_all_info,
    )

    cfg = make_cfg(
        {"dev": {"flux": {"value": 3.0}}},
        _DeviceOnlyCfg,
        ml=ModuleLibrary(),
        device_snapshot={"flux": _device(1.0)},
    )

    _assert_fake_device_value(cfg, "flux", 3.0)


def test_assemble_experiment_cfg_uses_ml_from_each_call() -> None:
    raw_cfg = {"modules": {"drive": "drive"}}

    cfg_a = assemble_experiment_cfg(
        raw_cfg,
        _PulseExperimentCfg,
        ml=_ml_with_drive(1000.0),
        device_snapshot={},
    )
    cfg_b = assemble_experiment_cfg(
        raw_cfg,
        _PulseExperimentCfg,
        ml=_ml_with_drive(2000.0),
        device_snapshot={},
    )

    assert cfg_a.modules.drive.freq == pytest.approx(1000.0)
    assert cfg_b.modules.drive.freq == pytest.approx(2000.0)


def test_module_library_make_cfg_forwards_to_cfg_assembler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ml = ModuleLibrary()
    sentinel = object()
    calls: dict[str, object] = {}

    def fake_make_cfg(
        raw_cfg: dict[str, Any],
        cfg_model: type[_DeviceOnlyCfg],
        *,
        ml: ModuleLibrary,
        overrides: dict[str, Any] | None = None,
    ) -> object:
        calls["raw_cfg"] = raw_cfg
        calls["cfg_model"] = cfg_model
        calls["ml"] = ml
        calls["overrides"] = overrides
        return sentinel

    monkeypatch.setattr("zcu_tools.experiment.cfg_assembler.make_cfg", fake_make_cfg)

    assert ml.make_cfg({"reps": 1}, _DeviceOnlyCfg, relax_delay=2.0) is sentinel
    assert calls == {
        "raw_cfg": {"reps": 1},
        "cfg_model": _DeviceOnlyCfg,
        "ml": ml,
        "overrides": {"relax_delay": 2.0},
    }
