from __future__ import annotations

from typing import Any

import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.cfg_assembler import CfgEnv, assemble_experiment_cfg, make_cfg
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.program.v2 import PulseCfg
from zcu_tools.resources.context import MetaDict, ModuleLibrary


class _DeviceOnlyCfg(ExpCfgModel):
    pass


class _PulseModules(ConfigBase):
    drive: PulseCfg


class _PulseExperimentCfg(ExpCfgModel):
    reps: int = 1
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


def test_make_cfg_reads_current_devices_without_setup_or_mutating_prior_cfg() -> None:
    manager = DeviceManager()
    device = FakeDevice(fast_mode=True)
    manager.register_device("flux", device)
    env = CfgEnv(md=MetaDict(), ml=ModuleLibrary(), device_manager=manager)
    raw = {"dev": {"flux": {"output": "on"}}}

    first = make_cfg(raw, _DeviceOnlyCfg, env)
    device.set_value(0.4)
    second = make_cfg(raw, _DeviceOnlyCfg, env)
    _assert_fake_device_value(first, "flux", 0.0)
    _assert_fake_device_value(second, "flux", 0.4)
    assert device.get_output() == "off"
    assert raw == {"dev": {"flux": {"output": "on"}}}


def test_make_cfg_propagates_device_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = FakeDevice(fast_mode=True)
    manager = DeviceManager()
    manager.register_device("flux", device)
    env = CfgEnv(md=MetaDict(), ml=ModuleLibrary(), device_manager=manager)
    make_cfg({}, _DeviceOnlyCfg, env)

    def failed_read():
        raise RuntimeError("device read failed")

    monkeypatch.setattr(device, "get_info", failed_read)
    with pytest.raises(RuntimeError, match="device read failed"):
        make_cfg({}, _DeviceOnlyCfg, env)


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


def test_make_cfg_uses_each_environment_library_and_overrides() -> None:
    manager = DeviceManager()
    raw = {"modules": {"drive": "drive"}}
    first = CfgEnv(md=MetaDict(), ml=_ml_with_drive(1000.0), device_manager=manager)
    second = CfgEnv(md=MetaDict(), ml=_ml_with_drive(2000.0), device_manager=manager)
    cfg_a = make_cfg(raw, _PulseExperimentCfg, first)
    cfg_b = make_cfg(raw, _PulseExperimentCfg, second, overrides={"reps": 3})
    assert cfg_a.modules.drive.freq == pytest.approx(1000.0)
    assert cfg_b.modules.drive.freq == pytest.approx(2000.0)
    assert cfg_b.reps == 3
    assert raw == {"modules": {"drive": "drive"}}
