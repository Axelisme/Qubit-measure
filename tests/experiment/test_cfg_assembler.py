from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, ConstWaveformCfg, PulseCfg, PulseSpec
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.cfg_assembler import (
    CfgEnv,
    ExperimentCfgInput,
    ExperimentCfgValue,
    assemble_experiment_cfg,
    make_cfg,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.resources.context import MetaDict
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class _DeviceOnlyCfg(ExpCfgModel):
    pass


class _PulseModules(ConfigBase):
    drive: PulseCfg


class _PulseExperimentCfg(ExpCfgModel):
    reps: int = 1
    modules: _PulseModules


class _SingleSweep(ConfigBase):
    gain: NDArray[np.float64]


class _SingleSweepExperimentCfg(ExpCfgModel):
    sweep: _SingleSweep


class _NamedSweeps(ConfigBase):
    gain: NDArray[np.float64]
    delay: NDArray[np.float64]


class _NamedSweepExperimentCfg(ExpCfgModel):
    sweep: _NamedSweeps


def _device(value: float) -> FakeDeviceInfo:
    return FakeDeviceInfo(address="fake", value=value)


def _assert_fake_device_value(cfg: _DeviceOnlyCfg, name: str, value: float) -> None:
    assert cfg.dev is not None
    dev = cfg.dev[name]
    assert isinstance(dev, FakeDeviceInfo)
    assert dev.value == pytest.approx(value)


def _pulse_raw(freq: float) -> dict[str, ExperimentCfgValue]:
    return {
        "type": "pulse",
        "waveform": {"style": "const", "length": 1.0},
        "ch": 0,
        "nqz": 1,
        "freq": freq,
        "gain": 0.1,
    }


def _library(freq: float | None = None) -> LibrarySnapshot:
    components = {}
    if freq is not None:
        components["Q1"] = ComponentLibrary(
            pulses={
                "drive": PulseSpec(
                    ch=0,
                    nqz=1,
                    freq=freq,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=1.0),
                )
            }
        )
    return LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library", format_version="1.0", components=components
        )
    )


def test_assemble_experiment_cfg_uses_explicit_device_snapshot() -> None:
    snapshot = {"flux": _device(1.0)}

    cfg = assemble_experiment_cfg(
        ExperimentCfgInput({"dev": {"flux": {"value": 2.5, "output": "on"}}}),
        _DeviceOnlyCfg,
        context=None,
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
    env = CfgEnv(md=MetaDict(), ml=_library(), device_manager=manager)
    raw: dict[str, ExperimentCfgValue] = {"dev": {"flux": {"output": "on"}}}

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
    env = CfgEnv(md=MetaDict(), ml=_library(), device_manager=manager)
    make_cfg({}, _DeviceOnlyCfg, env)

    def failed_read() -> FakeDeviceInfo:
        raise RuntimeError("device read failed")

    monkeypatch.setattr(device, "get_info", failed_read)
    with pytest.raises(RuntimeError, match="device read failed"):
        make_cfg({}, _DeviceOnlyCfg, env)


def test_assemble_experiment_cfg_uses_ml_from_each_call() -> None:
    fields = ExperimentCfgInput({"modules": {"drive": "Q1.pulses.drive"}})

    cfg_a = assemble_experiment_cfg(
        fields,
        _PulseExperimentCfg,
        context=_library(1000.0).validation_context(),
        device_snapshot={},
    )
    cfg_b = assemble_experiment_cfg(
        fields,
        _PulseExperimentCfg,
        context=_library(2000.0).validation_context(),
        device_snapshot={},
    )

    assert cfg_a.modules.drive.freq == pytest.approx(1000.0)
    assert cfg_b.modules.drive.freq == pytest.approx(2000.0)


def test_make_cfg_uses_each_environment_library_and_overrides() -> None:
    manager = DeviceManager()
    raw: dict[str, ExperimentCfgValue] = {"modules": {"drive": "Q1.pulses.drive"}}
    first = CfgEnv(md=MetaDict(), ml=_library(1000.0), device_manager=manager)
    second = CfgEnv(md=MetaDict(), ml=_library(2000.0), device_manager=manager)
    cfg_a = make_cfg(raw, _PulseExperimentCfg, first)
    cfg_b = make_cfg(raw, _PulseExperimentCfg, second, overrides={"reps": 3})
    assert cfg_a.modules.drive.freq == pytest.approx(1000.0)
    assert cfg_b.modules.drive.freq == pytest.approx(2000.0)
    assert cfg_b.reps == 3
    assert raw == {"modules": {"drive": "Q1.pulses.drive"}}


def test_input_preserves_runtime_tokens_and_isolates_recursive_containers() -> None:
    samples = np.array([0.1, 0.2], dtype=np.float64)
    child: list[ExperimentCfgValue] = [2.0]
    pulse = PulseSpec(
        ch=0, nqz=1, freq=1000.0, gain=0.1, waveform=ConstWaveformCfg(length=1.0)
    )
    original: dict[str, ExperimentCfgValue] = {
        "nested": MappingProxyType({"range": (1.0, child)}),
        "samples": samples,
        "module": pulse,
    }
    fields = ExperimentCfgInput(MappingProxyType(original))
    samples[0] = 9.0
    child.append(3.0)
    pulse.freq = 2000.0

    first = fields.to_raw()
    np.testing.assert_array_equal(first["samples"], [0.1, 0.2])
    assert first["nested"] == {"range": (1.0, [2.0])}
    copied_pulse = first["module"]
    assert isinstance(copied_pulse, PulseSpec)
    assert copied_pulse.freq == pytest.approx(1000.0)
    copied_pulse.freq = 3000.0
    copied_samples = first["samples"]
    assert isinstance(copied_samples, np.ndarray)
    copied_samples[0] = 8.0
    first["nested"] = None

    second = fields.to_raw()
    np.testing.assert_array_equal(second["samples"], [0.1, 0.2])
    assert second["nested"] == {"range": (1.0, [2.0])}
    second_pulse = second["module"]
    assert isinstance(second_pulse, PulseSpec)
    assert second_pulse.freq == pytest.approx(1000.0)


@pytest.mark.parametrize("token_kind", ["mapping", "waveform", "spec", "cfg"])
def test_inline_module_tokens_do_not_require_a_library(token_kind: str) -> None:
    token: ExperimentCfgValue = _pulse_raw(1000.0)
    if token_kind == "waveform":
        raw = _pulse_raw(1000.0)
        raw["waveform"] = ConstWaveformCfg(length=1.0)
        token = raw
    elif token_kind == "spec":
        token = PulseSpec(
            ch=0, nqz=1, freq=1000.0, gain=0.1, waveform=ConstWaveformCfg(length=1.0)
        )
    elif token_kind == "cfg":
        token = PulseCfg(
            ch=0, nqz=1, freq=1000.0, gain=0.1, waveform=ConstWaveformCfg(length=1.0)
        )
    fields = ExperimentCfgInput({"modules": {"drive": token}})
    cfg = assemble_experiment_cfg(
        fields, _PulseExperimentCfg, context=None, device_snapshot={}
    )
    assert cfg.modules.drive.freq == pytest.approx(1000.0)
    assert cfg.modules.drive.waveform.length == pytest.approx(1.0)
    cfg.modules.drive.freq = 2000.0
    again = assemble_experiment_cfg(
        fields, _PulseExperimentCfg, context=None, device_snapshot={}
    )
    assert again.modules.drive.freq == pytest.approx(1000.0)


def test_module_reference_without_context_preserves_resolution_error() -> None:
    with pytest.raises(ValueError, match="Q1.pulses.drive"):
        assemble_experiment_cfg(
            ExperimentCfgInput({"modules": {"drive": "Q1.pulses.drive"}}),
            _PulseExperimentCfg,
            context=None,
            device_snapshot={},
        )


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ({"dev": 1}, "field 'dev' must be a mapping"),
        ({"dev": {"flux": 1}}, "patch for 'flux' must be a mapping"),
        ({"modules": 1}, "field 'modules' must be a mapping"),
        ({"modules": {"drive": 1}}, "modules.drive"),
    ],
)
def test_invalid_assembly_tokens_fail_at_their_path(
    values: dict[str, ExperimentCfgValue], reason: str
) -> None:
    with pytest.raises(TypeError, match=reason):
        assemble_experiment_cfg(
            ExperimentCfgInput(values),
            _PulseExperimentCfg,
            context=None,
            device_snapshot={"flux": _device(1.0)},
        )


def test_unknown_device_patch_does_not_add_a_device() -> None:
    with pytest.raises(KeyError, match="not found in device snapshot"):
        assemble_experiment_cfg(
            ExperimentCfgInput({"dev": {"missing": {"value": 2.0}}}),
            _DeviceOnlyCfg,
            context=None,
            device_snapshot={"flux": _device(1.0)},
        )


def test_validation_failure_retains_target_name_and_cause() -> None:
    with pytest.raises(ValueError, match="_PulseExperimentCfg") as failure:
        assemble_experiment_cfg(
            ExperimentCfgInput({"reps": "invalid"}),
            _PulseExperimentCfg,
            context=None,
            device_snapshot={},
        )
    assert isinstance(failure.value.__cause__, ValueError)


@pytest.mark.parametrize("representation", ["list", "array", "named"])
def test_single_sweep_preserves_values_and_capture_isolation(
    representation: str,
) -> None:
    samples = np.array([0.1, 0.2], dtype=np.float64)
    sweep: ExperimentCfgValue = [0.1, 0.2]
    if representation == "array":
        sweep = samples
    elif representation == "named":
        sweep = {"gain": samples}
    fields = ExperimentCfgInput({"sweep": sweep})
    samples[0] = 9.0

    first = assemble_experiment_cfg(
        fields, _SingleSweepExperimentCfg, context=None, device_snapshot={}
    )
    np.testing.assert_array_equal(first.sweep.gain, [0.1, 0.2])
    first.sweep.gain[0] = 8.0
    second = assemble_experiment_cfg(
        fields, _SingleSweepExperimentCfg, context=None, device_snapshot={}
    )
    np.testing.assert_array_equal(second.sweep.gain, [0.1, 0.2])


def test_named_multi_sweep_is_not_single_sweep_normalized() -> None:
    fields = ExperimentCfgInput(
        {
            "sweep": {
                "gain": np.array([0.1, 0.2], dtype=np.float64),
                "delay": np.array([1.0, 2.0], dtype=np.float64),
            }
        }
    )
    cfg = assemble_experiment_cfg(
        fields, _NamedSweepExperimentCfg, context=None, device_snapshot={}
    )
    np.testing.assert_array_equal(cfg.sweep.gain, [0.1, 0.2])
    np.testing.assert_array_equal(cfg.sweep.delay, [1.0, 2.0])
