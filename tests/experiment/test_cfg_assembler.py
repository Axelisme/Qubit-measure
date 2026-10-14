from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    DirectReadoutCfg,
    FlatTopWaveformSpec,
    PulseCfg,
    PulseReadoutSpec,
    PulseSpec,
    waveform_spec_from_raw,
)
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.cfg_assembler import (
    CfgEnv,
    ExperimentCfgInput,
    ExperimentCfgValue,
    assemble_experiment_cfg,
    make_cfg,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.program.v2.modules import ModuleCfg as NativeModuleCfg
from zcu_tools.program.v2.modules import ModuleCfgFactory as NativeModuleCfgFactory
from zcu_tools.program.v2.modules.pulse import PulseCfg as NativePulseCfg
from zcu_tools.program.v2.modules.readout import (
    PulseReadoutCfg as NativePulseReadoutCfg,
)
from zcu_tools.program.v2.modules.waveform import ArbWaveformCfg as NativeArbWaveformCfg
from zcu_tools.program.v2.modules.waveform import (
    FlatTopWaveformCfg as NativeFlatTopWaveformCfg,
)
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


class _DeviceOnlyCfg(ExpCfgModel):
    dev: Mapping[str, str] | None = None


class _PulseModules(ConfigBase):
    drive: NativePulseCfg


class _PulseExperimentCfg(ExpCfgModel):
    reps: int = 1
    modules: _PulseModules


class _NativeModules(ConfigBase):
    drive: NativeModuleCfg


class _NativeExperimentCfg(ExpCfgModel):
    modules: _NativeModules


@pytest.fixture(scope="module", autouse=True)
def repository_state_guard() -> Iterator[None]:
    before = vars(ArbWaveformDatabase)["_database_path"]
    yield
    assert vars(ArbWaveformDatabase)["_database_path"] == before


@pytest.mark.parametrize(
    ("family", "path"),
    [
        ("pulse", ("waveform",)),
        ("readout/pulse", ("pulse_cfg", "waveform")),
        ("reset/pulse", ("pulse_cfg", "waveform")),
        ("reset/two_pulse", ("pulse1_cfg", "waveform")),
        ("reset/bath", ("qubit_tone_cfg", "waveform")),
    ],
)
def test_assembly_converts_nested_arb_with_capture_bound_paths(
    tmp_path: Path, family: str, path: tuple[str, ...]
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    np.savez(first / "shape.npz", idata=[0.0, 0.5], qdata=[0.0, 0.0], time=[0.0, 1.0])
    captured = LibrarySnapshot(
        _library().document(), assets=ArbWaveformDatabase.info_at(first)
    )
    pulse = _pulse_raw(1000.0)
    pulse["waveform"] = {
        "style": "flat_top",
        "length": 3.0,
        "raise_waveform": {"style": "arb", "data": "shape"},
    }
    raw: dict[str, ExperimentCfgValue] = pulse
    if family == "readout/pulse":
        raw = {
            "type": family,
            "pulse_cfg": pulse,
            "ro_cfg": {
                "type": "readout/direct",
                "ro_ch": 0,
                "ro_length": 3.0,
                "trig_offset": 0.0,
            },
        }
    elif family == "reset/pulse":
        raw = {"type": family, "pulse_cfg": pulse}
    elif family == "reset/two_pulse":
        raw = {"type": family, "pulse1_cfg": pulse, "pulse2_cfg": pulse}
    elif family == "reset/bath":
        raw = {
            "type": family,
            "qubit_tone_cfg": pulse,
            "cavity_tone_cfg": pulse,
            "pi2_cfg": pulse,
        }
    fields = ExperimentCfgInput({"modules": {"drive": raw}})
    cfg = assemble_experiment_cfg(
        fields,
        _NativeExperimentCfg,
        context=captured.validation_context(),
    )
    # Inspect the public serialized native cfg, not factory internals.
    restored = NativeModuleCfgFactory.from_raw(cfg.modules.drive.to_dict())
    native = restored.to_dict()
    for part in (*path, "raise_waveform"):
        child = native[part]
        assert isinstance(child, dict)
        native = child
    arb = NativeArbWaveformCfg.model_validate(native)
    assert arb.path == (first / "shape.npz").resolve().as_posix()
    assert arb.data == "shape" and arb.length == 1.0
    assert pulse["waveform"] == {
        "style": "flat_top",
        "length": 3.0,
        "raise_waveform": {"style": "arb", "data": "shape"},
    }


@pytest.mark.parametrize("readout", [False, True])
@pytest.mark.parametrize("flat_top", [False, True])
def test_library_arb_assembly_json_roundtrip_needs_no_provider(
    tmp_path: Path, readout: bool, flat_top: bool
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    np.savez(first / "shape.npz", idata=[0.0, 0.5], qdata=[0.0, 0.0], time=[0.0, 1.0])
    ramp = waveform_spec_from_raw({"style": "arb", "data": "shape"})
    waveform = (
        FlatTopWaveformSpec(length=3.0, raise_waveform="Q1.waveforms.ramp")
        if flat_top
        else ramp
    )
    pulse = PulseSpec(
        ch=0, nqz=1, freq=1000.0, gain=0.1, waveform="Q1.waveforms.envelope"
    )
    component = ComponentLibrary(
        waveforms={"ramp": ramp, "envelope": waveform},
        pulses={"drive": pulse},
        readout={
            "measure": PulseReadoutSpec(
                pulse_cfg="Q1.pulses.drive",
                ro_cfg=DirectReadoutCfg(
                    ro_ch=0, ro_freq=1000.0, ro_length=3.0, trig_offset=0.0
                ),
            )
        },
    )
    captured = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="1.0",
            components={"Q1": component},
        ),
        assets=ArbWaveformDatabase.info_at(first),
    )
    cfg = assemble_experiment_cfg(
        ExperimentCfgInput(
            {
                "modules": {
                    "drive": "Q1.readout.measure" if readout else "Q1.pulses.drive"
                }
            }
        ),
        _NativeExperimentCfg,
        context=captured.validation_context(),
    )
    restored = _NativeExperimentCfg.model_validate_json(cfg.model_dump_json())
    module = restored.modules.drive
    if isinstance(module, NativePulseReadoutCfg):
        module = module.pulse_cfg
    assert isinstance(module, NativePulseCfg)
    envelope = module.waveform
    if isinstance(envelope, NativeFlatTopWaveformCfg):
        assert envelope.length == 3.0
        envelope = envelope.raise_waveform
    assert isinstance(envelope, NativeArbWaveformCfg)
    assert envelope.model_dump() == {
        "style": "arb",
        "data": "shape",
        "path": (tmp_path / "first" / "shape.npz").resolve().as_posix(),
    }
    assert envelope.length == 1.0
    np.savez(envelope.path, time=[0, 2], idata=[0, 0.7], qdata=[0, 0])
    assert envelope.length == 2.0


def test_assembly_rejects_arb_without_assets() -> None:
    pulse = _pulse_raw(1000.0)
    pulse["waveform"] = {"style": "arb", "data": "missing"}
    with pytest.raises(ValueError, match="assets"):
        assemble_experiment_cfg(
            ExperimentCfgInput({"modules": {"drive": pulse}}),
            _PulseExperimentCfg,
            context=None,
        )


class _SingleSweep(ConfigBase):
    gain: NDArray[np.float64]


class _SingleSweepExperimentCfg(ExpCfgModel):
    sweep: _SingleSweep


class _NamedSweeps(ConfigBase):
    gain: NDArray[np.float64]
    delay: NDArray[np.float64]


class _NamedSweepExperimentCfg(ExpCfgModel):
    sweep: _NamedSweeps


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


def test_assemble_experiment_cfg_uses_ml_from_each_call() -> None:
    fields = ExperimentCfgInput({"modules": {"drive": "Q1.pulses.drive"}})

    cfg_a = assemble_experiment_cfg(
        fields,
        _PulseExperimentCfg,
        context=_library(1000.0).validation_context(),
    )
    cfg_b = assemble_experiment_cfg(
        fields,
        _PulseExperimentCfg,
        context=_library(2000.0).validation_context(),
    )

    assert cfg_a.modules.drive.freq == pytest.approx(1000.0)
    assert cfg_b.modules.drive.freq == pytest.approx(2000.0)


def test_make_cfg_uses_each_environment_library_and_overrides() -> None:
    manager = DeviceManager()
    raw: dict[str, ExperimentCfgValue] = {"modules": {"drive": "Q1.pulses.drive"}}
    first = CfgEnv(ml=_library(1000.0), device_manager=manager)
    second = CfgEnv(ml=_library(2000.0), device_manager=manager)
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
    cfg = assemble_experiment_cfg(fields, _PulseExperimentCfg, context=None)
    assert cfg.modules.drive.freq == pytest.approx(1000.0)
    assert cfg.modules.drive.waveform.length == pytest.approx(1.0)
    cfg.modules.drive.freq = 2000.0
    again = assemble_experiment_cfg(fields, _PulseExperimentCfg, context=None)
    assert again.modules.drive.freq == pytest.approx(1000.0)


def test_module_reference_without_context_preserves_resolution_error() -> None:
    with pytest.raises(ValueError, match="Q1.pulses.drive"):
        assemble_experiment_cfg(
            ExperimentCfgInput({"modules": {"drive": "Q1.pulses.drive"}}),
            _PulseExperimentCfg,
            context=None,
        )


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ({"dev": 1}, "field 'dev' must be a mapping"),
        ({"dev": {"flux": 1}}, "name for role 'flux' must be a string"),
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
        )


def test_validation_failure_retains_target_name_and_cause() -> None:
    with pytest.raises(ValueError, match="_PulseExperimentCfg") as failure:
        assemble_experiment_cfg(
            ExperimentCfgInput({"reps": "invalid"}),
            _PulseExperimentCfg,
            context=None,
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

    first = assemble_experiment_cfg(fields, _SingleSweepExperimentCfg, context=None)
    np.testing.assert_array_equal(first.sweep.gain, [0.1, 0.2])
    first.sweep.gain[0] = 8.0
    second = assemble_experiment_cfg(fields, _SingleSweepExperimentCfg, context=None)
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
    cfg = assemble_experiment_cfg(fields, _NamedSweepExperimentCfg, context=None)
    np.testing.assert_array_equal(cfg.sweep.gain, [0.1, 0.2])
    np.testing.assert_array_equal(cfg.sweep.delay, [1.0, 2.0])


def test_make_cfg_preserves_role_names_without_hardware_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DeviceManager()
    device = FakeDevice()
    manager.register_device("flux_source", device)

    info = device.get_info()
    reads: list[str] = []

    def recording_info() -> FakeDeviceInfo:
        reads.append("flux_source")
        return info

    monkeypatch.setattr(device, "get_info", recording_info)
    env = CfgEnv(ml=_library(), device_manager=manager)
    make_cfg({}, _DeviceOnlyCfg, env)
    assert reads == []
    raw: dict[str, ExperimentCfgValue] = {"dev": {"flux_dev": "flux_source"}}
    cfg = make_cfg(raw, _DeviceOnlyCfg, env)
    assert reads == []
    assert cfg.dev == {"flux_dev": "flux_source"}
    assert cfg.model_dump(mode="json")["dev"] == {"flux_dev": "flux_source"}
    assert raw == {"dev": {"flux_dev": "flux_source"}}
    assert device.get_value() == 0.0
    assert device.get_output() == "off"


def test_make_cfg_without_roles_does_not_inject_connected_settings() -> None:
    manager = DeviceManager()
    device = FakeDevice()
    manager.register_device("unrelated", device)
    env = CfgEnv(ml=_library(), device_manager=manager)
    cfg = make_cfg({}, _DeviceOnlyCfg, env)
    assert cfg.dev is None
    assert device.get_value() == 0.0
