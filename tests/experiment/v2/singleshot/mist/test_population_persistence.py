from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pytest
from zcu_tools.experiment.v2.singleshot.mist.freq import (
    FreqCfg,
    FreqDepExp,
    FreqModuleCfg,
    FreqResult,
    FreqSweepCfg,
)
from zcu_tools.experiment.v2.singleshot.mist.power import (
    PowerCfg,
    PowerExp,
    PowerModuleCfg,
    PowerResult,
    PowerSweepCfg,
)
from zcu_tools.experiment.v2.singleshot.mist.pre_freq import (
    PreFreqCfg,
    PreFreqExp,
    PreFreqModuleCfg,
    PreFreqResult,
    PreFreqSweepCfg,
)
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import LabberData, load_labber_data, save_labber_data

from script.migrate_experiment_data import migrate_experiment_data

MistCfg: TypeAlias = PowerCfg | FreqCfg | PreFreqCfg
MistResult: TypeAlias = PowerResult | FreqResult | PreFreqResult


@dataclass(frozen=True)
class MistCase:
    name: str
    exp_cls: type[PowerExp] | type[FreqDepExp] | type[PreFreqExp]
    sweep_axis_label: str
    sweep_axis_unit: str
    disk_scale: float
    tag: str
    legacy_tag: str
    converter_id: str


CASES = (
    MistCase(
        name="power",
        exp_cls=PowerExp,
        sweep_axis_label="Drive gain",
        sweep_axis_unit="a.u.",
        disk_scale=1.0,
        tag="singleshot/mist/power",
        legacy_tag="singleshot/mist/gain",
        converter_id="singleshot/mist/power",
    ),
    MistCase(
        name="freq",
        exp_cls=FreqDepExp,
        sweep_axis_label="Drive Freq",
        sweep_axis_unit="Hz",
        disk_scale=1e6,
        tag="singleshot/mist/freq",
        legacy_tag="singleshot/mist/freq",
        converter_id="singleshot/mist/freq",
    ),
    MistCase(
        name="pre_freq",
        exp_cls=PreFreqExp,
        sweep_axis_label="PrePulse frequency",
        sweep_axis_unit="Hz",
        disk_scale=1e6,
        tag="singleshot/mist/pre_freq",
        legacy_tag="singleshot/mist/pre_freq",
        converter_id="singleshot/mist/pre_freq",
    ),
)


def _pulse(
    *,
    ch: int = 0,
    freq: float = 100.0,
    gain: float = 0.1,
    length: float = 1.0,
) -> PulseCfg:
    return PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=length),
        ch=ch,
        nqz=1,
        freq=freq,
        gain=gain,
    )


def _readout() -> DirectReadoutCfg:
    return DirectReadoutCfg(
        type="readout/direct",
        ro_ch=0,
        ro_length=1.0,
        ro_freq=6000.0,
        gen_ch=0,
    )


def _sweep_cfg() -> SweepCfg:
    return SweepCfg(start=0.1, stop=0.3, step=0.1, expts=3)


def _freq_sweep_cfg() -> SweepCfg:
    return SweepCfg(start=4100.0, stop=4300.0, step=100.0, expts=3)


def _power_cfg() -> PowerCfg:
    return PowerCfg(
        reps=1,
        rounds=1,
        modules=PowerModuleCfg(
            reset=None,
            init_pulse=None,
            probe_pulse=_pulse(ch=1, freq=3000.0, gain=0.2),
            readout=_readout(),
        ),
        sweep=PowerSweepCfg(gain=_sweep_cfg()),
    )


def _freq_cfg() -> FreqCfg:
    return FreqCfg(
        reps=1,
        rounds=1,
        modules=FreqModuleCfg(
            reset=None,
            init_pulse=None,
            probe_pulse=_pulse(ch=1, freq=4100.0, gain=0.2),
            readout=_readout(),
        ),
        sweep=FreqSweepCfg(freq=_freq_sweep_cfg()),
    )


def _pre_freq_cfg() -> PreFreqCfg:
    return PreFreqCfg(
        reps=1,
        rounds=1,
        modules=PreFreqModuleCfg(
            reset=None,
            init_pulse=_pulse(ch=1, freq=4100.0, gain=0.2),
            pi_pulse=None,
            probe_pulse=_pulse(ch=2, freq=3000.0, gain=0.3),
            readout=_readout(),
        ),
        sweep=PreFreqSweepCfg(freq=_freq_sweep_cfg()),
    )


def _cfg(case: MistCase) -> MistCfg:
    if case.name == "power":
        return _power_cfg()
    if case.name == "freq":
        return _freq_cfg()
    if case.name == "pre_freq":
        return _pre_freq_cfg()
    raise ValueError(f"unknown MIST case: {case.name}")


def _signals() -> np.ndarray:
    return np.array(
        [
            [0.8, 0.1],
            [0.5, 0.4],
            [0.2, 0.7],
        ],
        dtype=np.float64,
    )


def _sample_result(
    case: MistCase,
    *,
    with_cfg: bool = True,
    omit_population_states: bool = False,
) -> MistResult:
    population_states = np.array([0, 1], dtype=np.int64)
    if case.name == "power":
        gains = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        cfg_snapshot = _power_cfg() if with_cfg else None
        if omit_population_states:
            return PowerResult(
                gains=gains,
                signals=_signals(),
                cfg_snapshot=cfg_snapshot,
            )
        return PowerResult(
            gains=gains,
            signals=_signals(),
            population_states=population_states,
            cfg_snapshot=cfg_snapshot,
        )
    if case.name == "freq":
        freqs = np.array([4100.0, 4200.0, 4300.0], dtype=np.float64)
        cfg_snapshot = _freq_cfg() if with_cfg else None
        if omit_population_states:
            return FreqResult(
                freqs=freqs,
                signals=_signals(),
                cfg_snapshot=cfg_snapshot,
            )
        return FreqResult(
            freqs=freqs,
            signals=_signals(),
            population_states=population_states,
            cfg_snapshot=cfg_snapshot,
        )
    if case.name == "pre_freq":
        freqs = np.array([4100.0, 4200.0, 4300.0], dtype=np.float64)
        cfg_snapshot = _pre_freq_cfg() if with_cfg else None
        if omit_population_states:
            return PreFreqResult(
                freqs=freqs,
                signals=_signals(),
                cfg_snapshot=cfg_snapshot,
            )
        return PreFreqResult(
            freqs=freqs,
            signals=_signals(),
            population_states=population_states,
            cfg_snapshot=cfg_snapshot,
        )
    raise ValueError(f"unknown MIST case: {case.name}")


def _exp(case: MistCase) -> Any:
    return case.exp_cls()


def _sweep_values(result: MistResult) -> np.ndarray:
    if isinstance(result, PowerResult):
        return result.gains
    return result.freqs


def _cfg_expts(loaded: MistResult) -> int:
    cfg = loaded.cfg_snapshot
    assert cfg is not None
    if isinstance(cfg, PowerCfg):
        return cfg.sweep.gain.expts
    return cfg.sweep.freq.expts


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}.hdf5"


def _legacy_sweep_axis(
    case: MistCase,
    result: MistResult,
) -> tuple[str, str, np.ndarray]:
    return (
        case.sweep_axis_label,
        case.sweep_axis_unit,
        _sweep_values(result) * case.disk_scale,
    )


def _write_legacy_population(
    path: Path,
    case: MistCase,
    result: MistResult,
    *,
    axes: list[tuple[str, str, np.ndarray]] | None = None,
    z: tuple[str, str, np.ndarray] | None = None,
    comment: str = "legacy comment",
    tags: str | None = None,
) -> None:
    save_labber_data(
        str(path),
        z=z or ("Population", "a.u.", result.signals.T),
        axes=axes
        or [
            _legacy_sweep_axis(case, result),
            ("GE population", "a.u.", result.population_states),
        ],
        comment=comment,
        tags=tags or case.legacy_tag,
    )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_mist_population_raw_labber_axes_tag_shape_and_disk_units(
    tmp_path: Path,
    case: MistCase,
) -> None:
    result = _sample_result(case)

    _exp(case).save(str(tmp_path / case.name), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, case.name)))

    assert [axis.name for axis in raw.axes] == [
        "GE Population",
        case.sweep_axis_label,
    ]
    assert [axis.unit for axis in raw.axes] == ["None", case.sweep_axis_unit]
    np.testing.assert_array_equal(raw.axes[0].values, result.population_states)
    np.testing.assert_allclose(
        raw.axes[1].values,
        _sweep_values(result) * case.disk_scale,
        rtol=0,
    )
    assert raw.data.name == "Population"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (len(_sweep_values(result)), len(result.population_states))
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)
    assert raw.tags == [case.tag]


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_mist_population_save_load_roundtrip_with_cfg(
    tmp_path: Path,
    case: MistCase,
) -> None:
    result = _sample_result(case)

    _exp(case).save(str(tmp_path / "roundtrip"), result=result, comment="note")
    load_exp = _exp(case)
    loaded = load_exp.load(str(_saved_path(tmp_path, "roundtrip")))

    np.testing.assert_allclose(
        _sweep_values(loaded),
        _sweep_values(result),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    assert _sweep_values(loaded).dtype == np.float64
    assert loaded.population_states.dtype == np.int64
    assert loaded.signals.shape == result.signals.shape == (3, 2)
    assert loaded.signals.dtype == np.float64
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert _cfg_expts(loaded) == 3


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_mist_population_save_fast_fails_on_legacy_shape(
    tmp_path: Path,
    case: MistCase,
) -> None:
    result = _sample_result(case)
    if isinstance(result, PowerResult):
        bad_result = PowerResult(
            gains=result.gains,
            signals=result.signals.T,
            population_states=result.population_states,
            cfg_snapshot=_power_cfg(),
        )
    elif isinstance(result, FreqResult):
        bad_result = FreqResult(
            freqs=result.freqs,
            signals=result.signals.T,
            population_states=result.population_states,
            cfg_snapshot=_freq_cfg(),
        )
    else:
        bad_result = PreFreqResult(
            freqs=result.freqs,
            signals=result.signals.T,
            population_states=result.population_states,
            cfg_snapshot=_pre_freq_cfg(),
        )

    with pytest.raises(ValueError):
        _exp(case).save(str(tmp_path / "bad_shape"), result=bad_result)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_mist_population_save_fast_fails_without_cfg_snapshot(
    tmp_path: Path,
    case: MistCase,
) -> None:
    result = _sample_result(case, with_cfg=False)

    with pytest.raises(ValueError, match="cfg_snapshot"):
        _exp(case).save(str(tmp_path / "no_cfg"), result=result)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_mist_population_runtime_load_rejects_legacy_axis_order(
    tmp_path: Path,
    case: MistCase,
) -> None:
    legacy_path = tmp_path / f"legacy_{case.name}.hdf5"
    _write_legacy_population(legacy_path, case, _sample_result(case, with_cfg=False))

    with pytest.raises(ValueError, match="axis 0 label"):
        _exp(case).load(str(legacy_path))


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_mist_population_result_default_population_states_are_isolated(
    case: MistCase,
) -> None:
    first = _sample_result(case, with_cfg=False, omit_population_states=True)
    second = _sample_result(case, with_cfg=False, omit_population_states=True)

    expected = np.array([0, 1], dtype=np.int64)
    np.testing.assert_array_equal(first.population_states, expected)
    np.testing.assert_array_equal(second.population_states, expected)
    assert first.population_states.dtype == np.int64
    assert second.population_states.dtype == np.int64
    assert first.population_states is not second.population_states

    first.population_states[0] = -1
    assert second.population_states[0] == 0


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_migrate_mist_population_legacy_to_canonical_hdf5(
    tmp_path: Path,
    case: MistCase,
) -> None:
    result = _sample_result(case, with_cfg=False)
    legacy_path = tmp_path / f"legacy_{case.name}.hdf5"
    _write_legacy_population(legacy_path, case, result)
    legacy_bytes = legacy_path.read_bytes()

    migrated = migrate_experiment_data(
        experiment=case.converter_id,
        input_path=legacy_path,
        output_path=tmp_path / "canonical.hdf5",
    )

    assert migrated == tmp_path / "canonical.hdf5"
    assert legacy_path.read_bytes() == legacy_bytes
    loaded = _exp(case).load(str(migrated))
    np.testing.assert_allclose(
        _sweep_values(loaded),
        _sweep_values(result),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.cfg_snapshot is None
    raw = load_labber_data(str(migrated))
    assert [axis.name for axis in raw.axes] == [
        "GE Population",
        case.sweep_axis_label,
    ]
    assert [axis.unit for axis in raw.axes] == ["None", case.sweep_axis_unit]
    assert raw.z.shape == (len(_sweep_values(result)), len(result.population_states))
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)
    assert raw.comment == "legacy comment"
    assert raw.tags == [case.tag]


def test_migrate_mist_population_requires_overwrite_and_allows_overwrite(
    tmp_path: Path,
) -> None:
    case = CASES[0]
    result = _sample_result(case, with_cfg=False)
    legacy_path = tmp_path / "legacy_power.hdf5"
    _write_legacy_population(legacy_path, case, result)
    output = tmp_path / "canonical.hdf5"
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_path,
            output_path=output,
        )
    assert output.read_text() == "existing"

    migrated = migrate_experiment_data(
        experiment=case.converter_id,
        input_path=legacy_path,
        output_path=output,
        overwrite=True,
    )

    assert migrated == output
    loaded = _exp(case).load(str(output))
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)


def test_migrate_mist_population_does_not_register_gain_alias(tmp_path: Path) -> None:
    case = CASES[0]
    legacy_path = tmp_path / "legacy_power.hdf5"
    _write_legacy_population(
        legacy_path,
        case,
        _sample_result(case, with_cfg=False),
    )

    with pytest.raises(ValueError, match="unsupported experiment"):
        migrate_experiment_data(
            experiment="singleshot/mist/gain",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.parametrize(
    "axis_mutation",
    ["order", "name", "unit"],
)
def test_migrate_mist_population_rejects_wrong_sweep_axis_order_name_or_unit(
    tmp_path: Path,
    case: MistCase,
    axis_mutation: str,
) -> None:
    result = _sample_result(case, with_cfg=False)
    if axis_mutation == "order":
        axes = [
            ("GE population", "a.u.", result.population_states),
            _legacy_sweep_axis(case, result),
        ]
        z = ("Population", "a.u.", result.signals)
    elif axis_mutation == "name":
        axes = [
            (
                "Wrong Sweep",
                case.sweep_axis_unit,
                _sweep_values(result) * case.disk_scale,
            ),
            ("GE population", "a.u.", result.population_states),
        ]
        z = ("Population", "a.u.", result.signals.T)
    elif axis_mutation == "unit":
        axes = [
            (
                case.sweep_axis_label,
                "wrong",
                _sweep_values(result) * case.disk_scale,
            ),
            ("GE population", "a.u.", result.population_states),
        ]
        z = ("Population", "a.u.", result.signals.T)
    else:
        raise ValueError(f"unknown axis mutation: {axis_mutation}")
    legacy_path = tmp_path / f"legacy_bad_axis_{case.name}.hdf5"
    _write_legacy_population(legacy_path, case, result, axes=axes, z=z)

    with pytest.raises(ValueError, match="axis 0"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.parametrize(
    "population_axis",
    [
        ("Wrong Population", "a.u.", np.array([0, 1], dtype=np.int64)),
        ("GE population", "wrong", np.array([0, 1], dtype=np.int64)),
    ],
)
def test_migrate_mist_population_rejects_wrong_population_axis_name_or_unit(
    tmp_path: Path,
    case: MistCase,
    population_axis: tuple[str, str, np.ndarray],
) -> None:
    result = _sample_result(case, with_cfg=False)
    legacy_path = tmp_path / f"legacy_bad_population_axis_{case.name}.hdf5"
    _write_legacy_population(
        legacy_path,
        case,
        result,
        axes=[_legacy_sweep_axis(case, result), population_axis],
    )

    with pytest.raises(ValueError, match="axis 1"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_migrate_mist_population_rejects_wrong_population_values(
    tmp_path: Path,
    case: MistCase,
) -> None:
    result = _sample_result(case, with_cfg=False)
    legacy_path = tmp_path / f"legacy_bad_population_{case.name}.hdf5"
    _write_legacy_population(
        legacy_path,
        case,
        result,
        axes=[
            _legacy_sweep_axis(case, result),
            ("GE population", "a.u.", np.array([0, 2], dtype=np.int64)),
        ],
    )

    with pytest.raises(ValueError, match="population state"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.parametrize(
    "z",
    [
        ("Wrong Population", "a.u.", _signals().T),
        ("Population", "wrong", _signals().T),
    ],
)
def test_migrate_mist_population_rejects_wrong_z_channel(
    tmp_path: Path,
    case: MistCase,
    z: tuple[str, str, np.ndarray],
) -> None:
    legacy_path = tmp_path / f"legacy_bad_z_{case.name}.hdf5"
    _write_legacy_population(
        legacy_path,
        case,
        _sample_result(case, with_cfg=False),
        z=z,
    )

    with pytest.raises(ValueError, match="z channel"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_migrate_mist_population_rejects_wrong_z_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: MistCase,
) -> None:
    result = _sample_result(case, with_cfg=False)
    legacy_path = tmp_path / f"legacy_bad_shape_{case.name}.hdf5"
    legacy_path.write_text("placeholder")
    payload = LabberData(
        data=("Population", "a.u.", np.ones((2, 4), dtype=np.float64)),
        axes=[
            _legacy_sweep_axis(case, result),
            ("GE population", "a.u.", result.population_states),
        ],
    )
    monkeypatch.setattr(
        "script.migrate_experiment_data.load_labber_data",
        lambda _path: payload,
    )

    with pytest.raises(ValueError, match="z shape"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.parametrize("imaginary", [0.25, 1e-12])
def test_migrate_mist_population_rejects_non_real_z(
    tmp_path: Path,
    case: MistCase,
    imaginary: float,
) -> None:
    result = _sample_result(case, with_cfg=False)
    legacy_path = tmp_path / f"legacy_complex_population_{case.name}_{imaginary}.hdf5"
    z = (
        "Population",
        "a.u.",
        result.signals.T.astype(np.complex128) + 1j * imaginary,
    )
    _write_legacy_population(legacy_path, case, result, z=z)

    with pytest.raises(ValueError, match="imaginary"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )
