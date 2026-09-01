from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.v2.singleshot.len_rabi import (
    LenRabiCfg,
    LenRabiExp,
    LenRabiModuleCfg,
    LenRabiResult,
    LenRabiSweepCfg,
)
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import load_labber_data, save_labber_data

from script.migrate_experiment_data import migrate_experiment_data


def _pulse() -> PulseCfg:
    return PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=1.0),
        ch=1,
        nqz=1,
        freq=3000.0,
        gain=0.2,
    )


def _cfg() -> LenRabiCfg:
    return LenRabiCfg(
        reps=4,
        rounds=1,
        shots=4,
        modules=LenRabiModuleCfg(
            reset=None,
            qub_pulse=_pulse(),
            readout=DirectReadoutCfg(
                type="readout/direct",
                ro_ch=0,
                ro_length=1.0,
                ro_freq=6000.0,
                gen_ch=0,
            ),
        ),
        sweep=LenRabiSweepCfg(
            length=SweepCfg(start=0.1, stop=0.3, step=0.1, expts=3),
        ),
    )


def _sample_result(*, with_cfg: bool = True) -> LenRabiResult:
    lengths = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    shot_indices = np.arange(4, dtype=np.int64)
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    signals = (values + 1j * (values + 0.25)).astype(np.complex128)
    return LenRabiResult(
        lengths=lengths,
        shot_indices=shot_indices,
        signals=signals,
        cfg_snapshot=_cfg() if with_cfg else None,
    )


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}.hdf5"


def _write_legacy_population_file(path: Path) -> None:
    lengths = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    populations = np.array([[0.8, 0.1], [0.5, 0.4], [0.2, 0.7]], dtype=np.float64)
    save_labber_data(
        str(path),
        z=("Population", "a.u.", populations.T),
        axes=[
            ("Length", "s", lengths * 1e-6),
            ("GE population", "a.u.", np.array([0, 1], dtype=np.int64)),
        ],
        comment="legacy comment",
        tags="singleshot/len_rabi",
    )


def test_len_rabi_raw_labber_axes_tag_shape_dtype_and_units(tmp_path: Path) -> None:
    result = _sample_result()

    LenRabiExp().save(str(tmp_path / "len_rabi"), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, "len_rabi")))

    assert [axis.name for axis in raw.axes] == ["Shot Index", "Length"]
    assert [axis.unit for axis in raw.axes] == ["None", "s"]
    np.testing.assert_array_equal(raw.axes[0].values, result.shot_indices)
    np.testing.assert_allclose(raw.axes[1].values, result.lengths * 1e-6, rtol=0)
    assert raw.data.name == "Signal"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == result.signals.shape == (3, 4)
    assert raw.z.dtype == np.complex128
    np.testing.assert_array_equal(raw.z, result.signals)
    assert raw.tags == ["singleshot/len_rabi"]


def test_len_rabi_save_load_roundtrip_with_cfg(tmp_path: Path) -> None:
    result = _sample_result()

    LenRabiExp().save(str(tmp_path / "roundtrip"), result=result, comment="note")
    load_exp = LenRabiExp()
    loaded = load_exp.load(str(_saved_path(tmp_path, "roundtrip")))

    np.testing.assert_allclose(loaded.lengths, result.lengths, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(loaded.shot_indices, result.shot_indices)
    np.testing.assert_array_equal(loaded.signals, result.signals)
    assert loaded.lengths.dtype == np.float64
    assert loaded.shot_indices.dtype == np.int64
    assert loaded.signals.dtype == np.complex128
    assert loaded.signals.shape == (3, 4)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.shots == 4
    assert loaded.cfg_snapshot.rounds == 1
    assert loaded.cfg_snapshot.reps == 4


def test_len_rabi_save_fast_fails_wrong_raw_shape(tmp_path: Path) -> None:
    result = _sample_result()
    invalid = LenRabiResult(
        lengths=result.lengths,
        shot_indices=result.shot_indices,
        signals=result.signals.T,
        cfg_snapshot=_cfg(),
    )

    with pytest.raises(ValueError, match="axis 'Shot Index'.*z dim"):
        LenRabiExp().save(str(tmp_path / "bad_shape"), result=invalid)


def test_len_rabi_save_fast_fails_without_cfg_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cfg_snapshot"):
        LenRabiExp().save(
            str(tmp_path / "no_cfg"), result=_sample_result(with_cfg=False)
        )


def test_len_rabi_runtime_load_rejects_population_only_layout(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_len_rabi.hdf5"
    _write_legacy_population_file(legacy_path)

    with pytest.raises(ValueError, match="axis 0 label"):
        LenRabiExp().load(str(legacy_path))


def test_len_rabi_population_only_migration_refuses_to_invent_raw_iq(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy_len_rabi.hdf5"
    output = tmp_path / "canonical.hdf5"
    _write_legacy_population_file(legacy_path)
    legacy_bytes = legacy_path.read_bytes()

    with pytest.raises(ValueError, match="cannot be migrated.*raw-IQ"):
        migrate_experiment_data(
            experiment="singleshot/len_rabi",
            input_path=legacy_path,
            output_path=output,
        )

    assert legacy_path.read_bytes() == legacy_bytes
    assert not output.exists()
