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
from zcu_tools.utils.datasaver import LabberData, load_labber_data, save_labber_data

from script.migrate_experiment_data import migrate_experiment_data


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


def _cfg() -> LenRabiCfg:
    return LenRabiCfg(
        reps=1,
        rounds=1,
        modules=LenRabiModuleCfg(
            reset=None,
            init_pulse=None,
            qub_pulse=_pulse(ch=1, freq=3000.0, gain=0.2),
            readout=_readout(),
        ),
        sweep=LenRabiSweepCfg(
            length=SweepCfg(start=0.1, stop=0.3, step=0.1, expts=3),
        ),
    )


def _sample_result(
    *,
    with_cfg: bool = True,
    omit_population_states: bool = False,
) -> LenRabiResult:
    lengths = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    signals = np.array(
        [
            [0.8, 0.1],
            [0.5, 0.4],
            [0.2, 0.7],
        ],
        dtype=np.float64,
    )
    cfg_snapshot = _cfg() if with_cfg else None
    if omit_population_states:
        return LenRabiResult(
            lengths=lengths,
            signals=signals,
            cfg_snapshot=cfg_snapshot,
        )
    return LenRabiResult(
        lengths=lengths,
        signals=signals,
        population_states=np.array([0, 1], dtype=np.int64),
        cfg_snapshot=cfg_snapshot,
    )


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}.hdf5"


def _write_legacy_len_rabi(
    path: Path,
    result: LenRabiResult,
    *,
    axes: list[tuple[str, str, np.ndarray]] | None = None,
    z: tuple[str, str, np.ndarray] | None = None,
    comment: str = "legacy comment",
    tags: str = "singleshot/len_rabi",
) -> None:
    save_labber_data(
        str(path),
        z=z or ("Population", "a.u.", result.signals.T),
        axes=axes
        or [
            ("Length", "s", result.lengths * 1e-6),
            ("GE population", "a.u.", result.population_states),
        ],
        comment=comment,
        tags=tags,
    )


def test_len_rabi_raw_labber_axes_tag_shape_and_disk_units(tmp_path: Path) -> None:
    result = _sample_result()

    LenRabiExp().save(str(tmp_path / "len_rabi"), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, "len_rabi")))

    assert [axis.name for axis in raw.axes] == ["GE Population", "Length"]
    assert [axis.unit for axis in raw.axes] == ["None", "s"]
    np.testing.assert_array_equal(raw.axes[0].values, result.population_states)
    np.testing.assert_allclose(raw.axes[1].values, result.lengths * 1e-6, rtol=0)
    assert raw.data.name == "Population"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (len(result.lengths), len(result.population_states))
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)
    assert raw.tags == ["singleshot/len_rabi"]


def test_len_rabi_save_load_roundtrip_with_cfg(tmp_path: Path) -> None:
    result = _sample_result()

    LenRabiExp().save(str(tmp_path / "roundtrip"), result=result, comment="note")
    load_exp = LenRabiExp()
    loaded = load_exp.load(str(_saved_path(tmp_path, "roundtrip")))

    np.testing.assert_allclose(loaded.lengths, result.lengths, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    assert loaded.lengths.dtype == np.float64
    assert loaded.population_states.dtype == np.int64
    assert loaded.signals.shape == result.signals.shape == (3, 2)
    assert loaded.signals.dtype == np.float64
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.length.expts == 3


def test_len_rabi_save_fast_fails_on_legacy_shape(tmp_path: Path) -> None:
    result = _sample_result()
    bad_result = LenRabiResult(
        lengths=result.lengths,
        signals=result.signals.T,
        population_states=result.population_states,
        cfg_snapshot=_cfg(),
    )

    with pytest.raises(ValueError):
        LenRabiExp().save(str(tmp_path / "bad_shape"), result=bad_result)


def test_len_rabi_save_fast_fails_without_cfg_snapshot(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)

    with pytest.raises(ValueError, match="cfg_snapshot"):
        LenRabiExp().save(str(tmp_path / "no_cfg"), result=result)


def test_len_rabi_runtime_load_rejects_legacy_axis_order(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_len_rabi.hdf5"
    _write_legacy_len_rabi(legacy_path, _sample_result(with_cfg=False))

    with pytest.raises(ValueError, match="axis 0 label"):
        LenRabiExp().load(str(legacy_path))


def test_len_rabi_result_default_population_states_are_isolated() -> None:
    first = _sample_result(with_cfg=False, omit_population_states=True)
    second = _sample_result(with_cfg=False, omit_population_states=True)

    expected = np.array([0, 1], dtype=np.int64)
    np.testing.assert_array_equal(first.population_states, expected)
    np.testing.assert_array_equal(second.population_states, expected)
    assert first.population_states.dtype == np.int64
    assert second.population_states.dtype == np.int64
    assert first.population_states is not second.population_states

    first.population_states[0] = -1
    assert second.population_states[0] == 0


def test_migrate_len_rabi_legacy_to_canonical_hdf5(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_len_rabi.hdf5"
    _write_legacy_len_rabi(legacy_path, result)
    legacy_bytes = legacy_path.read_bytes()

    migrated = migrate_experiment_data(
        experiment="singleshot/len_rabi",
        input_path=legacy_path,
        output_path=tmp_path / "canonical.hdf5",
    )

    assert migrated == tmp_path / "canonical.hdf5"
    assert legacy_path.read_bytes() == legacy_bytes
    loaded = LenRabiExp().load(str(migrated))
    np.testing.assert_allclose(loaded.lengths, result.lengths, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.cfg_snapshot is None
    raw = load_labber_data(str(migrated))
    assert [axis.name for axis in raw.axes] == ["GE Population", "Length"]
    assert [axis.unit for axis in raw.axes] == ["None", "s"]
    assert raw.z.shape == (len(result.lengths), len(result.population_states))
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)
    assert raw.comment == "legacy comment"
    assert raw.tags == ["singleshot/len_rabi"]


def test_migrate_len_rabi_requires_overwrite_and_allows_overwrite(
    tmp_path: Path,
) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_len_rabi.hdf5"
    _write_legacy_len_rabi(legacy_path, result)
    output = tmp_path / "canonical.hdf5"
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment="singleshot/len_rabi",
            input_path=legacy_path,
            output_path=output,
        )
    assert output.read_text() == "existing"

    migrated = migrate_experiment_data(
        experiment="singleshot/len_rabi",
        input_path=legacy_path,
        output_path=output,
        overwrite=True,
    )

    assert migrated == output
    loaded = LenRabiExp().load(str(output))
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("axes", "z"),
    [
        (
            [
                ("GE population", "a.u.", np.array([0, 1], dtype=np.int64)),
                ("Length", "s", np.array([0.1, 0.2, 0.3]) * 1e-6),
            ],
            ("Population", "a.u.", _sample_result(with_cfg=False).signals),
        ),
        (
            [
                ("Wrong Length", "s", np.array([0.1, 0.2, 0.3]) * 1e-6),
                ("GE population", "a.u.", np.array([0, 1], dtype=np.int64)),
            ],
            ("Population", "a.u.", _sample_result(with_cfg=False).signals.T),
        ),
        (
            [
                ("Length", "wrong", np.array([0.1, 0.2, 0.3]) * 1e-6),
                ("GE population", "a.u.", np.array([0, 1], dtype=np.int64)),
            ],
            ("Population", "a.u.", _sample_result(with_cfg=False).signals.T),
        ),
    ],
)
def test_migrate_len_rabi_rejects_wrong_axis_order_name_or_unit(
    tmp_path: Path,
    axes: list[tuple[str, str, np.ndarray]],
    z: tuple[str, str, np.ndarray],
) -> None:
    legacy_path = tmp_path / "legacy_bad_axis.hdf5"
    _write_legacy_len_rabi(
        legacy_path,
        _sample_result(with_cfg=False),
        axes=axes,
        z=z,
    )

    with pytest.raises(ValueError, match="axis 0"):
        migrate_experiment_data(
            experiment="singleshot/len_rabi",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_len_rabi_rejects_wrong_population_values(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_bad_population.hdf5"
    _write_legacy_len_rabi(
        legacy_path,
        result,
        axes=[
            ("Length", "s", result.lengths * 1e-6),
            ("GE population", "a.u.", np.array([0, 2], dtype=np.int64)),
        ],
    )

    with pytest.raises(ValueError, match="population state"):
        migrate_experiment_data(
            experiment="singleshot/len_rabi",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


@pytest.mark.parametrize(
    "z",
    [
        ("Wrong Population", "a.u.", _sample_result(with_cfg=False).signals.T),
        ("Population", "wrong", _sample_result(with_cfg=False).signals.T),
    ],
)
def test_migrate_len_rabi_rejects_wrong_z_channel(
    tmp_path: Path,
    z: tuple[str, str, np.ndarray],
) -> None:
    legacy_path = tmp_path / "legacy_bad_z.hdf5"
    _write_legacy_len_rabi(legacy_path, _sample_result(with_cfg=False), z=z)

    with pytest.raises(ValueError, match="z channel"):
        migrate_experiment_data(
            experiment="singleshot/len_rabi",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_len_rabi_rejects_wrong_z_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_bad_shape.hdf5"
    legacy_path.write_text("placeholder")
    payload = LabberData(
        data=("Population", "a.u.", np.ones((2, 4), dtype=np.float64)),
        axes=[
            ("Length", "s", result.lengths * 1e-6),
            ("GE population", "a.u.", result.population_states),
        ],
    )
    monkeypatch.setattr(
        "script.migrate_experiment_data.load_labber_data",
        lambda _path: payload,
    )

    with pytest.raises(ValueError, match="z shape"):
        migrate_experiment_data(
            experiment="singleshot/len_rabi",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


@pytest.mark.parametrize("imaginary", [0.25, 1e-12])
def test_migrate_len_rabi_rejects_non_real_z(
    tmp_path: Path,
    imaginary: float,
) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / f"legacy_complex_population_{imaginary}.hdf5"
    z = (
        "Population",
        "a.u.",
        result.signals.T.astype(np.complex128) + 1j * imaginary,
    )
    _write_legacy_len_rabi(legacy_path, result, z=z)

    with pytest.raises(ValueError, match="imaginary"):
        migrate_experiment_data(
            experiment="singleshot/len_rabi",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )
