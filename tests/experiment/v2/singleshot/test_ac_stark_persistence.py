from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.v2.singleshot.ac_stark import (
    AcStarkCfg,
    AcStarkExp,
    AcStarkModuleCfg,
    AcStarkResult,
    AcStarkSweepCfg,
)
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import (
    LabberData,
    load_labber_data,
    reserve_labber_filepath,
    save_labber_data,
)

from script.migrate_experiment_data import migrate_experiment_data


def _pulse(*, ch: int = 0, freq: float = 100.0, gain: float = 0.1) -> PulseCfg:
    return PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=1.0),
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


def _cfg() -> AcStarkCfg:
    return AcStarkCfg(
        reps=1,
        rounds=1,
        modules=AcStarkModuleCfg(
            reset=None,
            init_pulse=None,
            stark_pulse1=_pulse(ch=1, freq=3000.0, gain=0.1),
            stark_pulse2=_pulse(ch=2, freq=4100.0, gain=0.2),
            readout=_readout(),
        ),
        sweep=AcStarkSweepCfg(
            gain=SweepCfg(start=0.1, stop=0.3, step=0.1, expts=3),
            freq=SweepCfg(start=4100.0, stop=4200.0, step=100.0, expts=2),
        ),
    )


def _sample_result(
    *,
    with_cfg: bool = True,
    omit_population_states: bool = False,
) -> AcStarkResult:
    gains = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    freqs = np.array([4100.0, 4200.0], dtype=np.float64)
    populations = np.arange(12, dtype=np.float64).reshape(3, 2, 2) / 10.0
    cfg_snapshot = _cfg() if with_cfg else None
    if omit_population_states:
        return AcStarkResult(
            gains=gains,
            freqs=freqs,
            populations=populations,
            cfg_snapshot=cfg_snapshot,
        )
    return AcStarkResult(
        gains=gains,
        freqs=freqs,
        populations=populations,
        population_states=np.array([0, 1], dtype=np.int64),
        cfg_snapshot=cfg_snapshot,
    )


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}.hdf5"


def _sidecar_paths(base: Path) -> tuple[Path, Path]:
    return (
        Path(reserve_labber_filepath(str(base.with_name(base.name + "_g_pop")))),
        Path(reserve_labber_filepath(str(base.with_name(base.name + "_e_pop")))),
    )


def _write_legacy_sidecars(
    base: Path,
    result: AcStarkResult,
    *,
    ground_axes: list[tuple[str, str, np.ndarray]] | None = None,
    excited_axes: list[tuple[str, str, np.ndarray]] | None = None,
    ground_z: tuple[str, str, np.ndarray] | None = None,
    excited_z: tuple[str, str, np.ndarray] | None = None,
    ground_comment: str = "legacy comment",
    excited_comment: str = "legacy comment",
    ground_tags: str = "singleshot/ac_stark",
    excited_tags: str = "singleshot/ac_stark",
) -> tuple[Path, Path]:
    ground_path, excited_path = _sidecar_paths(base)
    axes = [
        ("Stark Pulse Gain", "a.u.", result.gains),
        ("Frequency", "Hz", result.freqs * 1e6),
    ]
    save_labber_data(
        str(ground_path),
        z=ground_z or ("Signal", "a.u.", result.populations[..., 0].T),
        axes=ground_axes or axes,
        comment=ground_comment,
        tags=ground_tags,
    )
    save_labber_data(
        str(excited_path),
        z=excited_z or ("Signal", "a.u.", result.populations[..., 1].T),
        axes=excited_axes or axes,
        comment=excited_comment,
        tags=excited_tags,
    )
    return ground_path, excited_path


def test_ac_stark_raw_labber_axes_tag_shape_and_disk_units(tmp_path: Path) -> None:
    result = _sample_result()

    AcStarkExp().save(str(tmp_path / "ac_stark"), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, "ac_stark")))

    assert [axis.name for axis in raw.axes] == [
        "GE Population",
        "Frequency",
        "Stark Pulse Gain",
    ]
    assert [axis.unit for axis in raw.axes] == ["None", "Hz", "a.u."]
    np.testing.assert_array_equal(raw.axes[0].values, result.population_states)
    np.testing.assert_allclose(raw.axes[1].values, result.freqs * 1e6, rtol=0)
    np.testing.assert_allclose(raw.axes[2].values, result.gains, rtol=0)
    assert raw.data.name == "Population"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (3, 2, 2)
    np.testing.assert_allclose(raw.z, result.populations, rtol=0, atol=0)
    assert raw.tags == ["singleshot/ac_stark"]


def test_ac_stark_save_load_roundtrip_with_cfg(tmp_path: Path) -> None:
    result = _sample_result()

    AcStarkExp().save(str(tmp_path / "roundtrip"), result=result, comment="note")
    load_exp = AcStarkExp()
    loaded = load_exp.load(str(_saved_path(tmp_path, "roundtrip")))

    np.testing.assert_allclose(loaded.gains, result.gains, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.freqs, result.freqs, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    assert loaded.population_states.dtype == np.int64
    assert loaded.populations.shape == result.populations.shape == (3, 2, 2)
    assert loaded.populations.dtype == np.float64
    np.testing.assert_allclose(loaded.populations, result.populations, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.gain.expts == 3


def test_ac_stark_save_fast_fails_on_legacy_shape(tmp_path: Path) -> None:
    result = _sample_result()
    bad_result = AcStarkResult(
        gains=result.gains,
        freqs=result.freqs,
        populations=np.transpose(result.populations, (1, 0, 2)),
        population_states=result.population_states,
        cfg_snapshot=_cfg(),
    )

    with pytest.raises(ValueError):
        AcStarkExp().save(str(tmp_path / "bad_shape"), result=bad_result)


def test_ac_stark_save_fast_fails_without_cfg_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cfg_snapshot"):
        AcStarkExp().save(
            str(tmp_path / "no_cfg"), result=_sample_result(with_cfg=False)
        )


def test_ac_stark_runtime_load_rejects_legacy_axis_order(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_ac"
    ground_path, _ = _write_legacy_sidecars(legacy_base, result)

    with pytest.raises(ValueError, match="canonical data has|axis 0 label"):
        AcStarkExp().load(str(ground_path))


def test_ac_stark_default_population_states_are_isolated() -> None:
    first = _sample_result(with_cfg=False, omit_population_states=True)
    second = _sample_result(with_cfg=False, omit_population_states=True)

    expected = np.array([0, 1], dtype=np.int64)
    np.testing.assert_array_equal(first.population_states, expected)
    np.testing.assert_array_equal(second.population_states, expected)
    assert first.population_states is not second.population_states
    first.population_states[0] = -1
    assert second.population_states[0] == 0


def test_migrate_ac_stark_legacy_sidecars_to_canonical_hdf5(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_ac"
    ground_path, excited_path = _write_legacy_sidecars(legacy_base, result)
    ground_bytes = ground_path.read_bytes()
    excited_bytes = excited_path.read_bytes()

    migrated = migrate_experiment_data(
        experiment="singleshot/ac_stark",
        input_path=legacy_base,
        output_path=tmp_path / "canonical.hdf5",
    )

    assert migrated == tmp_path / "canonical.hdf5"
    assert ground_path.read_bytes() == ground_bytes
    assert excited_path.read_bytes() == excited_bytes
    loaded = AcStarkExp().load(str(migrated))
    np.testing.assert_allclose(loaded.gains, result.gains, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.freqs, result.freqs, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    np.testing.assert_allclose(loaded.populations, result.populations, rtol=0, atol=0)
    assert loaded.cfg_snapshot is None
    raw = load_labber_data(str(migrated))
    assert raw.comment == "legacy comment"
    assert raw.tags == ["singleshot/ac_stark"]


def test_migrate_ac_stark_requires_overwrite_and_allows_overwrite(
    tmp_path: Path,
) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_ac"
    _write_legacy_sidecars(legacy_base, result)
    output = tmp_path / "canonical.hdf5"
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment="singleshot/ac_stark",
            input_path=legacy_base,
            output_path=output,
        )
    assert output.read_text() == "existing"

    migrated = migrate_experiment_data(
        experiment="singleshot/ac_stark",
        input_path=legacy_base,
        output_path=output,
        overwrite=True,
    )

    assert migrated == output
    loaded = AcStarkExp().load(str(output))
    np.testing.assert_allclose(loaded.populations, result.populations, rtol=0, atol=0)


def test_migrate_ac_stark_rejects_missing_sidecar(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_ac"
    ground_path, _ = _sidecar_paths(legacy_base)
    save_labber_data(
        str(ground_path),
        z=("Signal", "a.u.", result.populations[..., 0].T),
        axes=[
            ("Stark Pulse Gain", "a.u.", result.gains),
            ("Frequency", "Hz", result.freqs * 1e6),
        ],
        comment="legacy comment",
        tags="singleshot/ac_stark",
    )

    with pytest.raises(FileNotFoundError, match="sidecar"):
        migrate_experiment_data(
            experiment="singleshot/ac_stark",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ac_stark_rejects_wrong_axis(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_ac"
    _write_legacy_sidecars(
        legacy_base,
        result,
        ground_axes=[
            ("Wrong Gain", "a.u.", result.gains),
            ("Frequency", "Hz", result.freqs * 1e6),
        ],
    )

    with pytest.raises(ValueError, match="axis 0"):
        migrate_experiment_data(
            experiment="singleshot/ac_stark",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ac_stark_rejects_mismatched_axes_and_metadata(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_ac"
    _write_legacy_sidecars(
        legacy_base,
        result,
        excited_axes=[
            ("Stark Pulse Gain", "a.u.", result.gains + 1.0),
            ("Frequency", "Hz", result.freqs * 1e6),
        ],
    )

    with pytest.raises(ValueError, match="Stark Pulse Gain"):
        migrate_experiment_data(
            experiment="singleshot/ac_stark",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )

    legacy_base = tmp_path / "legacy_ac_meta"
    _write_legacy_sidecars(legacy_base, result, excited_comment="different")
    with pytest.raises(ValueError, match="comment"):
        migrate_experiment_data(
            experiment="singleshot/ac_stark",
            input_path=legacy_base,
            output_path=tmp_path / "canonical_meta.hdf5",
        )


def test_migrate_ac_stark_rejects_wrong_z_channel_and_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_ac"
    _write_legacy_sidecars(
        legacy_base,
        result,
        ground_z=("Wrong Signal", "a.u.", result.populations[..., 0].T),
    )
    with pytest.raises(ValueError, match="z channel"):
        migrate_experiment_data(
            experiment="singleshot/ac_stark",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )

    legacy_base = tmp_path / "legacy_ac_shape"
    for path in _sidecar_paths(legacy_base):
        path.write_text("placeholder")
    payload = LabberData(
        data=("Signal", "a.u.", np.ones((2, 4), dtype=np.float64)),
        axes=[
            ("Stark Pulse Gain", "a.u.", result.gains),
            ("Frequency", "Hz", result.freqs * 1e6),
        ],
        comment="legacy comment",
        tags=["singleshot/ac_stark"],
    )
    monkeypatch.setattr(
        "script.migrate_experiment_data.load_labber_data",
        lambda _path: payload,
    )
    with pytest.raises(ValueError, match="z shape"):
        migrate_experiment_data(
            experiment="singleshot/ac_stark",
            input_path=legacy_base,
            output_path=tmp_path / "canonical_shape.hdf5",
        )


@pytest.mark.parametrize("imaginary", [0.25, 1e-12])
def test_migrate_ac_stark_rejects_non_real_z(tmp_path: Path, imaginary: float) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / f"legacy_ac_complex_{imaginary}"
    _write_legacy_sidecars(
        legacy_base,
        result,
        ground_z=(
            "Signal",
            "a.u.",
            result.populations[..., 0].T.astype(np.complex128) + 1j * imaginary,
        ),
    )

    with pytest.raises(ValueError, match="imaginary"):
        migrate_experiment_data(
            experiment="singleshot/ac_stark",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )
