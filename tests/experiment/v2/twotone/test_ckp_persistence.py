from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.v2.twotone.ckp import (
    CKP_Cfg,
    CKP_Exp,
    CKP_Result,
    CKPModuleCfg,
    CKPSweepCfg,
)
from zcu_tools.experiment.utils import make_comment
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import format_ext, load_labber_data, save_labber_data

import script.migrate_experiment_data as migrate_mod
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


def _cfg() -> CKP_Cfg:
    return CKP_Cfg(
        reps=1,
        rounds=1,
        modules=CKPModuleCfg(
            reset=None,
            pi_pulse=_pulse(ch=1, freq=3000.0, gain=0.4),
            res_pulse=_pulse(ch=0, freq=5500.0, gain=0.2),
            qub_pulse=_pulse(ch=1, freq=3000.0, gain=0.1),
            readout=DirectReadoutCfg(
                type="readout/direct",
                ro_ch=0,
                ro_length=1.0,
                ro_freq=6000.0,
            ),
        ),
        sweep=CKPSweepCfg(
            res_freq=SweepCfg(start=5480.0, stop=5520.0, step=40.0, expts=2),
            qub_freq=SweepCfg(start=3000.0, stop=3040.0, step=20.0, expts=3),
        ),
    )


def _sample_result(*, with_cfg: bool = True) -> CKP_Result:
    res_freqs = np.array([5480.0, 5520.0], dtype=np.float64)
    qub_freqs = np.array([3000.0, 3020.0, 3040.0], dtype=np.float64)
    values = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    signals = (values + 1j * (values + 0.5)).astype(np.complex128)
    return CKP_Result(
        res_freqs=res_freqs,
        qub_freqs=qub_freqs,
        signals=signals,
        cfg_snapshot=_cfg() if with_cfg else None,
    )


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}_1.hdf5"


def _legacy_sidecar_path(base_path: Path, suffix: str) -> Path:
    return Path(format_ext(str(base_path.with_name(base_path.name + suffix))))


def _write_legacy_ckp_sidecars(
    base_path: Path,
    result: CKP_Result,
    *,
    comment: str = "legacy comment",
    excited_comment: str | None = None,
    tags: str | list[str] = "twotone/ge/ckp",
    excited_tags: str | list[str] | None = None,
    write_excited: bool = True,
) -> None:
    axes = [
        ("Resonator Frequency", "Hz", result.res_freqs * 1e6),
        ("Qubit Frequency", "Hz", result.qub_freqs * 1e6),
    ]
    save_labber_data(
        str(base_path.with_name(base_path.name + "_ground")),
        z=("Signal", "a.u.", result.signals[0].T),
        axes=axes,
        comment=comment,
        tags=tags,
    )
    if write_excited:
        save_labber_data(
            str(base_path.with_name(base_path.name + "_excited")),
            z=("Signal", "a.u.", result.signals[1].T),
            axes=axes,
            comment=comment if excited_comment is None else excited_comment,
            tags=tags if excited_tags is None else excited_tags,
        )


def _replace_legacy_sidecar(
    sidecar_path: Path,
    *,
    z: tuple[str, str, np.ndarray],
    axes: list[tuple[str, str, np.ndarray]],
    comment: str = "legacy comment",
    tags: str | list[str] = "twotone/ge/ckp",
) -> None:
    sidecar_path.unlink()
    save_labber_data(
        str(sidecar_path),
        z=z,
        axes=axes,
        comment=comment,
        tags=tags,
    )


def test_ckp_save_load_roundtrip(tmp_path: Path) -> None:
    result = _sample_result()

    CKP_Exp().save(str(tmp_path / "ckp"), result=result, comment="note")
    path = _saved_path(tmp_path, "ckp")
    assert path.exists()

    load_exp = CKP_Exp()
    loaded = load_exp.load(str(path))

    np.testing.assert_allclose(loaded.res_freqs, result.res_freqs, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.qub_freqs, result.qub_freqs, rtol=0, atol=0)
    np.testing.assert_array_equal(loaded.initial_states, result.initial_states)
    assert loaded.initial_states.dtype == np.int64
    assert loaded.signals.shape == (2, len(result.res_freqs), len(result.qub_freqs))
    assert loaded.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.res_freq.expts == 2
    assert loaded.cfg_snapshot.sweep.qub_freq.expts == 3


def test_ckp_raw_labber_axes_tag_shape_and_no_sidecar(tmp_path: Path) -> None:
    result = _sample_result()

    CKP_Exp().save(str(tmp_path / "raw"), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, "raw")))

    assert [axis.name for axis in raw.axes] == [
        "Qubit Frequency",
        "Resonator Frequency",
        "Initial State",
    ]
    assert [axis.unit for axis in raw.axes] == ["Hz", "Hz", "None"]
    np.testing.assert_allclose(
        raw.axes[0].values, result.qub_freqs * 1e6, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        raw.axes[1].values, result.res_freqs * 1e6, rtol=0, atol=0
    )
    np.testing.assert_array_equal(raw.axes[2].values, result.initial_states)
    assert raw.data.name == "Signal"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (2, len(result.res_freqs), len(result.qub_freqs))
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)
    assert raw.tags == ["twotone/ge/ckp"]
    assert not any(tmp_path.glob("raw*_ground*"))
    assert not any(tmp_path.glob("raw*_excited*"))


def test_ckp_save_fast_fails_on_shape_mismatch(tmp_path: Path) -> None:
    result = _sample_result()
    bad_signals = np.ones(
        (len(result.initial_states), len(result.qub_freqs), len(result.res_freqs)),
        dtype=np.complex128,
    )
    bad_result = CKP_Result(
        res_freqs=result.res_freqs,
        qub_freqs=result.qub_freqs,
        signals=bad_signals,
        cfg_snapshot=_cfg(),
    )

    with pytest.raises(ValueError):
        CKP_Exp().save(str(tmp_path / "bad_shape"), result=bad_result)


def test_ckp_save_fast_fails_without_cfg_snapshot(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)

    with pytest.raises(ValueError, match="cfg_snapshot"):
        CKP_Exp().save(str(tmp_path / "no_cfg"), result=result)


def test_ckp_result_default_initial_states_are_isolated() -> None:
    first = _sample_result(with_cfg=False)
    second = _sample_result(with_cfg=False)

    expected = np.array([0, 1], dtype=np.int64)
    np.testing.assert_array_equal(first.initial_states, expected)
    np.testing.assert_array_equal(second.initial_states, expected)
    assert first.initial_states.dtype == np.int64
    assert second.initial_states.dtype == np.int64
    assert first.initial_states is not second.initial_states

    first.initial_states[0] = -1
    assert second.initial_states[0] == 0


def test_migrate_ckp_sidecars_to_single_hdf5(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy"
    _write_legacy_ckp_sidecars(legacy_base, result)
    legacy_ground = _legacy_sidecar_path(legacy_base, "_ground")
    legacy_excited = _legacy_sidecar_path(legacy_base, "_excited")
    ground_bytes = legacy_ground.read_bytes()
    excited_bytes = legacy_excited.read_bytes()

    migrated = migrate_experiment_data(
        experiment="twotone/ckp",
        input_path=legacy_base,
        output_path=tmp_path / "canonical.hdf5",
    )

    assert migrated == tmp_path / "canonical.hdf5"
    assert legacy_ground.read_bytes() == ground_bytes
    assert legacy_excited.read_bytes() == excited_bytes
    loaded = CKP_Exp().load(str(migrated))
    np.testing.assert_allclose(loaded.res_freqs, result.res_freqs, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.qub_freqs, result.qub_freqs, rtol=0, atol=0)
    np.testing.assert_array_equal(loaded.initial_states, result.initial_states)
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.cfg_snapshot is None

    raw = load_labber_data(str(migrated))
    assert raw.comment == "legacy comment"
    assert raw.tags == ["twotone/ge/ckp"]
    assert raw.z.shape == (2, len(result.res_freqs), len(result.qub_freqs))


def test_migrate_ckp_restores_cfg_snapshot_from_legacy_comment(
    tmp_path: Path,
) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_cfg"
    _write_legacy_ckp_sidecars(legacy_base, result, comment=make_comment(_cfg()))

    migrated = migrate_experiment_data(
        experiment="twotone/ckp",
        input_path=legacy_base,
        output_path=tmp_path / "canonical_cfg.hdf5",
    )

    loaded = CKP_Exp().load(str(migrated))
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.res_freq.expts == 2
    assert loaded.cfg_snapshot.sweep.qub_freq.expts == 3


def test_migrate_ckp_rejects_comment_metadata_mismatch(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_comment"
    _write_legacy_ckp_sidecars(
        legacy_base,
        result,
        comment="ground comment",
        excited_comment="excited comment",
    )

    with pytest.raises(ValueError, match="comment metadata"):
        migrate_experiment_data(
            experiment="twotone/ckp",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ckp_rejects_tag_metadata_mismatch(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_tags"
    _write_legacy_ckp_sidecars(
        legacy_base,
        result,
        tags="twotone/ge/ckp",
        excited_tags="other/tag",
    )

    with pytest.raises(ValueError, match="tags"):
        migrate_experiment_data(
            experiment="twotone/ckp",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ckp_rejects_axis_name_mismatch(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_axis_name"
    _write_legacy_ckp_sidecars(legacy_base, result)
    excited_path = _legacy_sidecar_path(legacy_base, "_excited")

    _replace_legacy_sidecar(
        excited_path,
        z=("Signal", "a.u.", result.signals[1].T),
        axes=[
            ("Wrong Resonator Frequency", "Hz", result.res_freqs * 1e6),
            ("Qubit Frequency", "Hz", result.qub_freqs * 1e6),
        ],
    )

    with pytest.raises(ValueError, match="axis 0"):
        migrate_experiment_data(
            experiment="twotone/ckp",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ckp_rejects_axis_value_mismatch(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_axis_values"
    _write_legacy_ckp_sidecars(legacy_base, result)
    excited_path = _legacy_sidecar_path(legacy_base, "_excited")

    _replace_legacy_sidecar(
        excited_path,
        z=("Signal", "a.u.", result.signals[1].T),
        axes=[
            ("Resonator Frequency", "Hz", result.res_freqs * 1e6 + 1.0),
            ("Qubit Frequency", "Hz", result.qub_freqs * 1e6),
        ],
    )

    with pytest.raises(ValueError, match="Resonator Frequency"):
        migrate_experiment_data(
            experiment="twotone/ckp",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ckp_rejects_z_channel_mismatch(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_z_channel"
    _write_legacy_ckp_sidecars(legacy_base, result)
    excited_path = _legacy_sidecar_path(legacy_base, "_excited")

    _replace_legacy_sidecar(
        excited_path,
        z=("Wrong Signal", "a.u.", result.signals[1].T),
        axes=[
            ("Resonator Frequency", "Hz", result.res_freqs * 1e6),
            ("Qubit Frequency", "Hz", result.qub_freqs * 1e6),
        ],
    )

    with pytest.raises(ValueError, match="z channel"):
        migrate_experiment_data(
            experiment="twotone/ckp",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ckp_requires_overwrite_and_allows_overwrite(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy"
    _write_legacy_ckp_sidecars(legacy_base, result)
    output = tmp_path / "canonical.hdf5"
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment="twotone/ckp",
            input_path=legacy_base,
            output_path=output,
        )
    assert output.read_text() == "existing"

    migrated = migrate_experiment_data(
        experiment="twotone/ckp",
        input_path=legacy_base,
        output_path=output,
        overwrite=True,
    )

    assert migrated == output
    loaded = CKP_Exp().load(str(output))
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)


def test_migrate_ckp_missing_sidecar_raises(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy"
    _write_legacy_ckp_sidecars(legacy_base, result, write_excited=False)

    with pytest.raises(FileNotFoundError, match="_excited"):
        migrate_experiment_data(
            experiment="twotone/ckp",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ckp_validation_failure_cleans_temp_and_keeps_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _sample_result(with_cfg=False)
    legacy_base = tmp_path / "legacy"
    _write_legacy_ckp_sidecars(legacy_base, result)
    output = tmp_path / "canonical.hdf5"
    output.write_text("existing")

    spec = migrate_mod.CONVERTERS["twotone/ckp"]

    def fail_validation(_path: str) -> object:
        raise ValueError("forced validation failure")

    monkeypatch.setitem(
        migrate_mod.CONVERTERS,
        "twotone/ckp",
        migrate_mod.ConverterSpec(
            convert=spec.convert,
            validate=fail_validation,
            validate_input=spec.validate_input,
        ),
    )

    with pytest.raises(ValueError, match="forced validation failure"):
        migrate_experiment_data(
            experiment="twotone/ckp",
            input_path=legacy_base,
            output_path=output,
            overwrite=True,
        )

    assert output.read_text() == "existing"
    assert list(tmp_path.glob(".canonical.hdf5.*.hdf5")) == []
