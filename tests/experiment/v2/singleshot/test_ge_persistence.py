from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.v2.singleshot.ge import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEModuleCfg,
)
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import load_labber_data, save_labber_data

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


def _readout() -> PulseReadoutCfg:
    return PulseReadoutCfg(
        pulse_cfg=_pulse(ch=0, freq=6000.0, gain=0.2),
        ro_cfg=DirectReadoutCfg(
            type="readout/direct",
            ro_ch=0,
            ro_length=1.0,
            ro_freq=6000.0,
        ),
    )


def _cfg() -> GE_Cfg:
    return GE_Cfg(
        reps=1,
        rounds=1,
        modules=GEModuleCfg(
            reset=None,
            init_pulse=None,
            probe_pulse=_pulse(ch=1, freq=3000.0, gain=0.3),
            readout=_readout(),
        ),
        shots=4,
    )


def _sample_result(*, with_cfg: bool = True) -> GE_Result:
    signals = (
        np.arange(8, dtype=np.float64).reshape(2, 4)
        + 1j * np.arange(8, dtype=np.float64).reshape(2, 4)
    ).astype(np.complex128)
    return GE_Result(
        signals=signals,
        shot_indices=np.arange(signals.shape[1], dtype=np.int64),
        prepared_states=np.array([0, 1], dtype=np.int64),
        cfg_snapshot=_cfg() if with_cfg else None,
    )


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}.hdf5"


def _write_legacy_ge(
    path: Path,
    result: GE_Result,
    *,
    axes: list[tuple[str, str, np.ndarray]] | None = None,
    z: tuple[str, str, np.ndarray] | None = None,
    comment: str = "legacy comment",
    tags: str = "singleshot/ge",
) -> None:
    save_labber_data(
        str(path),
        z=z or ("Signal", "a.u.", result.signals),
        axes=axes
        or [
            ("shot", "point", result.shot_indices),
            ("ge", "", result.prepared_states),
        ],
        comment=comment,
        tags=tags,
    )


def test_ge_raw_labber_axes_tag_and_shape(tmp_path: Path) -> None:
    result = _sample_result()

    GE_Exp().save(str(tmp_path / "ge"), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, "ge")))

    assert [axis.name for axis in raw.axes] == ["Shot Index", "Prepared State"]
    assert [axis.unit for axis in raw.axes] == ["None", "None"]
    np.testing.assert_array_equal(raw.axes[0].values, result.shot_indices)
    np.testing.assert_array_equal(raw.axes[1].values, result.prepared_states)
    assert raw.data.name == "Signal"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (2, len(result.shot_indices))
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)
    assert raw.tags == ["singleshot/ge"]


def test_ge_save_load_roundtrip_with_cfg(tmp_path: Path) -> None:
    result = _sample_result()

    GE_Exp().save(str(tmp_path / "roundtrip"), result=result, comment="note")
    load_exp = GE_Exp()
    loaded = load_exp.load(str(_saved_path(tmp_path, "roundtrip")))

    np.testing.assert_array_equal(loaded.shot_indices, result.shot_indices)
    np.testing.assert_array_equal(loaded.prepared_states, result.prepared_states)
    assert loaded.shot_indices.dtype == np.int64
    assert loaded.prepared_states.dtype == np.int64
    assert loaded.signals.shape == result.signals.shape == (2, 4)
    assert loaded.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.shots == 4


def test_ge_save_fast_fails_on_shape_mismatch(tmp_path: Path) -> None:
    result = _sample_result()
    bad_result = GE_Result(
        signals=result.signals,
        shot_indices=np.arange(3, dtype=np.int64),
        prepared_states=result.prepared_states,
        cfg_snapshot=_cfg(),
    )

    with pytest.raises(ValueError):
        GE_Exp().save(str(tmp_path / "bad_shape"), result=bad_result)


def test_ge_save_fast_fails_without_cfg_snapshot(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)

    with pytest.raises(ValueError, match="cfg_snapshot"):
        GE_Exp().save(str(tmp_path / "no_cfg"), result=result)


def test_ge_runtime_load_rejects_legacy_axis_labels(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_ge.hdf5"
    _write_legacy_ge(legacy_path, _sample_result(with_cfg=False))

    with pytest.raises(ValueError, match="axis 0 label"):
        GE_Exp().load(str(legacy_path))


def test_migrate_ge_legacy_to_canonical_hdf5(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_ge.hdf5"
    _write_legacy_ge(legacy_path, result)
    legacy_bytes = legacy_path.read_bytes()

    migrated = migrate_experiment_data(
        experiment="singleshot/ge",
        input_path=legacy_path,
        output_path=tmp_path / "canonical.hdf5",
    )

    assert migrated == tmp_path / "canonical.hdf5"
    assert legacy_path.read_bytes() == legacy_bytes
    loaded = GE_Exp().load(str(migrated))
    np.testing.assert_array_equal(loaded.shot_indices, result.shot_indices)
    np.testing.assert_array_equal(loaded.prepared_states, result.prepared_states)
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.cfg_snapshot is None
    raw = load_labber_data(str(migrated))
    assert [axis.name for axis in raw.axes] == ["Shot Index", "Prepared State"]
    assert raw.comment == "legacy comment"
    assert raw.tags == ["singleshot/ge"]


def test_migrate_ge_requires_overwrite_and_allows_overwrite(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_ge.hdf5"
    _write_legacy_ge(legacy_path, result)
    output = tmp_path / "canonical.hdf5"
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment="singleshot/ge",
            input_path=legacy_path,
            output_path=output,
        )
    assert output.read_text() == "existing"

    migrated = migrate_experiment_data(
        experiment="singleshot/ge",
        input_path=legacy_path,
        output_path=output,
        overwrite=True,
    )

    assert migrated == output
    loaded = GE_Exp().load(str(output))
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)


@pytest.mark.parametrize(
    "axes",
    [
        [
            ("Wrong Shot", "point", np.arange(4, dtype=np.int64)),
            ("ge", "", np.array([0, 1], dtype=np.int64)),
        ],
        [
            ("shot", "wrong", np.arange(4, dtype=np.int64)),
            ("ge", "", np.array([0, 1], dtype=np.int64)),
        ],
    ],
)
def test_migrate_ge_rejects_wrong_axis_name_or_unit(
    tmp_path: Path, axes: list[tuple[str, str, np.ndarray]]
) -> None:
    legacy_path = tmp_path / "legacy_bad_axis.hdf5"
    _write_legacy_ge(legacy_path, _sample_result(with_cfg=False), axes=axes)

    with pytest.raises(ValueError, match="axis 0"):
        migrate_experiment_data(
            experiment="singleshot/ge",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ge_rejects_wrong_prepared_state_values(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_bad_state.hdf5"
    _write_legacy_ge(
        legacy_path,
        result,
        axes=[
            ("shot", "point", result.shot_indices),
            ("ge", "", np.array([0, 2], dtype=np.int64)),
        ],
    )

    with pytest.raises(ValueError, match="prepared state"):
        migrate_experiment_data(
            experiment="singleshot/ge",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_ge_rejects_wrong_z_channel(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_bad_z.hdf5"
    _write_legacy_ge(legacy_path, result, z=("Wrong Signal", "a.u.", result.signals))

    with pytest.raises(ValueError, match="z channel"):
        migrate_experiment_data(
            experiment="singleshot/ge",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )
