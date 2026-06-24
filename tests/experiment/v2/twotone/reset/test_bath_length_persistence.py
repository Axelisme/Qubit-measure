from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.experiment.v2.twotone.reset.bath.length import (
    LengthCfg,
    LengthExp,
    LengthModuleCfg,
    LengthResult,
    LengthSweepCfg,
)
from zcu_tools.program.v2 import BathResetCfg, DirectReadoutCfg, PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import load_labber_data, save_labber_data

from script.migrate_experiment_data import migrate_experiment_data


def _pulse(
    *,
    freq: float = 100.0,
    gain: float = 0.1,
    length: float = 1.0,
    phase: float = 0.0,
) -> PulseCfg:
    return PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=length),
        ch=0,
        nqz=1,
        freq=freq,
        gain=gain,
        phase=phase,
    )


def _cfg() -> LengthCfg:
    return LengthCfg(
        reps=1,
        rounds=1,
        modules=LengthModuleCfg(
            reset=None,
            init_pulse=None,
            tested_reset=BathResetCfg(
                cavity_tone_cfg=_pulse(freq=5500.0, gain=0.7, length=2.0),
                qubit_tone_cfg=_pulse(freq=3000.0, gain=0.1, length=1.0),
                pi2_cfg=_pulse(freq=3000.0, gain=0.2, phase=90.0),
            ),
            readout=DirectReadoutCfg(
                type="readout/direct",
                ro_ch=0,
                ro_length=1.0,
                ro_freq=6000.0,
            ),
        ),
        sweep=LengthSweepCfg(
            length=SweepCfg(start=0.1, stop=0.3, step=0.1, expts=3),
        ),
    )


def _sample_result(
    *,
    with_cfg: bool = True,
    omit_phases: bool = False,
) -> LengthResult:
    lengths = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    signals = (values + 1j * (values + 0.5)).astype(np.complex128)
    cfg_snapshot = _cfg() if with_cfg else None
    if omit_phases:
        return LengthResult(
            lengths=lengths,
            signals=signals,
            cfg_snapshot=cfg_snapshot,
        )
    return LengthResult(
        lengths=lengths,
        signals=signals,
        phases=np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64),
        cfg_snapshot=cfg_snapshot,
    )


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}_1.hdf5"


def _write_legacy_length(
    path: Path,
    result: LengthResult,
    *,
    axes: list[tuple[str, str, np.ndarray]] | None = None,
    z: tuple[str, str, np.ndarray] | None = None,
    comment: str = "legacy comment",
    tags: str = "twotone/reset/bath/length",
) -> None:
    save_labber_data(
        str(path),
        z=z or ("Signal", "a.u.", result.signals.T),
        axes=axes
        or [
            ("Length", "s", result.lengths * 1e-6),
            ("Pi2 Phase", "deg", result.phases),
        ],
        comment=comment,
        tags=tags,
    )


def test_bath_length_raw_labber_axes_tag_shape_and_disk_units(
    tmp_path: Path,
) -> None:
    result = _sample_result()

    LengthExp().save(str(tmp_path / "bath_length"), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, "bath_length")))

    assert [axis.name for axis in raw.axes] == ["Pi2 Phase", "Length"]
    assert [axis.unit for axis in raw.axes] == ["deg", "s"]
    np.testing.assert_allclose(raw.axes[0].values, result.phases, rtol=0, atol=0)
    np.testing.assert_allclose(raw.axes[1].values, result.lengths * 1e-6, rtol=0)
    assert raw.data.name == "Signal"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (len(result.lengths), len(result.phases))
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)
    assert raw.tags == ["twotone/reset/bath/length"]


def test_bath_length_save_load_roundtrip_with_cfg(tmp_path: Path) -> None:
    result = _sample_result()

    LengthExp().save(str(tmp_path / "roundtrip"), result=result, comment="note")
    load_exp = LengthExp()
    loaded = load_exp.load(str(_saved_path(tmp_path, "roundtrip")))

    np.testing.assert_allclose(loaded.lengths, result.lengths, rtol=0, atol=1e-12)
    np.testing.assert_allclose(loaded.phases, result.phases, rtol=0, atol=0)
    assert loaded.lengths.dtype == np.float64
    assert loaded.phases.dtype == np.float64
    assert loaded.signals.shape == result.signals.shape == (3, 4)
    assert loaded.signals.dtype == np.complex128
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.length.expts == 3


def test_bath_length_save_fast_fails_on_legacy_shape(tmp_path: Path) -> None:
    result = _sample_result()
    bad_result = LengthResult(
        lengths=result.lengths,
        phases=result.phases,
        signals=result.signals.T,
        cfg_snapshot=_cfg(),
    )

    with pytest.raises(ValueError):
        LengthExp().save(str(tmp_path / "bad_shape"), result=bad_result)


def test_bath_length_runtime_load_rejects_legacy_file(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_length.hdf5"
    _write_legacy_length(legacy_path, _sample_result(with_cfg=False))

    with pytest.raises(ValueError, match="axis 0 label"):
        LengthExp().load(str(legacy_path))


def test_bath_length_save_fast_fails_without_cfg_snapshot(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)

    with pytest.raises(ValueError, match="cfg_snapshot"):
        LengthExp().save(str(tmp_path / "no_cfg"), result=result)


def test_bath_length_result_default_phases_are_isolated() -> None:
    first = _sample_result(with_cfg=False, omit_phases=True)
    second = _sample_result(with_cfg=False, omit_phases=True)

    expected = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    np.testing.assert_array_equal(first.phases, expected)
    np.testing.assert_array_equal(second.phases, expected)
    assert first.phases.dtype == np.float64
    assert second.phases.dtype == np.float64
    assert first.phases is not second.phases

    first.phases[0] = -1.0
    assert second.phases[0] == 0.0


def test_migrate_bath_length_legacy_to_canonical_hdf5(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_length.hdf5"
    _write_legacy_length(legacy_path, result)
    legacy_bytes = legacy_path.read_bytes()

    migrated = migrate_experiment_data(
        experiment="twotone/reset/bath/length",
        input_path=legacy_path,
        output_path=tmp_path / "canonical.hdf5",
    )

    assert migrated == tmp_path / "canonical.hdf5"
    assert legacy_path.read_bytes() == legacy_bytes
    loaded = LengthExp().load(str(migrated))
    np.testing.assert_allclose(loaded.lengths, result.lengths, rtol=0, atol=1e-12)
    np.testing.assert_allclose(loaded.phases, result.phases, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.cfg_snapshot is None
    raw = load_labber_data(str(migrated))
    assert [axis.name for axis in raw.axes] == ["Pi2 Phase", "Length"]
    assert raw.z.shape == (len(result.lengths), len(result.phases))
    assert raw.comment == "legacy comment"
    assert raw.tags == ["twotone/reset/bath/length"]


def test_migrate_bath_length_requires_overwrite_and_allows_overwrite(
    tmp_path: Path,
) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_length.hdf5"
    _write_legacy_length(legacy_path, result)
    output = tmp_path / "canonical.hdf5"
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment="twotone/reset/bath/length",
            input_path=legacy_path,
            output_path=output,
        )
    assert output.read_text() == "existing"

    migrated = migrate_experiment_data(
        experiment="twotone/reset/bath/length",
        input_path=legacy_path,
        output_path=output,
        overwrite=True,
    )

    assert migrated == output
    loaded = LengthExp().load(str(output))
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)


def test_migrate_bath_length_rejects_wrong_phase_values(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_bad_phase.hdf5"
    _write_legacy_length(
        legacy_path,
        result,
        axes=[
            ("Length", "s", result.lengths * 1e-6),
            ("Pi2 Phase", "deg", np.array([0.0, 90.0, 180.0, 180.0])),
        ],
    )

    with pytest.raises(ValueError, match="phase"):
        migrate_experiment_data(
            experiment="twotone/reset/bath/length",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


@pytest.mark.parametrize(
    ("axes", "z"),
    [
        (
            [
                ("Pi2 Phase", "deg", np.array([0.0, 90.0, 180.0, 270.0])),
                ("Length", "s", np.array([0.1, 0.2, 0.3]) * 1e-6),
            ],
            ("Signal", "a.u.", _sample_result(with_cfg=False).signals),
        ),
        (
            [
                ("Wrong Length", "s", np.array([0.1, 0.2, 0.3]) * 1e-6),
                ("Pi2 Phase", "deg", np.array([0.0, 90.0, 180.0, 270.0])),
            ],
            ("Signal", "a.u.", _sample_result(with_cfg=False).signals.T),
        ),
        (
            [
                ("Length", "wrong", np.array([0.1, 0.2, 0.3]) * 1e-6),
                ("Pi2 Phase", "deg", np.array([0.0, 90.0, 180.0, 270.0])),
            ],
            ("Signal", "a.u.", _sample_result(with_cfg=False).signals.T),
        ),
    ],
)
def test_migrate_bath_length_rejects_wrong_axis_order_name_or_unit(
    tmp_path: Path,
    axes: list[tuple[str, str, np.ndarray]],
    z: tuple[str, str, np.ndarray],
) -> None:
    legacy_path = tmp_path / "legacy_bad_axis.hdf5"
    _write_legacy_length(
        legacy_path,
        _sample_result(with_cfg=False),
        axes=axes,
        z=z,
    )

    with pytest.raises(ValueError, match="axis 0"):
        migrate_experiment_data(
            experiment="twotone/reset/bath/length",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )


def test_migrate_bath_length_rejects_wrong_z_channel(tmp_path: Path) -> None:
    result = _sample_result(with_cfg=False)
    legacy_path = tmp_path / "legacy_bad_z.hdf5"
    _write_legacy_length(
        legacy_path,
        result,
        z=("Wrong Signal", "a.u.", result.signals.T),
    )

    with pytest.raises(ValueError, match="z channel"):
        migrate_experiment_data(
            experiment="twotone/reset/bath/length",
            input_path=legacy_path,
            output_path=tmp_path / "canonical.hdf5",
        )
