from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from zcu_tools.experiment.v2.singleshot.t1.t1 import (
    T1Cfg,
    T1Exp,
    T1ModuleCfg,
    T1Result,
    T1SweepCfg,
)
from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone import (
    T1WithToneCfg,
    T1WithToneExp,
    T1WithToneModuleCfg,
    T1WithToneResult,
    T1WithToneSweepCfg,
)
from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone_sweep import (
    T1WithToneSweepCfg as T1ToneSweepCfg,
)
from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone_sweep import (
    T1WithToneSweepExp,
    T1WithToneSweepModuleCfg,
    T1WithToneSweepResult,
    T1WithToneSweepSweepCfg,
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


@dataclass(frozen=True)
class T1Case:
    name: str
    exp_cls: type[T1Exp] | type[T1WithToneExp]
    result_cls: type[T1Result] | type[T1WithToneResult]
    tag: str
    converter_id: str


T1_CASES = (
    T1Case(
        name="t1",
        exp_cls=T1Exp,
        result_cls=T1Result,
        tag="singleshot/t1",
        converter_id="singleshot/t1/t1",
    ),
    T1Case(
        name="t1_with_tone",
        exp_cls=T1WithToneExp,
        result_cls=T1WithToneResult,
        tag="singleshot/t1/t1_with_tone",
        converter_id="singleshot/t1/t1_with_tone",
    ),
)


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


def _length_sweep() -> SweepCfg:
    return SweepCfg(start=0.1, stop=10.0, step=4.95, expts=3)


def _outer_sweep() -> SweepCfg:
    return SweepCfg(start=0.0, stop=1.0, step=0.5, expts=3)


def _cfg(case: T1Case) -> T1Cfg | T1WithToneCfg:
    if case.name == "t1":
        return T1Cfg(
            reps=1,
            rounds=1,
            modules=T1ModuleCfg(reset=None, pi_pulse=_pulse(ch=1), readout=_readout()),
            sweep=T1SweepCfg(length=_length_sweep()),
        )
    return T1WithToneCfg(
        reps=1,
        rounds=1,
        modules=T1WithToneModuleCfg(
            reset=None,
            init_pulse=None,
            pi_pulse=_pulse(ch=1),
            probe_pulse=_pulse(ch=2),
            readout=_readout(),
        ),
        sweep=T1WithToneSweepCfg(length=_length_sweep()),
    )


def _exp(case: T1Case) -> Any:
    return case.exp_cls()


def _result_cls(case: T1Case) -> Any:
    return case.result_cls


def _sweep_cfg() -> T1ToneSweepCfg:
    return T1ToneSweepCfg(
        reps=1,
        rounds=1,
        modules=T1WithToneSweepModuleCfg(
            reset=None,
            pi_pulse=_pulse(ch=1),
            probe_pulse=_pulse(ch=2),
            readout=_readout(),
        ),
        sweep=T1WithToneSweepSweepCfg(length=_length_sweep(), gain=_outer_sweep()),
    )


def _signals() -> np.ndarray:
    return np.arange(12, dtype=np.float64).reshape(3, 2, 2) / 10.0


def _sample_t1_result(
    case: T1Case,
    *,
    with_cfg: bool = True,
    omit_state_axes: bool = False,
) -> T1Result | T1WithToneResult:
    lengths = np.array([0.1, 1.0, 10.0], dtype=np.float64)
    cfg_snapshot = _cfg(case) if with_cfg else None
    kwargs: dict[str, Any] = dict(
        lengths=lengths,
        signals=_signals(),
        cfg_snapshot=cfg_snapshot,
    )
    if not omit_state_axes:
        kwargs["initial_states"] = np.array([0, 1], dtype=np.int64)
        kwargs["population_states"] = np.array([0, 1], dtype=np.int64)
    return _result_cls(case)(**kwargs)


def _sample_sweep_result(
    *,
    with_cfg: bool = True,
    omit_state_axes: bool = False,
    legacy_other_shape: bool = False,
) -> T1WithToneSweepResult:
    xs = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    lengths = np.array([0.1, 1.0], dtype=np.float64)
    last_dim = 3 if legacy_other_shape else 2
    signals = np.arange(
        len(xs) * 2 * len(lengths) * last_dim, dtype=np.float64
    ).reshape(len(xs), 2, len(lengths), last_dim)
    kwargs: dict[str, Any] = dict(
        xs=xs,
        lengths=lengths,
        signals=signals,
        cfg_snapshot=_sweep_cfg() if with_cfg else None,
    )
    if not omit_state_axes:
        kwargs["initial_states"] = np.array([0, 1], dtype=np.int64)
        kwargs["population_states"] = np.array([0, 1], dtype=np.int64)
    return T1WithToneSweepResult(**kwargs)


def _saved_path(tmp_path: Path, base: str) -> Path:
    return tmp_path / f"{base}.hdf5"


def _t1_sidecar_paths(base: Path) -> tuple[Path, Path]:
    return (
        Path(reserve_labber_filepath(str(base.with_name(base.stem + "_initg")))),
        Path(reserve_labber_filepath(str(base.with_name(base.stem + "_inite")))),
    )


def _sweep_sidecar_paths(base: Path) -> tuple[Path, Path, Path, Path]:
    return (
        Path(reserve_labber_filepath(str(base.with_name(base.name + "_gg_pop")))),
        Path(reserve_labber_filepath(str(base.with_name(base.name + "_ge_pop")))),
        Path(reserve_labber_filepath(str(base.with_name(base.name + "_eg_pop")))),
        Path(reserve_labber_filepath(str(base.with_name(base.name + "_ee_pop")))),
    )


def _write_t1_sidecars(
    base: Path,
    result: T1Result | T1WithToneResult,
    *,
    initg_axes: list[tuple[str, str, np.ndarray]] | None = None,
    inite_axes: list[tuple[str, str, np.ndarray]] | None = None,
    initg_z: tuple[str, str, np.ndarray] | None = None,
    inite_z: tuple[str, str, np.ndarray] | None = None,
    initg_comment: str = "legacy comment",
    inite_comment: str = "legacy comment",
    initg_tags: str = "singleshot/t1",
    inite_tags: str = "singleshot/t1",
) -> tuple[Path, Path]:
    initg_path, inite_path = _t1_sidecar_paths(base)
    axes = [
        ("Time", "s", result.lengths * 1e-6),
        ("GE population", "a.u.", np.array([0, 1], dtype=np.int64)),
    ]
    save_labber_data(
        str(initg_path),
        z=initg_z or ("Signal", "a.u.", result.signals[:, 0].T),
        axes=initg_axes or axes,
        comment=initg_comment,
        tags=initg_tags,
    )
    save_labber_data(
        str(inite_path),
        z=inite_z or ("Signal", "a.u.", result.signals[:, 1].T),
        axes=inite_axes or axes,
        comment=inite_comment,
        tags=inite_tags,
    )
    return initg_path, inite_path


def _write_sweep_sidecars(
    base: Path,
    result: T1WithToneSweepResult,
    *,
    axes: list[tuple[str, str, np.ndarray]] | None = None,
    z_name: str = "Ground Populations",
    z_values: dict[str, np.ndarray] | None = None,
    comments: tuple[str, str, str, str] = (
        "legacy comment",
        "legacy comment",
        "legacy comment",
        "legacy comment",
    ),
    tags: tuple[str, str, str, str] = (
        "singleshot/t1/t1_with_tone_sweep",
        "singleshot/t1/t1_with_tone_sweep",
        "singleshot/t1/t1_with_tone_sweep",
        "singleshot/t1/t1_with_tone_sweep",
    ),
) -> tuple[Path, Path, Path, Path]:
    paths = _sweep_sidecar_paths(base)
    default_axes = [
        ("Time", "s", result.lengths * 1e-6),
        ("sweep value", "a.u.", result.xs),
    ]
    values = z_values or {
        "gg": result.signals[:, 0, :, 0],
        "ge": result.signals[:, 0, :, 1],
        "eg": result.signals[:, 1, :, 0],
        "ee": result.signals[:, 1, :, 1],
    }
    for path, key, comment, tag in zip(
        paths, ("gg", "ge", "eg", "ee"), comments, tags, strict=True
    ):
        save_labber_data(
            str(path),
            z=(z_name, "a.u.", values[key]),
            axes=axes or default_axes,
            comment=comment,
            tags=tag,
        )
    return paths


@pytest.mark.parametrize("case", T1_CASES, ids=[case.name for case in T1_CASES])
def test_t1_raw_labber_axes_tag_shape_and_disk_units(
    tmp_path: Path, case: T1Case
) -> None:
    result = _sample_t1_result(case)

    _exp(case).save(str(tmp_path / case.name), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, case.name)))

    assert [axis.name for axis in raw.axes] == [
        "GE Population",
        "Initial State",
        "Time",
    ]
    assert [axis.unit for axis in raw.axes] == ["None", "None", "s"]
    np.testing.assert_array_equal(raw.axes[0].values, result.population_states)
    np.testing.assert_array_equal(raw.axes[1].values, result.initial_states)
    np.testing.assert_allclose(raw.axes[2].values, result.lengths * 1e-6, rtol=0)
    assert raw.data.name == "Population"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (3, 2, 2)
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)
    assert raw.tags == [case.tag]


@pytest.mark.parametrize("case", T1_CASES, ids=[case.name for case in T1_CASES])
def test_t1_save_load_roundtrip_with_cfg(tmp_path: Path, case: T1Case) -> None:
    result = _sample_t1_result(case)

    _exp(case).save(str(tmp_path / "roundtrip"), result=result, comment="note")
    load_exp = _exp(case)
    loaded = load_exp.load(str(_saved_path(tmp_path, "roundtrip")))

    np.testing.assert_allclose(loaded.lengths, result.lengths, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(loaded.initial_states, result.initial_states)
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    assert loaded.initial_states.dtype == np.int64
    assert loaded.population_states.dtype == np.int64
    assert loaded.signals.shape == result.signals.shape == (3, 2, 2)
    assert loaded.signals.dtype == np.float64
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None


@pytest.mark.parametrize("case", T1_CASES, ids=[case.name for case in T1_CASES])
def test_t1_save_fast_fails_on_legacy_shape_and_missing_cfg(
    tmp_path: Path, case: T1Case
) -> None:
    result = _sample_t1_result(case)
    bad_result = _result_cls(case)(
        lengths=result.lengths,
        signals=np.transpose(result.signals, (2, 0, 1)),
        initial_states=result.initial_states,
        population_states=result.population_states,
        cfg_snapshot=_cfg(case),
    )
    with pytest.raises(ValueError):
        _exp(case).save(str(tmp_path / f"bad_shape_{case.name}"), result=bad_result)

    with pytest.raises(ValueError, match="cfg_snapshot"):
        _exp(case).save(
            str(tmp_path / f"no_cfg_{case.name}"),
            result=_sample_t1_result(case, with_cfg=False),
        )


@pytest.mark.parametrize("case", T1_CASES, ids=[case.name for case in T1_CASES])
def test_t1_runtime_load_rejects_legacy_axis_order(
    tmp_path: Path, case: T1Case
) -> None:
    result = _sample_t1_result(case, with_cfg=False)
    legacy_base = tmp_path / f"legacy_{case.name}"
    initg_path, _ = _write_t1_sidecars(legacy_base, result)

    with pytest.raises(ValueError, match="canonical data has|axis 0 label"):
        _exp(case).load(str(initg_path))


@pytest.mark.parametrize("case", T1_CASES, ids=[case.name for case in T1_CASES])
def test_t1_default_state_axes_are_isolated(case: T1Case) -> None:
    first = _sample_t1_result(case, with_cfg=False, omit_state_axes=True)
    second = _sample_t1_result(case, with_cfg=False, omit_state_axes=True)

    expected = np.array([0, 1], dtype=np.int64)
    np.testing.assert_array_equal(first.initial_states, expected)
    np.testing.assert_array_equal(second.initial_states, expected)
    np.testing.assert_array_equal(first.population_states, expected)
    np.testing.assert_array_equal(second.population_states, expected)
    assert first.initial_states is not second.initial_states
    assert first.population_states is not second.population_states
    first.initial_states[0] = -1
    first.population_states[0] = -1
    assert second.initial_states[0] == 0
    assert second.population_states[0] == 0


@pytest.mark.parametrize("case", T1_CASES, ids=[case.name for case in T1_CASES])
def test_migrate_t1_legacy_sidecars_to_canonical_hdf5(
    tmp_path: Path, case: T1Case
) -> None:
    result = _sample_t1_result(case, with_cfg=False)
    legacy_base = tmp_path / f"legacy_{case.name}"
    initg_path, inite_path = _write_t1_sidecars(
        legacy_base,
        result,
        initg_tags=case.tag,
        inite_tags=case.tag,
    )
    initg_bytes = initg_path.read_bytes()
    inite_bytes = inite_path.read_bytes()

    migrated = migrate_experiment_data(
        experiment=case.converter_id,
        input_path=legacy_base,
        output_path=tmp_path / f"canonical_{case.name}.hdf5",
    )

    assert initg_path.read_bytes() == initg_bytes
    assert inite_path.read_bytes() == inite_bytes
    loaded = _exp(case).load(str(migrated))
    np.testing.assert_allclose(loaded.lengths, result.lengths, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(loaded.initial_states, result.initial_states)
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.cfg_snapshot is None
    raw = load_labber_data(str(migrated))
    assert raw.comment == "legacy comment"
    assert raw.tags == [case.tag]


def test_migrate_t1_requires_overwrite_and_allows_overwrite(tmp_path: Path) -> None:
    case = T1_CASES[0]
    result = _sample_t1_result(case, with_cfg=False)
    legacy_base = tmp_path / "legacy_t1"
    _write_t1_sidecars(legacy_base, result)
    output = tmp_path / "canonical.hdf5"
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_base,
            output_path=output,
        )
    assert output.read_text() == "existing"

    migrated = migrate_experiment_data(
        experiment=case.converter_id,
        input_path=legacy_base,
        output_path=output,
        overwrite=True,
    )

    assert migrated == output
    loaded = _exp(case).load(str(output))
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)


def test_migrate_t1_rejects_missing_sidecar_wrong_axes_and_metadata(
    tmp_path: Path,
) -> None:
    case = T1_CASES[0]
    result = _sample_t1_result(case, with_cfg=False)
    legacy_base = tmp_path / "legacy_t1_missing"
    initg_path, _ = _t1_sidecar_paths(legacy_base)
    save_labber_data(
        str(initg_path),
        z=("Signal", "a.u.", result.signals[:, 0].T),
        axes=[
            ("Time", "s", result.lengths * 1e-6),
            ("GE population", "a.u.", np.array([0, 1], dtype=np.int64)),
        ],
        comment="legacy comment",
        tags=case.tag,
    )
    with pytest.raises(FileNotFoundError, match="sidecar"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_base,
            output_path=tmp_path / "canonical_missing.hdf5",
        )

    legacy_base = tmp_path / "legacy_t1_bad_axis"
    _write_t1_sidecars(
        legacy_base,
        result,
        initg_axes=[
            ("Wrong Time", "s", result.lengths * 1e-6),
            ("GE population", "a.u.", np.array([0, 1], dtype=np.int64)),
        ],
    )
    with pytest.raises(ValueError, match="axis 0"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_base,
            output_path=tmp_path / "canonical_axis.hdf5",
        )

    legacy_base = tmp_path / "legacy_t1_mismatch"
    _write_t1_sidecars(
        legacy_base,
        result,
        inite_axes=[
            ("Time", "s", result.lengths * 1e-6 + 1e-9),
            ("GE population", "a.u.", np.array([0, 1], dtype=np.int64)),
        ],
    )
    with pytest.raises(ValueError, match="Time"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_base,
            output_path=tmp_path / "canonical_mismatch.hdf5",
        )

    legacy_base = tmp_path / "legacy_t1_meta"
    _write_t1_sidecars(legacy_base, result, inite_comment="different")
    with pytest.raises(ValueError, match="comment"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_base,
            output_path=tmp_path / "canonical_meta.hdf5",
        )


def test_migrate_t1_rejects_wrong_population_z_and_non_real(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = T1_CASES[0]
    result = _sample_t1_result(case, with_cfg=False)
    legacy_base = tmp_path / "legacy_t1_z"
    _write_t1_sidecars(
        legacy_base,
        result,
        initg_z=("Wrong Signal", "a.u.", result.signals[:, 0].T),
    )
    with pytest.raises(ValueError, match="z channel"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_base,
            output_path=tmp_path / "canonical_z.hdf5",
        )

    legacy_base = tmp_path / "legacy_t1_shape"
    for path in _t1_sidecar_paths(legacy_base):
        path.write_text("placeholder")
    payload = LabberData(
        data=("Signal", "a.u.", np.ones((2, 4), dtype=np.float64)),
        axes=[
            ("Time", "s", result.lengths * 1e-6),
            ("GE population", "a.u.", np.array([0, 1], dtype=np.int64)),
        ],
        comment="legacy comment",
        tags=[case.tag],
    )
    monkeypatch.setattr(
        "script.migrate_experiment_data.load_labber_data",
        lambda _path: payload,
    )
    with pytest.raises(ValueError, match="z shape"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_base,
            output_path=tmp_path / "canonical_shape.hdf5",
        )
    monkeypatch.undo()

    legacy_base = tmp_path / "legacy_t1_complex"
    _write_t1_sidecars(
        legacy_base,
        result,
        initg_z=(
            "Signal",
            "a.u.",
            result.signals[:, 0].T.astype(np.complex128) + 1e-12j,
        ),
    )
    with pytest.raises(ValueError, match="imaginary"):
        migrate_experiment_data(
            experiment=case.converter_id,
            input_path=legacy_base,
            output_path=tmp_path / "canonical_complex.hdf5",
        )


def test_t1_tone_sweep_raw_labber_axes_tag_shape_and_disk_units(tmp_path: Path) -> None:
    result = _sample_sweep_result()

    T1WithToneSweepExp().save(str(tmp_path / "sweep"), result=result)
    raw = load_labber_data(str(_saved_path(tmp_path, "sweep")))

    assert [axis.name for axis in raw.axes] == [
        "GE Population",
        "Time",
        "Initial State",
        "Sweep Value",
    ]
    assert [axis.unit for axis in raw.axes] == ["None", "s", "None", "a.u."]
    np.testing.assert_array_equal(raw.axes[0].values, result.population_states)
    np.testing.assert_allclose(raw.axes[1].values, result.lengths * 1e-6, rtol=0)
    np.testing.assert_array_equal(raw.axes[2].values, result.initial_states)
    np.testing.assert_allclose(raw.axes[3].values, result.xs, rtol=0)
    assert raw.data.name == "Population"
    assert raw.data.unit == "a.u."
    assert raw.z.shape == (3, 2, 2, 2)
    np.testing.assert_allclose(raw.z, result.signals, rtol=0, atol=0)
    assert raw.tags == ["singleshot/t1/t1_with_tone_sweep"]


def test_t1_tone_sweep_save_load_roundtrip_with_cfg(tmp_path: Path) -> None:
    result = _sample_sweep_result()

    T1WithToneSweepExp().save(
        str(tmp_path / "roundtrip"), result=result, comment="note"
    )
    load_exp = T1WithToneSweepExp()
    loaded = load_exp.load(str(_saved_path(tmp_path, "roundtrip")))

    np.testing.assert_allclose(loaded.xs, result.xs, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.lengths, result.lengths, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(loaded.initial_states, result.initial_states)
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    assert loaded.signals.shape == result.signals.shape == (3, 2, 2, 2)
    assert loaded.signals.dtype == np.float64
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert load_exp.last_result is loaded
    assert loaded.cfg_snapshot is not None


def test_t1_tone_sweep_fast_fails_on_legacy_other_shape_and_missing_cfg(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        T1WithToneSweepExp().save(
            str(tmp_path / "bad_shape"),
            result=_sample_sweep_result(legacy_other_shape=True),
        )

    with pytest.raises(ValueError, match="cfg_snapshot"):
        T1WithToneSweepExp().save(
            str(tmp_path / "no_cfg"),
            result=_sample_sweep_result(with_cfg=False),
        )


def test_t1_tone_sweep_default_state_axes_are_isolated() -> None:
    first = _sample_sweep_result(with_cfg=False, omit_state_axes=True)
    second = _sample_sweep_result(with_cfg=False, omit_state_axes=True)

    expected = np.array([0, 1], dtype=np.int64)
    np.testing.assert_array_equal(first.initial_states, expected)
    np.testing.assert_array_equal(second.initial_states, expected)
    np.testing.assert_array_equal(first.population_states, expected)
    np.testing.assert_array_equal(second.population_states, expected)
    assert first.initial_states is not second.initial_states
    assert first.population_states is not second.population_states
    first.initial_states[0] = -1
    first.population_states[0] = -1
    assert second.initial_states[0] == 0
    assert second.population_states[0] == 0


def test_t1_tone_sweep_runtime_load_rejects_legacy_axis_order(tmp_path: Path) -> None:
    result = _sample_sweep_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_sweep"
    gg_path, *_ = _write_sweep_sidecars(legacy_base, result)

    with pytest.raises(ValueError, match="canonical data has|axis 0 label"):
        T1WithToneSweepExp().load(str(gg_path))


def test_migrate_t1_tone_sweep_legacy_sidecars_to_canonical_hdf5(
    tmp_path: Path,
) -> None:
    result = _sample_sweep_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_sweep"
    paths = _write_sweep_sidecars(legacy_base, result)
    input_bytes = [path.read_bytes() for path in paths]

    migrated = migrate_experiment_data(
        experiment="singleshot/t1/t1_with_tone_sweep",
        input_path=legacy_base,
        output_path=tmp_path / "canonical_sweep.hdf5",
    )

    for path, original in zip(paths, input_bytes, strict=True):
        assert path.read_bytes() == original
    loaded = T1WithToneSweepExp().load(str(migrated))
    np.testing.assert_allclose(loaded.xs, result.xs, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.lengths, result.lengths, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(loaded.initial_states, result.initial_states)
    np.testing.assert_array_equal(loaded.population_states, result.population_states)
    assert loaded.signals.shape[-1] == 2
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)
    assert loaded.cfg_snapshot is None
    raw = load_labber_data(str(migrated))
    assert raw.comment == "legacy comment"
    assert raw.tags == ["singleshot/t1/t1_with_tone_sweep"]


def test_migrate_t1_tone_sweep_requires_overwrite_and_allows_overwrite(
    tmp_path: Path,
) -> None:
    result = _sample_sweep_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_sweep"
    _write_sweep_sidecars(legacy_base, result)
    output = tmp_path / "canonical.hdf5"
    output.write_text("existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment="singleshot/t1/t1_with_tone_sweep",
            input_path=legacy_base,
            output_path=output,
        )
    assert output.read_text() == "existing"

    migrated = migrate_experiment_data(
        experiment="singleshot/t1/t1_with_tone_sweep",
        input_path=legacy_base,
        output_path=output,
        overwrite=True,
    )

    assert migrated == output
    loaded = T1WithToneSweepExp().load(str(output))
    np.testing.assert_allclose(loaded.signals, result.signals, rtol=0, atol=0)


def test_migrate_t1_tone_sweep_rejects_missing_wrong_axes_metadata_and_z(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _sample_sweep_result(with_cfg=False)
    legacy_base = tmp_path / "legacy_sweep_missing"
    gg_path, *_ = _sweep_sidecar_paths(legacy_base)
    save_labber_data(
        str(gg_path),
        z=("Ground Populations", "a.u.", result.signals[:, 0, :, 0]),
        axes=[
            ("Time", "s", result.lengths * 1e-6),
            ("sweep value", "a.u.", result.xs),
        ],
        comment="legacy comment",
        tags="singleshot/t1/t1_with_tone_sweep",
    )
    with pytest.raises(FileNotFoundError, match="sidecar"):
        migrate_experiment_data(
            experiment="singleshot/t1/t1_with_tone_sweep",
            input_path=legacy_base,
            output_path=tmp_path / "canonical_missing.hdf5",
        )

    legacy_base = tmp_path / "legacy_sweep_axis"
    _write_sweep_sidecars(
        legacy_base,
        result,
        axes=[
            ("Wrong Time", "s", result.lengths * 1e-6),
            ("sweep value", "a.u.", result.xs),
        ],
    )
    with pytest.raises(ValueError, match="axis 0"):
        migrate_experiment_data(
            experiment="singleshot/t1/t1_with_tone_sweep",
            input_path=legacy_base,
            output_path=tmp_path / "canonical_axis.hdf5",
        )

    legacy_base = tmp_path / "legacy_sweep_meta"
    _write_sweep_sidecars(
        legacy_base,
        result,
        comments=("legacy comment", "different", "legacy comment", "legacy comment"),
    )
    with pytest.raises(ValueError, match="comment"):
        migrate_experiment_data(
            experiment="singleshot/t1/t1_with_tone_sweep",
            input_path=legacy_base,
            output_path=tmp_path / "canonical_meta.hdf5",
        )

    legacy_base = tmp_path / "legacy_sweep_z"
    _write_sweep_sidecars(legacy_base, result, z_name="Wrong Population")
    with pytest.raises(ValueError, match="z channel"):
        migrate_experiment_data(
            experiment="singleshot/t1/t1_with_tone_sweep",
            input_path=legacy_base,
            output_path=tmp_path / "canonical_z.hdf5",
        )

    legacy_base = tmp_path / "legacy_sweep_shape"
    for path in _sweep_sidecar_paths(legacy_base):
        path.write_text("placeholder")
    payload = LabberData(
        data=("Ground Populations", "a.u.", np.ones((4, 2), dtype=np.float64)),
        axes=[
            ("Time", "s", result.lengths * 1e-6),
            ("sweep value", "a.u.", result.xs),
        ],
        comment="legacy comment",
        tags=["singleshot/t1/t1_with_tone_sweep"],
    )
    monkeypatch.setattr(
        "script.migrate_experiment_data.load_labber_data",
        lambda _path: payload,
    )
    with pytest.raises(ValueError, match="z shape"):
        migrate_experiment_data(
            experiment="singleshot/t1/t1_with_tone_sweep",
            input_path=legacy_base,
            output_path=tmp_path / "canonical_shape.hdf5",
        )


@pytest.mark.parametrize("imaginary", [0.25, 1e-12])
def test_migrate_t1_tone_sweep_rejects_non_real_z(
    tmp_path: Path, imaginary: float
) -> None:
    result = _sample_sweep_result(with_cfg=False)
    legacy_base = tmp_path / f"legacy_sweep_complex_{imaginary}"
    _write_sweep_sidecars(
        legacy_base,
        result,
        z_values={
            "gg": result.signals[:, 0, :, 0].astype(np.complex128) + 1j * imaginary,
            "ge": result.signals[:, 0, :, 1],
            "eg": result.signals[:, 1, :, 0],
            "ee": result.signals[:, 1, :, 1],
        },
    )

    with pytest.raises(ValueError, match="imaginary"):
        migrate_experiment_data(
            experiment="singleshot/t1/t1_with_tone_sweep",
            input_path=legacy_base,
            output_path=tmp_path / "canonical.hdf5",
        )
