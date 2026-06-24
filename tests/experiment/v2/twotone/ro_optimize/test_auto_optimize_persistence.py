from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

import zcu_tools.experiment.v2.twotone.ro_optimize.auto_optimize as ro_mod
from script.migrate_experiment_data import migrate_experiment_data
from zcu_tools.experiment.v2.twotone.ro_optimize.auto_optimize import (
    RO_AUTO_GROUPED_ROLES,
    RO_AUTO_READOUT_FREQ_ROLE,
    RO_AUTO_READOUT_GAIN_ROLE,
    RO_AUTO_READOUT_LENGTH_ROLE,
    RO_AUTO_SNR_ROLE,
    AutoOptCfg,
    AutoOptExp,
    AutoOptResult,
    load_auto_opt_grouped_result,
)
from zcu_tools.utils.datasaver import (
    DatasetRole,
    LabberPayload,
    load_grouped_labber_data,
    save_grouped_labber_data,
    save_labber_data,
)


def _sample_result(*, with_cfg: bool = False) -> AutoOptResult:
    params = np.array(
        [
            [6123.1, 0.10, 0.80],
            [6124.2, 0.20, 1.10],
            [6125.3, 0.35, 1.30],
            [6126.4, 0.50, 1.60],
        ],
        dtype=np.float64,
    )
    signals = np.array([3.0, 5.5, 4.0, 7.25], dtype=np.float64)
    cfg = cast(AutoOptCfg, object()) if with_cfg else None
    return AutoOptResult(params=params, signals=signals, cfg_snapshot=cfg)


def _sidecar_base(base: Path, suffix: str, *, numbered: bool = False) -> Path:
    number = "_1" if numbered else ""
    return base.with_name(base.name + suffix + number)


def _write_legacy_ro_sidecars(
    base: Path,
    result: AutoOptResult,
    *,
    comment: str = "legacy comment",
    signal_comment: str | None = None,
    numbered: bool = False,
    params_axes: list[tuple[str, str, np.ndarray]] | None = None,
    params_z: tuple[str, str, np.ndarray] | None = None,
    signal_axes: list[tuple[str, str, np.ndarray]] | None = None,
    signal_z: tuple[str, str, np.ndarray] | None = None,
) -> tuple[Path, Path]:
    num_points = result.params.shape[0]
    iterations = np.arange(num_points, dtype=np.int64)
    params_path = Path(
        save_labber_data(
            str(_sidecar_base(base, "_params", numbered=numbered)),
            z=params_z
            or ("Parameters", "a.u.", np.asarray(result.params, dtype=np.float64).T),
            axes=params_axes
            or [
                ("Iteration", "a.u.", iterations),
                ("Parameter Type", "a.u.", np.array([0, 1, 2], dtype=np.int64)),
            ],
            comment=comment,
            tags="twotone/ge/ro_optimize/auto/params",
        )
    )
    signals_path = Path(
        save_labber_data(
            str(_sidecar_base(base, "_signals", numbered=numbered)),
            z=signal_z
            or ("Signal", "a.u.", np.asarray(result.signals, dtype=np.float64)),
            axes=signal_axes or [("Iteration", "a.u.", iterations)],
            comment=comment if signal_comment is None else signal_comment,
            tags="twotone/ge/ro_optimize/auto/signals",
        )
    )
    return params_path, signals_path


def test_ro_auto_save_writes_one_grouped_hdf5_and_loads_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ro_mod, "make_comment", lambda _cfg, _comment: "stored")

    result = _sample_result(with_cfg=True)
    AutoOptExp().save(str(tmp_path / "ro_auto"), result=result, comment="ignored")

    path = tmp_path / "ro_auto.hdf5"
    assert path.exists()
    assert list(tmp_path.glob("ro_auto*_params*.hdf5")) == []
    assert list(tmp_path.glob("ro_auto*_signals*.hdf5")) == []

    grouped = load_grouped_labber_data(str(path), required_roles=RO_AUTO_GROUPED_ROLES)
    assert grouped.metadata.comment == "stored"
    assert grouped.metadata.tags == ["twotone/ge/ro_optimize/auto"]
    assert list(grouped.roles) == [DatasetRole(role) for role in RO_AUTO_GROUPED_ROLES]

    expected_axis = np.arange(result.params.shape[0], dtype=np.int64)
    for role in RO_AUTO_GROUPED_ROLES:
        payload = grouped.roles[DatasetRole(role)]
        assert [(axis.name, axis.unit) for axis in payload.axes] == [
            ("Iteration", "a.u.")
        ]
        assert np.array_equal(payload.axes[0].values, expected_axis)

    assert np.allclose(
        grouped.roles[DatasetRole(RO_AUTO_READOUT_FREQ_ROLE)].z,
        result.params[:, 0] * 1e6,
    )
    assert np.allclose(
        grouped.roles[DatasetRole(RO_AUTO_READOUT_GAIN_ROLE)].z,
        result.params[:, 1],
    )
    assert np.allclose(
        grouped.roles[DatasetRole(RO_AUTO_READOUT_LENGTH_ROLE)].z,
        result.params[:, 2] * 1e-6,
    )
    assert np.allclose(
        grouped.roles[DatasetRole(RO_AUTO_SNR_ROLE)].z,
        result.signals,
    )

    loaded = AutoOptExp().load(str(path))
    assert np.allclose(loaded.params, result.params)
    assert np.allclose(loaded.signals, result.signals)


def test_ro_auto_save_fast_fails_without_cfg_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="configuration snapshot"):
        AutoOptExp().save(str(tmp_path / "ro_auto"), result=_sample_result())


@pytest.mark.parametrize(
    "result",
    [
        AutoOptResult(
            params=np.ones((4, 2), dtype=np.float64),
            signals=np.ones(4, dtype=np.float64),
            cfg_snapshot=cast(AutoOptCfg, object()),
        ),
        AutoOptResult(
            params=np.ones((4, 3), dtype=np.float64),
            signals=np.ones(3, dtype=np.float64),
            cfg_snapshot=cast(AutoOptCfg, object()),
        ),
    ],
)
def test_ro_auto_save_fast_fails_on_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: AutoOptResult
) -> None:
    monkeypatch.setattr(ro_mod, "make_comment", lambda _cfg, _comment: "stored")

    with pytest.raises(ValueError, match="RO auto-optimize"):
        AutoOptExp().save(str(tmp_path / "bad"), result=result)


def test_ro_auto_grouped_loader_rejects_incomplete_or_wrong_role_set(
    tmp_path: Path,
) -> None:
    axis = [("Iteration", "a.u.", np.arange(2, dtype=np.int64))]
    payload = LabberPayload(("SNR", "a.u.", np.ones(2)), axes=axis)

    save_grouped_labber_data(str(tmp_path / "missing"), {RO_AUTO_SNR_ROLE: payload})
    with pytest.raises(ValueError, match="missing required dataset role"):
        load_auto_opt_grouped_result(str(tmp_path / "missing.hdf5"))

    save_grouped_labber_data(
        str(tmp_path / "wrong"),
        {role: payload for role in (*RO_AUTO_GROUPED_ROLES, "unexpected_role")},
    )
    with pytest.raises(ValueError, match="unknown dataset role"):
        load_auto_opt_grouped_result(str(tmp_path / "wrong.hdf5"))


def test_ro_auto_runtime_load_rejects_legacy_sidecar_artifact(tmp_path: Path) -> None:
    base = tmp_path / "legacy"
    params_path, _signals_path = _write_legacy_ro_sidecars(base, _sample_result())

    with pytest.raises(FileNotFoundError):
        AutoOptExp().load(str(base))
    with pytest.raises(ValueError, match="not a grouped"):
        AutoOptExp().load(str(params_path))


def test_migrate_ro_auto_legacy_sidecars_to_grouped_hdf5_keeps_input_read_only(
    tmp_path: Path,
) -> None:
    base = tmp_path / "legacy"
    result = _sample_result()
    params_path, signals_path = _write_legacy_ro_sidecars(base, result)
    input_bytes = (params_path.read_bytes(), signals_path.read_bytes())

    migrated = migrate_experiment_data(
        experiment="twotone/ro_optimize/auto_optimize",
        input_path=base,
        output_path=tmp_path / "canonical",
    )

    assert params_path.read_bytes() == input_bytes[0]
    assert signals_path.read_bytes() == input_bytes[1]
    loaded = load_auto_opt_grouped_result(str(migrated))
    assert np.allclose(loaded.params, result.params)
    assert np.allclose(loaded.signals, result.signals)

    grouped = load_grouped_labber_data(
        str(migrated), required_roles=RO_AUTO_GROUPED_ROLES
    )
    assert grouped.metadata.comment == "legacy comment"
    assert grouped.metadata.tags == ["twotone/ge/ro_optimize/auto"]


def test_migrate_ro_auto_requires_overwrite_and_allows_overwrite(
    tmp_path: Path,
) -> None:
    base = tmp_path / "legacy"
    _write_legacy_ro_sidecars(base, _sample_result())
    output = tmp_path / "canonical.hdf5"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment="twotone/ro_optimize/auto_optimize",
            input_path=base,
            output_path=output,
        )
    assert output.read_bytes() == b"existing"

    migrated = migrate_experiment_data(
        experiment="twotone/ro_optimize/auto_optimize",
        input_path=base,
        output_path=output,
        overwrite=True,
    )
    assert migrated == output
    loaded = load_auto_opt_grouped_result(str(output))
    assert np.allclose(loaded.params, _sample_result().params)


def test_migrate_ro_auto_accepts_numbered_legacy_sidecars(tmp_path: Path) -> None:
    base = tmp_path / "legacy"
    result = _sample_result()
    _write_legacy_ro_sidecars(base, result, numbered=True)

    migrated = migrate_experiment_data(
        experiment="twotone/ro_optimize/auto_optimize",
        input_path=base,
        output_path=tmp_path / "canonical.hdf5",
    )

    loaded = load_auto_opt_grouped_result(str(migrated))
    assert np.allclose(loaded.params, result.params)
    assert np.allclose(loaded.signals, result.signals)


def test_migrate_ro_auto_rejects_missing_ambiguous_and_invalid_sidecars(
    tmp_path: Path,
) -> None:
    result = _sample_result()

    missing_base = tmp_path / "missing"
    save_labber_data(
        str(_sidecar_base(missing_base, "_params")),
        z=("Parameters", "a.u.", result.params.T),
        axes=[
            ("Iteration", "a.u.", np.arange(result.params.shape[0])),
            ("Parameter Type", "a.u.", np.array([0, 1, 2])),
        ],
    )
    with pytest.raises(FileNotFoundError, match="signals"):
        migrate_experiment_data(
            experiment="twotone/ro_optimize/auto_optimize",
            input_path=missing_base,
            output_path=tmp_path / "missing_out",
        )

    ambiguous_base = tmp_path / "ambiguous"
    _write_legacy_ro_sidecars(ambiguous_base, result)
    _write_legacy_ro_sidecars(ambiguous_base, result, numbered=True)
    with pytest.raises(ValueError, match="ambiguous legacy sidecar"):
        migrate_experiment_data(
            experiment="twotone/ro_optimize/auto_optimize",
            input_path=ambiguous_base,
            output_path=tmp_path / "ambiguous_out",
        )

    wrong_axis_base = tmp_path / "wrong_axis"
    _write_legacy_ro_sidecars(
        wrong_axis_base,
        result,
        params_axes=[
            ("Bad Iteration", "a.u.", np.arange(result.params.shape[0])),
            ("Parameter Type", "a.u.", np.array([0, 1, 2])),
        ],
    )
    with pytest.raises(ValueError, match="axis 0"):
        migrate_experiment_data(
            experiment="twotone/ro_optimize/auto_optimize",
            input_path=wrong_axis_base,
            output_path=tmp_path / "wrong_axis_out",
        )

    wrong_z_base = tmp_path / "wrong_z"
    _write_legacy_ro_sidecars(
        wrong_z_base,
        result,
        params_z=("Bad Parameters", "a.u.", result.params.T),
    )
    with pytest.raises(ValueError, match="z channel"):
        migrate_experiment_data(
            experiment="twotone/ro_optimize/auto_optimize",
            input_path=wrong_z_base,
            output_path=tmp_path / "wrong_z_out",
        )

    mismatched_axis_base = tmp_path / "mismatched_axis"
    _write_legacy_ro_sidecars(
        mismatched_axis_base,
        result,
        signal_axes=[("Iteration", "a.u.", np.arange(result.params.shape[0]) + 1)],
    )
    with pytest.raises(ValueError, match="Iteration axis"):
        migrate_experiment_data(
            experiment="twotone/ro_optimize/auto_optimize",
            input_path=mismatched_axis_base,
            output_path=tmp_path / "mismatched_axis_out",
        )

    metadata_base = tmp_path / "metadata"
    _write_legacy_ro_sidecars(metadata_base, result, signal_comment="different")
    with pytest.raises(ValueError, match="comment metadata"):
        migrate_experiment_data(
            experiment="twotone/ro_optimize/auto_optimize",
            input_path=metadata_base,
            output_path=tmp_path / "metadata_out",
        )

    complex_base = tmp_path / "complex"
    _write_legacy_ro_sidecars(
        complex_base,
        result,
        params_z=("Parameters", "a.u.", result.params.T + 1j),
    )
    with pytest.raises(ValueError, match="imaginary"):
        migrate_experiment_data(
            experiment="twotone/ro_optimize/auto_optimize",
            input_path=complex_base,
            output_path=tmp_path / "complex_out",
        )
