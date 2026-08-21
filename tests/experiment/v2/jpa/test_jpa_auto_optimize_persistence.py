from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import zcu_tools.experiment.v2.jpa.jpa_auto_optimize as jpa_mod
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.utils import make_comment
from zcu_tools.experiment.v2.jpa.jpa_auto_optimize import (
    JPA_AUTO_FLUX_ROLE,
    JPA_AUTO_FREQ_ROLE,
    JPA_AUTO_GROUPED_ROLES,
    JPA_AUTO_PHASE_ROLE,
    JPA_AUTO_POWER_ROLE,
    JPA_AUTO_SNR_ROLE,
    AutoOptimizeExp,
    JPAOptCfg,
    JPAOptimizeResult,
    JPAOptModuleCfg,
    JPAOptSweepCfg,
    load_jpa_auto_grouped_result,
)
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import ConstWaveformCfg
from zcu_tools.utils.datasaver import (
    DatasetRole,
    LabberMetadata,
    LabberPayload,
    load_grouped_labber_data,
    save_grouped_labber_data,
    save_labber_data,
)

from script.migrate_experiment_data import migrate_experiment_data


def _sample_result(*, with_cfg: bool = False) -> JPAOptimizeResult:
    params = np.array(
        [
            [1.1e-3, 7234.1, -20.0],
            [1.2e-3, 7235.2, -18.0],
            [1.3e-3, 7236.3, -16.0],
            [1.4e-3, 7237.4, -14.0],
        ],
        dtype=np.float64,
    )
    phases = np.array([0, 1, 2, 3], dtype=np.int32)
    signals = np.array([2.0, 4.5, 6.0, 5.25], dtype=np.float64)
    cfg = cast(JPAOptCfg, object()) if with_cfg else None
    return JPAOptimizeResult(
        params=params,
        phases=phases,
        signals=signals,
        cfg_snapshot=cfg,
    )


def _sidecar_base(base: Path, suffix: str, *, numbered: bool = False) -> Path:
    number = "_1" if numbered else ""
    return base.with_name(base.name + suffix + number)


def _write_legacy_jpa_sidecars(
    base: Path,
    result: JPAOptimizeResult,
    *,
    comment: str = "legacy comment",
    phase_comment: str | None = None,
    signal_comment: str | None = None,
    numbered: bool = False,
    params_axes: list[tuple[str, str, np.ndarray]] | None = None,
    params_z: tuple[str, str, np.ndarray] | None = None,
    phase_z: tuple[str, str, np.ndarray] | None = None,
    signal_axes: list[tuple[str, str, np.ndarray]] | None = None,
    signal_z: tuple[str, str, np.ndarray] | None = None,
) -> tuple[Path, Path, Path]:
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
            tags="jpa/auto_optimize/params",
        )
    )
    phases_path = Path(
        save_labber_data(
            str(_sidecar_base(base, "_phases", numbered=numbered)),
            z=phase_z or ("Phase", "a.u.", np.asarray(result.phases, dtype=np.int32)),
            axes=[("Iteration", "a.u.", iterations)],
            comment=comment if phase_comment is None else phase_comment,
            tags="jpa/auto_optimize/phases",
        )
    )
    signals_path = Path(
        save_labber_data(
            str(_sidecar_base(base, "_signals", numbered=numbered)),
            z=signal_z
            or ("Signal", "a.u.", np.asarray(result.signals, dtype=np.float64)),
            axes=signal_axes or [("Iteration", "a.u.", iterations)],
            comment=comment if signal_comment is None else signal_comment,
            tags="jpa/auto_optimize/signals",
        )
    )
    return params_path, phases_path, signals_path


def test_jpa_auto_save_writes_one_grouped_hdf5_and_loads_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jpa_mod, "make_comment", lambda _cfg, _comment: "stored")

    result = _sample_result(with_cfg=True)
    AutoOptimizeExp().save(str(tmp_path / "jpa_auto"), result=result)

    path = tmp_path / "jpa_auto.hdf5"
    assert path.exists()
    assert list(tmp_path.glob("jpa_auto*_params*.hdf5")) == []
    assert list(tmp_path.glob("jpa_auto*_phases*.hdf5")) == []
    assert list(tmp_path.glob("jpa_auto*_signals*.hdf5")) == []

    grouped = load_grouped_labber_data(str(path), required_roles=JPA_AUTO_GROUPED_ROLES)
    assert grouped.metadata.comment == "stored"
    assert grouped.metadata.tags == ["jpa/auto_optimize"]
    assert list(grouped.roles) == [DatasetRole(role) for role in JPA_AUTO_GROUPED_ROLES]

    expected_axis = np.arange(result.params.shape[0], dtype=np.int64)
    for role in JPA_AUTO_GROUPED_ROLES:
        payload = grouped.roles[DatasetRole(role)]
        assert [(axis.name, axis.unit) for axis in payload.axes] == [
            ("Iteration", "a.u.")
        ]
        assert np.array_equal(payload.axes[0].values, expected_axis)

    assert np.allclose(
        grouped.roles[DatasetRole(JPA_AUTO_FLUX_ROLE)].z,
        result.params[:, 0],
    )
    flux_payload = grouped.roles[DatasetRole(JPA_AUTO_FLUX_ROLE)]
    assert (flux_payload.data.name, flux_payload.data.unit) == ("JPA Flux", "a.u.")
    assert np.array_equal(flux_payload.data.values, result.params[:, 0])
    assert np.allclose(
        grouped.roles[DatasetRole(JPA_AUTO_FREQ_ROLE)].z,
        result.params[:, 1] * 1e6,
    )
    assert np.allclose(
        grouped.roles[DatasetRole(JPA_AUTO_POWER_ROLE)].z,
        result.params[:, 2],
    )
    assert np.array_equal(
        grouped.roles[DatasetRole(JPA_AUTO_PHASE_ROLE)].z,
        result.phases.astype(np.int64),
    )
    assert np.allclose(grouped.roles[DatasetRole(JPA_AUTO_SNR_ROLE)].z, result.signals)

    loaded = AutoOptimizeExp().load(str(path))
    assert np.allclose(loaded.params, result.params)
    assert np.array_equal(loaded.phases, result.phases)
    assert loaded.phases.dtype == np.int32
    assert np.allclose(loaded.signals, result.signals)


def test_jpa_auto_save_fast_fails_without_cfg_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cfg_snapshot"):
        AutoOptimizeExp().save(str(tmp_path / "jpa_auto"), result=_sample_result())


@pytest.mark.parametrize(
    "result",
    [
        JPAOptimizeResult(
            params=np.ones((4, 2), dtype=np.float64),
            phases=np.ones(4, dtype=np.int32),
            signals=np.ones(4, dtype=np.float64),
            cfg_snapshot=cast(JPAOptCfg, object()),
        ),
        JPAOptimizeResult(
            params=np.ones((4, 3), dtype=np.float64),
            phases=np.ones(3, dtype=np.int32),
            signals=np.ones(4, dtype=np.float64),
            cfg_snapshot=cast(JPAOptCfg, object()),
        ),
        JPAOptimizeResult(
            params=np.ones((4, 3), dtype=np.float64),
            phases=np.ones(4, dtype=np.int32),
            signals=np.ones(3, dtype=np.float64),
            cfg_snapshot=cast(JPAOptCfg, object()),
        ),
    ],
)
def test_jpa_auto_save_fast_fails_on_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: JPAOptimizeResult
) -> None:
    monkeypatch.setattr(jpa_mod, "make_comment", lambda _cfg, _comment: "stored")

    with pytest.raises(ValueError, match="JPA auto-optimize"):
        AutoOptimizeExp().save(str(tmp_path / "bad"), result=result)


def test_jpa_auto_grouped_loader_rejects_incomplete_or_wrong_role_set(
    tmp_path: Path,
) -> None:
    axis = [("Iteration", "a.u.", np.arange(2, dtype=np.int64))]
    payload = LabberPayload(("SNR", "a.u.", np.ones(2)), axes=axis)

    save_grouped_labber_data(str(tmp_path / "missing"), {JPA_AUTO_SNR_ROLE: payload})
    with pytest.raises(ValueError, match="missing required dataset role"):
        load_jpa_auto_grouped_result(str(tmp_path / "missing.hdf5"))

    save_grouped_labber_data(
        str(tmp_path / "wrong"),
        {role: payload for role in (*JPA_AUTO_GROUPED_ROLES, "unexpected_role")},
    )
    with pytest.raises(ValueError, match="unknown dataset role"):
        load_jpa_auto_grouped_result(str(tmp_path / "wrong.hdf5"))


def test_jpa_auto_grouped_loader_rejects_non_integer_phase(tmp_path: Path) -> None:
    result = _sample_result()
    roles = jpa_mod.jpa_auto_result_to_grouped_payloads(result)
    roles[JPA_AUTO_PHASE_ROLE] = LabberPayload(
        ("JPA Phase", "index", np.array([0.0, 1.5, 2.0, 3.0])),
        axes=[("Iteration", "a.u.", np.arange(4, dtype=np.int64))],
    )
    save_grouped_labber_data(str(tmp_path / "bad_phase"), roles)

    with pytest.raises(ValueError, match="integers"):
        load_jpa_auto_grouped_result(str(tmp_path / "bad_phase.hdf5"))


def test_jpa_auto_runtime_load_rejects_legacy_sidecar_artifact(tmp_path: Path) -> None:
    base = tmp_path / "legacy"
    params_path, _phases_path, _signals_path = _write_legacy_jpa_sidecars(
        base, _sample_result()
    )

    with pytest.raises(FileNotFoundError):
        AutoOptimizeExp().load(str(base))
    with pytest.raises(ValueError, match="not a grouped"):
        AutoOptimizeExp().load(str(params_path))


def test_migrate_jpa_auto_legacy_sidecars_to_grouped_hdf5_keeps_input_read_only(
    tmp_path: Path,
) -> None:
    base = tmp_path / "legacy"
    result = _sample_result()
    paths = _write_legacy_jpa_sidecars(base, result)
    input_bytes = tuple(path.read_bytes() for path in paths)

    migrated = migrate_experiment_data(
        experiment="jpa/jpa_auto_optimize",
        input_path=base,
        output_path=tmp_path / "canonical",
    )

    assert tuple(path.read_bytes() for path in paths) == input_bytes
    loaded = load_jpa_auto_grouped_result(str(migrated))
    assert np.allclose(loaded.params, result.params)
    assert np.array_equal(loaded.phases, result.phases)
    assert np.allclose(loaded.signals, result.signals)

    grouped = load_grouped_labber_data(
        str(migrated), required_roles=JPA_AUTO_GROUPED_ROLES
    )
    assert grouped.metadata.comment == "legacy comment"
    assert grouped.metadata.tags == ["jpa/auto_optimize"]


def test_migrate_jpa_auto_requires_overwrite_and_allows_overwrite(
    tmp_path: Path,
) -> None:
    base = tmp_path / "legacy"
    _write_legacy_jpa_sidecars(base, _sample_result())
    output = tmp_path / "canonical.hdf5"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=base,
            output_path=output,
        )
    assert output.read_bytes() == b"existing"

    migrated = migrate_experiment_data(
        experiment="jpa/jpa_auto_optimize",
        input_path=base,
        output_path=output,
        overwrite=True,
    )
    assert migrated == output
    loaded = load_jpa_auto_grouped_result(str(output))
    assert np.allclose(loaded.params, _sample_result().params)


def test_migrate_jpa_auto_accepts_numbered_legacy_sidecars(tmp_path: Path) -> None:
    base = tmp_path / "legacy"
    result = _sample_result()
    _write_legacy_jpa_sidecars(base, result, numbered=True)

    migrated = migrate_experiment_data(
        experiment="jpa/jpa_auto_optimize",
        input_path=base,
        output_path=tmp_path / "canonical.hdf5",
    )

    loaded = load_jpa_auto_grouped_result(str(migrated))
    assert np.allclose(loaded.params, result.params)
    assert np.array_equal(loaded.phases, result.phases)
    assert np.allclose(loaded.signals, result.signals)


def test_migrate_jpa_auto_rejects_missing_ambiguous_and_invalid_sidecars(
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
    save_labber_data(
        str(_sidecar_base(missing_base, "_signals")),
        z=("Signal", "a.u.", result.signals),
        axes=[("Iteration", "a.u.", np.arange(result.params.shape[0]))],
    )
    with pytest.raises(FileNotFoundError, match="phases"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=missing_base,
            output_path=tmp_path / "missing_out",
        )

    ambiguous_base = tmp_path / "ambiguous"
    _write_legacy_jpa_sidecars(ambiguous_base, result)
    _write_legacy_jpa_sidecars(ambiguous_base, result, numbered=True)
    with pytest.raises(ValueError, match="ambiguous legacy sidecar"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=ambiguous_base,
            output_path=tmp_path / "ambiguous_out",
        )

    phase_base = tmp_path / "phase"
    _write_legacy_jpa_sidecars(
        phase_base,
        result,
        phase_z=("Phase", "a.u.", np.array([0.0, 1.5, 2.0, 3.0])),
    )
    with pytest.raises(ValueError, match="integers"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=phase_base,
            output_path=tmp_path / "phase_out",
        )

    wrong_axis_base = tmp_path / "wrong_axis"
    _write_legacy_jpa_sidecars(
        wrong_axis_base,
        result,
        params_axes=[
            ("Iteration", "a.u.", np.arange(result.params.shape[0])),
            ("Bad Parameter Type", "a.u.", np.array([0, 1, 2])),
        ],
    )
    with pytest.raises(ValueError, match="axis 1"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=wrong_axis_base,
            output_path=tmp_path / "wrong_axis_out",
        )

    mismatched_axis_base = tmp_path / "mismatched_axis"
    _write_legacy_jpa_sidecars(
        mismatched_axis_base,
        result,
        signal_axes=[("Iteration", "a.u.", np.arange(result.params.shape[0]) + 1)],
    )
    with pytest.raises(ValueError, match="Iteration axis"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=mismatched_axis_base,
            output_path=tmp_path / "mismatched_axis_out",
        )

    metadata_base = tmp_path / "metadata"
    _write_legacy_jpa_sidecars(metadata_base, result, signal_comment="different")
    with pytest.raises(ValueError, match="comment metadata"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=metadata_base,
            output_path=tmp_path / "metadata_out",
        )

    complex_base = tmp_path / "complex"
    _write_legacy_jpa_sidecars(
        complex_base,
        result,
        signal_z=("Signal", "a.u.", result.signals + 1j),
    )
    with pytest.raises(ValueError, match="imaginary"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=complex_base,
            output_path=tmp_path / "complex_out",
        )


# --- JPAH-03: neutral a.u. flux contract + legacy-A grouped migration ---------


def _legacy_a_grouped_payloads(
    result: JPAOptimizeResult,
) -> dict[str, LabberPayload]:
    """The grouped layout the pre-a.u. spec wrote: only jpa_flux z unit is 'A'
    (same unscaled device-native values, everything else already canonical)."""
    axis = [("Iteration", "a.u.", np.arange(result.params.shape[0], dtype=np.int64))]
    return {
        JPA_AUTO_FLUX_ROLE: LabberPayload(
            ("JPA Flux", "A", result.params[:, 0]), axes=axis
        ),
        JPA_AUTO_FREQ_ROLE: LabberPayload(
            ("JPA Frequency", "Hz", result.params[:, 1] * 1e6), axes=axis
        ),
        JPA_AUTO_POWER_ROLE: LabberPayload(
            ("JPA Power", "dBm", result.params[:, 2]), axes=axis
        ),
        JPA_AUTO_PHASE_ROLE: LabberPayload(
            ("JPA Phase", "index", result.phases.astype(np.int64)), axes=axis
        ),
        JPA_AUTO_SNR_ROLE: LabberPayload(("SNR", "a.u.", result.signals), axes=axis),
    }


def _write_legacy_a_grouped(
    base: Path, result: JPAOptimizeResult, *, comment: str = "legacy comment"
) -> Path:
    return Path(
        save_grouped_labber_data(
            str(base),
            _legacy_a_grouped_payloads(result),
            metadata=LabberMetadata(comment=comment, tags=["jpa/auto_optimize"]),
        )
    )


def test_jpa_auto_grouped_loader_rejects_legacy_a_flux_unit(tmp_path: Path) -> None:
    path = _write_legacy_a_grouped(tmp_path / "legacy", _sample_result())

    with pytest.raises(ValueError, match="unit"):
        load_jpa_auto_grouped_result(str(path))


def test_migrate_jpa_auto_legacy_a_grouped_to_canonical_preserves_input_and_data(
    tmp_path: Path,
) -> None:
    result = _sample_result()
    input_path = _write_legacy_a_grouped(tmp_path / "legacy", result)
    input_bytes = input_path.read_bytes()

    migrated = migrate_experiment_data(
        experiment="jpa/jpa_auto_optimize/legacy_a",
        input_path=input_path,
        output_path=tmp_path / "canonical",
    )

    assert input_path.read_bytes() == input_bytes
    assert migrated == tmp_path / "canonical.hdf5"

    grouped = load_grouped_labber_data(
        str(migrated), required_roles=JPA_AUTO_GROUPED_ROLES
    )
    assert grouped.metadata.comment == "legacy comment"
    assert grouped.metadata.tags == ["jpa/auto_optimize"]
    flux_payload = grouped.roles[DatasetRole(JPA_AUTO_FLUX_ROLE)]
    assert (flux_payload.data.name, flux_payload.data.unit) == ("JPA Flux", "a.u.")
    assert np.array_equal(flux_payload.data.values, result.params[:, 0])

    loaded = load_jpa_auto_grouped_result(str(migrated))
    assert np.allclose(loaded.params, result.params)
    assert np.array_equal(loaded.phases, result.phases)
    assert np.allclose(loaded.signals, result.signals)


def test_migrate_jpa_auto_legacy_a_rejects_canonical_or_mismatched_input(
    tmp_path: Path,
) -> None:
    result = _sample_result()
    canonical_payloads = _legacy_a_grouped_payloads(result)
    canonical_payloads[JPA_AUTO_FLUX_ROLE] = LabberPayload(
        ("JPA Flux", "a.u.", result.params[:, 0]),
        axes=[("Iteration", "a.u.", np.arange(result.params.shape[0], dtype=np.int64))],
    )
    canonical_path = Path(
        save_grouped_labber_data(
            str(tmp_path / "canonical_input"),
            canonical_payloads,
            metadata=LabberMetadata(comment="canonical"),
        )
    )
    with pytest.raises(ValueError, match="'A'"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize/legacy_a",
            input_path=canonical_path,
            output_path=tmp_path / "out",
        )

    wrong_freq = _legacy_a_grouped_payloads(result)
    wrong_freq[JPA_AUTO_FREQ_ROLE] = LabberPayload(
        ("JPA Frequency", "MHz", result.params[:, 1]),  # legacy files store Hz
        axes=[("Iteration", "a.u.", np.arange(result.params.shape[0], dtype=np.int64))],
    )
    wrong_path = Path(
        save_grouped_labber_data(
            str(tmp_path / "wrong_freq"),
            wrong_freq,
            metadata=LabberMetadata(comment="legacy"),
        )
    )
    with pytest.raises(ValueError, match="'MHz'"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize/legacy_a",
            input_path=wrong_path,
            output_path=tmp_path / "out2",
        )


def test_migrate_jpa_auto_legacy_a_requires_overwrite(tmp_path: Path) -> None:
    input_path = _write_legacy_a_grouped(tmp_path / "legacy", _sample_result())
    output = tmp_path / "canonical.hdf5"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize/legacy_a",
            input_path=input_path,
            output_path=output,
        )
    assert output.read_bytes() == b"existing"


def test_migrate_jpa_auto_legacy_a_rejects_same_input_and_output_with_overwrite(
    tmp_path: Path,
) -> None:
    input_path = _write_legacy_a_grouped(tmp_path / "legacy", _sample_result())
    input_bytes = input_path.read_bytes()

    with pytest.raises(ValueError, match="same file"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize/legacy_a",
            input_path=input_path,
            output_path=input_path,
            overwrite=True,
        )
    assert input_path.read_bytes() == input_bytes


def _jpa_opt_cfg_for_comment() -> JPAOptCfg:
    def pulse() -> PulseCfg:
        return PulseCfg(
            waveform=ConstWaveformCfg(length=1.0),
            ch=0,
            nqz=1,
            freq=3000.0,
            gain=0.3,
        )

    return JPAOptCfg(
        reps=1,
        rounds=2,
        dev={
            "jpa_flux_dev": FakeDeviceInfo(address="fake_flux", label="jpa_flux_dev"),
            "jpa_rf_dev": FakeDeviceInfo(address="fake_rf", label="jpa_rf_dev"),
        },
        modules=JPAOptModuleCfg(
            reset=None,
            pi_pulse=pulse(),
            readout=DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=6000.0, gen_ch=0),
        ),
        sweep=JPAOptSweepCfg(
            jpa_flux=SweepCfg(start=0.1, stop=0.3, expts=3, step=0.1),
            jpa_freq=SweepCfg(start=7200.0, stop=7240.0, expts=5, step=10.0),
            jpa_power=SweepCfg(start=-20.0, stop=-10.0, expts=3, step=5.0),
        ),
    )


def test_migrate_jpa_auto_legacy_a_preserves_full_metadata_and_timestamps(
    tmp_path: Path,
) -> None:
    result = _sample_result()
    creation_time = 1750000000.123
    entry_ts = np.array([creation_time + 5.25])
    comment = make_comment(_jpa_opt_cfg_for_comment(), "legacy note")
    payloads = {
        role: LabberPayload(
            (payload.data.name, payload.data.unit, payload.data.values),
            axes=payload.axes,
            timestamps=entry_ts,
        )
        for role, payload in _legacy_a_grouped_payloads(result).items()
    }
    input_path = Path(
        save_grouped_labber_data(
            str(tmp_path / "legacy"),
            payloads,
            metadata=LabberMetadata(
                comment=comment,
                tags=["legacy_tag"],
                project="jpa-cal",
                user="operator-1",
                creation_time=creation_time,
            ),
        )
    )
    input_grouped = load_grouped_labber_data(
        str(input_path), required_roles=JPA_AUTO_GROUPED_ROLES
    )
    input_ts = {
        str(role): payload.timestamps for role, payload in input_grouped.roles.items()
    }
    assert all(ts is not None for ts in input_ts.values())

    migrated = migrate_experiment_data(
        experiment="jpa/jpa_auto_optimize/legacy_a",
        input_path=input_path,
        output_path=tmp_path / "canonical",
    )

    grouped = load_grouped_labber_data(
        str(migrated), required_roles=JPA_AUTO_GROUPED_ROLES
    )
    assert grouped.metadata.comment == comment
    assert grouped.metadata.tags == ["legacy_tag"]
    assert grouped.metadata.project == "jpa-cal"
    assert grouped.metadata.user == "operator-1"
    assert grouped.metadata.creation_time == creation_time
    flux_payload = grouped.roles[DatasetRole(JPA_AUTO_FLUX_ROLE)]
    assert (flux_payload.data.name, flux_payload.data.unit) == ("JPA Flux", "a.u.")
    assert np.array_equal(flux_payload.data.values, result.params[:, 0])
    for role in JPA_AUTO_GROUPED_ROLES:
        out_ts = grouped.roles[DatasetRole(role)].timestamps
        assert out_ts is not None
        np.testing.assert_allclose(out_ts, input_ts[str(role)], rtol=1e-12)

    loaded = load_jpa_auto_grouped_result(str(migrated))
    assert np.allclose(loaded.params, result.params)
    assert loaded.cfg_snapshot is not None
    assert loaded.cfg_snapshot.sweep.jpa_freq.start == 7200.0
    assert loaded.cfg_snapshot.sweep.jpa_freq.stop == 7240.0


def test_migrate_jpa_auto_rejects_output_aliasing_params_sidecar_with_overwrite(
    tmp_path: Path,
) -> None:
    base = tmp_path / "legacy"
    result = _sample_result()
    params_path, phases_path, signals_path = _write_legacy_jpa_sidecars(base, result)
    input_bytes = tuple(
        path.read_bytes() for path in (params_path, phases_path, signals_path)
    )

    with pytest.raises(ValueError, match="same file"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=base,
            output_path=params_path,
            overwrite=True,
        )
    assert (
        tuple(path.read_bytes() for path in (params_path, phases_path, signals_path))
        == input_bytes
    )


def test_migrate_jpa_auto_rejects_symlink_alias_of_sidecar_with_overwrite(
    tmp_path: Path,
) -> None:
    base = tmp_path / "legacy"
    result = _sample_result()
    params_path, _phases_path, _signals_path = _write_legacy_jpa_sidecars(base, result)
    alias = tmp_path / "alias.hdf5"
    alias.symlink_to(params_path)
    params_bytes = params_path.read_bytes()

    with pytest.raises(ValueError, match="same file"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=base,
            output_path=alias,
            overwrite=True,
        )
    assert params_path.read_bytes() == params_bytes
    assert alias.read_bytes() == params_bytes


def test_migrate_jpa_auto_rejects_output_aliasing_numbered_sidecar_with_overwrite(
    tmp_path: Path,
) -> None:
    base = tmp_path / "legacy"
    result = _sample_result()
    params_path, _phases_path, _signals_path = _write_legacy_jpa_sidecars(
        base, result, numbered=True
    )
    params_bytes = params_path.read_bytes()

    with pytest.raises(ValueError, match="same file"):
        migrate_experiment_data(
            experiment="jpa/jpa_auto_optimize",
            input_path=base,
            output_path=params_path,
            overwrite=True,
        )
    assert params_path.read_bytes() == params_bytes
