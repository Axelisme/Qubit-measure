from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import zcu_tools.experiment.v2.twotone.time_domain.cpmg as cpmg_mod
from zcu_tools.experiment.v2.twotone.time_domain.cpmg import (
    CPMG_GROUPED_ROLES,
    CPMG_LENGTHS_ROLE,
    CPMG_SIGNALS_ROLE,
    CPMG_Cfg,
    CPMG_Exp,
    CPMG_Result,
    load_cpmg_grouped_result,
)
from zcu_tools.utils.datasaver import DatasetRole, load_grouped_labber_data

from script.migrate_experiment_data import main, migrate_experiment_data


def _sample_arrays() -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
]:
    times = np.array([1, 2, 4], dtype=np.int64)
    lengths = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.11, 0.22, 0.33, 0.44],
            [0.12, 0.24, 0.36, 0.48],
        ],
        dtype=np.float64,
    )
    signals = lengths.astype(np.complex128) + 1j * (lengths + 0.5)
    return times, lengths, signals


def _sample_result(*, with_cfg: bool = False) -> CPMG_Result:
    times, lengths, signals = _sample_arrays()
    cfg = cast(CPMG_Cfg, object()) if with_cfg else None
    return CPMG_Result(ns=times, delays=lengths, signals=signals, cfg_snapshot=cfg)


def test_cpmg_save_writes_one_grouped_hdf5_and_loads_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cpmg_mod,
        "make_comment",
        lambda _cfg, _comment: "stored comment",
    )

    exp = CPMG_Exp()
    result = _sample_result(with_cfg=True)
    exp.save(str(tmp_path / "cpmg"), result=result, comment="ignored")

    path = tmp_path / "cpmg.hdf5"
    assert path.exists()
    assert list(tmp_path.glob("cpmg*.npz")) == []
    assert list(tmp_path.glob("cpmg_length*.hdf5")) == []
    assert list(tmp_path.glob("cpmg_signals*.hdf5")) == []

    grouped = load_grouped_labber_data(str(path), required_roles=CPMG_GROUPED_ROLES)
    assert grouped.metadata.comment == "stored comment"
    assert grouped.metadata.tags == ["twotone/ge/cpmg"]

    lengths_payload = grouped.roles[DatasetRole(CPMG_LENGTHS_ROLE)]
    signals_payload = grouped.roles[DatasetRole(CPMG_SIGNALS_ROLE)]
    assert [axis.name for axis in lengths_payload.axes] == [
        "Time Index",
        "Number of Pi",
    ]
    assert [axis.name for axis in signals_payload.axes] == [
        "Time Index",
        "Number of Pi",
    ]
    assert np.allclose(lengths_payload.z, 1e-6 * result.delays)
    assert np.allclose(signals_payload.z, result.signals)

    loaded = CPMG_Exp().load(str(path))
    assert np.array_equal(loaded.ns, result.ns)
    assert np.allclose(loaded.delays, result.delays)
    assert np.allclose(loaded.signals, result.signals)


def test_cpmg_grouped_loader_rejects_incomplete_role_set(tmp_path: Path) -> None:
    from zcu_tools.utils.datasaver import LabberPayload, save_grouped_labber_data

    times, lengths, _signals = _sample_arrays()
    save_grouped_labber_data(
        str(tmp_path / "incomplete"),
        {
            CPMG_LENGTHS_ROLE: LabberPayload(
                ("Length", "s", 1e-6 * lengths),
                axes=[
                    ("Time Index", "a.u.", np.arange(lengths.shape[1])),
                    ("Number of Pi", "a.u.", times.astype(np.float64)),
                ],
            )
        },
    )

    with pytest.raises(ValueError, match="missing required dataset role"):
        load_cpmg_grouped_result(str(tmp_path / "incomplete.hdf5"))


def test_migrate_cpmg_npz_to_grouped_hdf5_keeps_input_read_only(
    tmp_path: Path,
) -> None:
    times, lengths, signals = _sample_arrays()
    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(
        legacy_path,
        times=times,
        lengths=lengths,
        signals2D=signals,
        comment=np.asarray("legacy comment"),
    )
    legacy_bytes = legacy_path.read_bytes()

    migrated = migrate_experiment_data(
        experiment="twotone/cpmg",
        input_path=legacy_path,
        output_path=tmp_path / "migrated.hdf5",
    )

    assert legacy_path.read_bytes() == legacy_bytes
    assert migrated == tmp_path / "migrated.hdf5"
    loaded = load_cpmg_grouped_result(str(migrated))
    assert np.array_equal(loaded.ns, times)
    assert np.allclose(loaded.delays, lengths)
    assert np.allclose(loaded.signals, signals)

    grouped = load_grouped_labber_data(str(migrated), required_roles=CPMG_GROUPED_ROLES)
    assert grouped.metadata.comment == "legacy comment"


def test_migration_cli_requires_overwrite_and_formats_output_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    times, lengths, signals = _sample_arrays()
    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(legacy_path, times=times, lengths=lengths, signals2D=signals)

    output = tmp_path / "migrated.h5"
    formatted_output = tmp_path / "migrated.hdf5"
    formatted_output.write_text("existing")
    with pytest.raises(SystemExit):
        main(
            [
                "--experiment",
                "twotone/cpmg",
                "--input",
                str(legacy_path),
                "--output",
                str(output),
            ]
        )
    assert formatted_output.read_text() == "existing"

    assert (
        main(
            [
                "--experiment",
                "twotone/cpmg",
                "--input",
                str(legacy_path),
                "--output",
                str(output),
                "--overwrite",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert str(formatted_output) in captured.out
    loaded = load_cpmg_grouped_result(str(formatted_output))
    assert np.array_equal(loaded.ns, times)
