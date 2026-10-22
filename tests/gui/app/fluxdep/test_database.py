"""Characterize the unchanged search metadata format at its persistence owner.

These cases accompany an owner move, not a new reader behavior/red proof.
Synthetic files exercise ranges, dimensions, errors and unreadable energy backing.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.database import (
    FluxdepDatabaseInfo,
    read_fluxdep_database_info,
)


def database_file(tmp_path: Path, *, flux_key: str = "fluxs") -> Path:
    """Create unsorted candidates with energy dimensions but no readable backing."""
    path = tmp_path / "search.h5"
    with h5py.File(path, "w") as database:
        database[flux_key] = [0.75, -0.25, 0.5]
        database["params"] = [[8.0, 0.5, 0.3], [3.0, 1.5, 0.1]]
        # Missing external storage makes energy values unavailable, not dimensions.
        database.create_dataset(
            "energies",
            shape=(2, 3, 4),
            dtype="f8",
            external=[(str(tmp_path / "absent-energies.raw"), 0, h5py.h5f.UNLIMITED)],
        )
    return path


@pytest.mark.parametrize("flux_key", ["fluxs", "flxs"])
def test_metadata_reports_ranges_without_reading_energy_values(
    tmp_path: Path, flux_key: str
) -> None:
    path = database_file(tmp_path, flux_key=flux_key)
    assert read_fluxdep_database_info(str(path)) == FluxdepDatabaseInfo(
        path=str(path),
        entries=2,
        levels=4,
        EJ=(3.0, 8.0),
        EC=(0.5, 1.5),
        EL=(0.1, 0.3),
        flux=(-0.25, 0.75),
    )


@pytest.mark.parametrize(
    "table,values",
    [
        ("fluxs", np.empty(0)),
        ("fluxs", np.zeros((1, 3))),
        ("fluxs", [0.0, np.nan, 1.0]),
        ("params", np.empty((0, 3))),
        ("params", np.zeros((2, 2))),
        ("params", [[1.0, np.inf, 3.0], [4.0, 5.0, 6.0]]),
        ("energies", np.zeros((1, 3, 4))),
        ("energies", np.zeros((2, 2, 4))),
        ("energies", np.zeros((2, 3))),
        ("energies", np.empty((2, 3, 0))),
    ],
    ids=[
        "empty-flux",
        "non-vector-flux",
        "nonfinite-flux",
        "empty-params",
        "param-columns",
        "nonfinite-params",
        "entry-count",
        "flux-count",
        "energy-rank",
        "zero-levels",
    ],
)
def test_invalid_tables_reject_metadata(
    tmp_path: Path, table: str, values: object
) -> None:
    path = database_file(tmp_path)
    with h5py.File(path, "a") as database:
        del database[table]
        database[table] = values
    with pytest.raises(
        ValueError, match="invalid database dimensions or parameter values"
    ):
        read_fluxdep_database_info(str(path))


@pytest.mark.parametrize("table", ["fluxs", "params", "energies"])
def test_group_instead_of_dataset_rejects_metadata(tmp_path: Path, table: str) -> None:
    path = database_file(tmp_path)
    with h5py.File(path, "a") as database:
        del database[table]
        database.create_group(table)
    with pytest.raises(ValueError, match="database tables must be datasets"):
        read_fluxdep_database_info(str(path))


def test_missing_dataset_propagates_key_error(tmp_path: Path) -> None:
    path = database_file(tmp_path)
    with h5py.File(path, "a") as database:
        del database["params"]
    with pytest.raises(KeyError):
        read_fluxdep_database_info(str(path))


def test_missing_file_propagates_native_io_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_fluxdep_database_info(str(tmp_path / "missing.h5"))
