"""Metadata reader for the fluxonium candidate search HDF5 format.

The energy table is inspected only for dimensions; no energy values are read.
This owner does not perform numerical analysis or populate a search cache.
"""

from dataclasses import dataclass

import numpy as np
from h5py import Dataset, File


@dataclass(frozen=True)
class FluxdepDatabaseInfo:
    """Fluxonium search database metadata, without loading energy values.

    path is the caller's HDF5 filename, retained without normalization.
    entries is the positive candidate count; levels is the positive level count.
    EJ, EC and EL are finite inclusive (minimum, maximum) GHz ranges over all
    candidates. flux is the finite inclusive normalized flux (Phi_0) range.
    """

    path: str
    entries: int
    levels: int
    EJ: tuple[float, float]
    EC: tuple[float, float]
    EL: tuple[float, float]
    flux: tuple[float, float]


def read_fluxdep_database_info(path: str) -> FluxdepDatabaseInfo:
    """Read candidate ranges and energy dimensions from an HDF5 search file.

    Accepts fluxs or the historical flxs spelling, params (entries, 3), and
    energies (entries, flux_count, levels). Does not read energy values or
    populate the search cache. Invalid/empty tables raise ValueError; missing
    datasets and native file I/O errors propagate. Does not mutate the file.
    """
    with File(path, "r") as database:
        flux_table = database["fluxs" if "fluxs" in database else "flxs"]
        param_table = database["params"]
        energy_table = database["energies"]
        if (
            not isinstance(flux_table, Dataset)
            or not isinstance(param_table, Dataset)
            or not isinstance(energy_table, Dataset)
        ):
            raise ValueError("database tables must be datasets")
        fluxs = np.asarray(flux_table, dtype=np.float64)
        params = np.asarray(param_table, dtype=np.float64)
        shape = energy_table.shape
        if (
            fluxs.ndim != 1
            or fluxs.size == 0
            or params.ndim != 2
            or params.shape[1] != 3
            or params.shape[0] == 0
            or shape is None
            or len(shape) != 3
            or shape[:2] != (params.shape[0], fluxs.size)
            or shape[2] < 1
            or not np.all(np.isfinite(fluxs))
            or not np.all(np.isfinite(params))
        ):
            raise ValueError("invalid database dimensions or parameter values")
        ranges = [
            (float(np.min(params[:, index])), float(np.max(params[:, index])))
            for index in range(3)
        ]
        return FluxdepDatabaseInfo(
            path,
            shape[0],
            shape[2],
            ranges[0],
            ranges[1],
            ranges[2],
            (float(np.min(fluxs)), float(np.max(fluxs))),
        )
