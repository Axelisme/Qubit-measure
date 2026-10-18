from __future__ import annotations

import os

import h5py as h5

from zcu_tools.analysis.fluxdep.models import SpectrumResult


def dump_spectrums(
    path: str, spectrums: dict[str, SpectrumResult], mode: str = "x"
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5.File(path, mode) as f:
        for name, spectrum in spectrums.items():
            grp = f.create_group(name)
            # ``type`` (OneTone / TwoTone) is metadata, stored as a group
            # attribute when present (NotRequired field).
            if "type" in spectrum:
                grp.attrs["type"] = spectrum["type"]
            grp.create_dataset("flux_half", data=spectrum["flux_half"])
            grp.create_dataset("flux_int", data=spectrum["flux_int"])
            grp.create_dataset("flux_period", data=spectrum["flux_period"])

            spect_data = spectrum["spectrum"]
            spect_grp = grp.create_group("spectrum")
            spect_grp.create_dataset("dev_values", data=spect_data["dev_values"])
            spect_grp.create_dataset("fluxs", data=spect_data["fluxs"])
            spect_grp.create_dataset("freqs", data=spect_data["freqs"])
            spect_grp.create_dataset("signals", data=spect_data["signals"])

            points_data = spectrum["points"]
            points_grp = grp.create_group("points")
            points_grp.create_dataset("dev_values", data=points_data["dev_values"])
            points_grp.create_dataset("fluxs", data=points_data["fluxs"])
            points_grp.create_dataset("freqs", data=points_data["freqs"])


def load_spectrums(path: str) -> dict[str, SpectrumResult]:
    """Read a spectrum collection without changing stored values or dtypes.

    The optional ``type`` group attribute is included only when present.
    Missing fields raise KeyError; non-group entries raise AssertionError.
    A calibration, spectrum, or points field that is not a Dataset raises
    TypeError identifying its HDF5 path.
    """
    spectrums = dict[str, SpectrumResult]()
    with h5.File(path, "r") as f:
        for name in f.keys():
            grp = f[name]
            assert isinstance(grp, h5.Group)
            spect_grp = grp["spectrum"]
            points_grp = grp["points"]
            assert isinstance(spect_grp, h5.Group)
            assert isinstance(points_grp, h5.Group)
            calibration_datasets: list[h5.Dataset] = []
            for field in ("flux_half", "flux_int", "flux_period"):
                dataset = grp[field]
                if not isinstance(dataset, h5.Dataset):
                    raise TypeError(f"Expected Dataset at {grp.name}/{field}")
                calibration_datasets.append(dataset)
            flux_half, flux_int, flux_period = calibration_datasets

            spect_datasets: list[h5.Dataset] = []
            for field in ("dev_values", "fluxs", "freqs", "signals"):
                dataset = spect_grp[field]
                if not isinstance(dataset, h5.Dataset):
                    raise TypeError(f"Expected Dataset at {spect_grp.name}/{field}")
                spect_datasets.append(dataset)
            spect_dev_values, spect_fluxs, spect_freqs, spect_signals = spect_datasets

            points_datasets: list[h5.Dataset] = []
            for field in ("dev_values", "fluxs", "freqs"):
                dataset = points_grp[field]
                if not isinstance(dataset, h5.Dataset):
                    raise TypeError(f"Expected Dataset at {points_grp.name}/{field}")
                points_datasets.append(dataset)
            points_dev_values, points_fluxs, points_freqs = points_datasets

            result = SpectrumResult(
                flux_half=flux_half[()],
                flux_int=flux_int[()],
                flux_period=flux_period[()],
                spectrum={
                    "dev_values": spect_dev_values[()],
                    "fluxs": spect_fluxs[()],
                    "freqs": spect_freqs[()],
                    "signals": spect_signals[()],
                },
                points={
                    "dev_values": points_dev_values[()],
                    "fluxs": points_fluxs[()],
                    "freqs": points_freqs[()],
                },
            )
            # ``type`` is optional for older spectrum files; current files store it
            # as a group attribute.
            if "type" in grp.attrs:
                result["type"] = str(grp.attrs["type"])
            spectrums[name] = result

    return spectrums
