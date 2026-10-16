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
    spectrums = dict[str, SpectrumResult]()
    with h5.File(path, "r") as f:
        for name in f.keys():
            grp = f[name]
            assert isinstance(grp, h5.Group)
            spect_grp = grp["spectrum"]
            points_grp = grp["points"]
            assert isinstance(spect_grp, h5.Group)
            assert isinstance(points_grp, h5.Group)
            # h5py group lookup does not distinguish datasets from other objects.
            # Keep the existing raw reads and their errors, without dtype coercion.
            result = SpectrumResult(
                flux_half=grp["flux_half"][()],  # pyright: ignore[reportIndexIssue, reportArgumentType]
                flux_int=grp["flux_int"][()],  # pyright: ignore[reportIndexIssue, reportArgumentType]
                flux_period=grp["flux_period"][()],  # pyright: ignore[reportIndexIssue, reportArgumentType]
                spectrum={
                    "dev_values": spect_grp["dev_values"][()],  # pyright: ignore[reportIndexIssue, reportArgumentType]
                    "fluxs": spect_grp["fluxs"][()],  # pyright: ignore[reportIndexIssue]
                    "freqs": spect_grp["freqs"][()],  # pyright: ignore[reportIndexIssue]
                    "signals": spect_grp["signals"][()],  # pyright: ignore[reportIndexIssue]
                },
                points={
                    "dev_values": points_grp["dev_values"][()],  # pyright: ignore[reportIndexIssue, reportArgumentType]
                    "fluxs": points_grp["fluxs"][()],  # pyright: ignore[reportIndexIssue]
                    "freqs": points_grp["freqs"][()],  # pyright: ignore[reportIndexIssue]
                },
            )
            # ``type`` is optional for older spectrum files; current files store it
            # as a group attribute.
            if "type" in grp.attrs:
                result["type"] = str(grp.attrs["type"])
            spectrums[name] = result

    return spectrums
