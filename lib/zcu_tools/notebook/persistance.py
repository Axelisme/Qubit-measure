from __future__ import annotations

import json
import os
import re

import h5py as h5
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, TypedDict, NotRequired, Optional, cast


def format_rawdata(
    dev_values: NDArray[np.float64],
    freqs: NDArray[np.float64],  # in Hz
    signals: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    freqs = freqs / 1e9  # convert to GHz

    if dev_values[0] > dev_values[-1]:  # Ensure that the fluxes are in increasing
        dev_values = dev_values[::-1]
        signals = signals[::-1, :]
    if freqs[0] > freqs[-1]:  # Ensure that the frequencies are in increasing
        freqs = freqs[::-1]
        signals = signals[:, ::-1]

    return dev_values, freqs, signals


class TransitionDict(TypedDict, extra_items=list[tuple[int, int]]):
    r_f: NotRequired[float]
    sample_f: NotRequired[float]


class FluxDepFitResult(TypedDict):
    params: dict[str, float]
    flx_half: float
    flx_int: float
    flx_period: float
    plot_transitions: TransitionDict


class DispersiveResult(TypedDict):
    bare_rf: float
    g: float


class ResultData(TypedDict):
    name: str
    fluxdep_fit: NotRequired[FluxDepFitResult]
    dispersive: NotRequired[DispersiveResult]


def dump_result(
    path: str,
    name: str,
    fluxdep_fit: Optional[FluxDepFitResult] = None,
    dispersive: Optional[DispersiveResult] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    result = ResultData(name=name)
    if fluxdep_fit is not None:
        result["fluxdep_fit"] = fluxdep_fit
    if dispersive is not None:
        result["dispersive"] = dispersive

    with open(path, "w", encoding="utf8") as f:
        output = json.dumps(result, ensure_ascii=False, indent=4)

        def format_list(match):
            return " ".join(match.group(0).split())

        output = re.sub(r"(?<=\[)[^\[\]]+(?=\])", format_list, output)
        f.write(output)


def load_result(path: str) -> ResultData:
    """Load the result from a json file"""

    with open(path, "r") as f:
        result_data = json.load(f)

    return cast(ResultData, result_data)


def update_result(path: str, update_dict: dict[str, Any]) -> None:
    result_data = load_result(path)
    result_data.update(update_dict)  # type: ignore
    dump_result(path, **result_data)


class SpectrumData(TypedDict):
    dev_values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


class PointsData(TypedDict):
    dev_values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]


class SpectrumResult(TypedDict):
    flx_half: float
    flx_int: float
    flx_period: float
    spectrum: SpectrumData
    points: PointsData


def dump_spectrums(
    path: str, spectrums: dict[str, SpectrumResult], mode: str = "x"
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5.File(path, mode) as f:
        for name, spectrum in spectrums.items():
            grp = f.create_group(name)
            grp.create_dataset("flx_half", data=spectrum["flx_half"])
            grp.create_dataset("flx_int", data=spectrum["flx_int"])
            grp.create_dataset("flx_period", data=spectrum["flx_period"])

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
            spectrums[name] = SpectrumResult(
                flx_half=grp["flx_half"][()],  # type: ignore
                flx_int=grp["flx_int"][()],  # type: ignore
                flx_period=grp["flx_period"][()],  # type: ignore
                spectrum={
                    "dev_values": spect_grp["dev_values"][()],  # type: ignore
                    "fluxs": spect_grp["fluxs"][()],  # type: ignore
                    "freqs": spect_grp["freqs"][()],  # type: ignore
                    "signals": spect_grp["signals"][()],  # type: ignore
                },
                points={
                    "dev_values": points_grp["dev_values"][()],  # type: ignore
                    "fluxs": points_grp["fluxs"][()],  # type: ignore
                    "freqs": points_grp["freqs"][()],  # type: ignore
                },
            )

    return spectrums
