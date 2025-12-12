import json
import os
from typing import Any, Dict, Tuple

import h5py as h5
import numpy as np


def format_rawdata(
    As: np.ndarray, fpts: np.ndarray, spectrum: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpts = fpts / 1e9  # convert to GHz

    if As[0] > As[-1]:  # Ensure that the fluxes are in increasing
        As = As[::-1]
        spectrum = spectrum[::-1, :]
    if fpts[0] > fpts[-1]:  # Ensure that the frequencies are in increasing
        fpts = fpts[::-1]
        spectrum = spectrum[:, ::-1]

    return As, fpts, spectrum


def dump_result(
    path: str,
    name: str,
    params: np.ndarray,
    cflx: float,
    period: float,
    allows: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(
            {
                "name": name,
                "params": {
                    "EJ": params[0],
                    "EC": params[1],
                    "EL": params[2],
                },
                "half flux": cflx,
                "period": period,
                "allows": allows,
            },
            f,
            indent=4,
        )


def load_result(
    path: str,
) -> Tuple[
    str, Tuple[float, float, float], float, float, Dict[str, Any], Dict[str, Any]
]:
    """
    Load the result from a json file

    Returns:
        name: str
        params: np.ndarray
        half_flux: float
        period: float
        allows: Dict[str, Any]
        data: Dict[str, Any], raw dict contain all result
    """

    with open(path, "r") as f:
        data = json.load(f)

    return (
        data["name"],
        (data["params"]["EJ"], data["params"]["EC"], data["params"]["EL"]),
        data["half flux"],
        data["period"],
        data["allows"],
        data,
    )


def update_result(path: str, update_dict: Dict[str, Any]) -> None:
    with open(path, "r") as f:
        data = json.load(f)

    data.update(update_dict)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def dump_spects(save_path: str, s_spects: Dict[str, Any], mode: str = "x") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5.File(save_path, mode) as f:
        for path, val in s_spects.items():
            grp = f.create_group(path)
            grp.create_dataset("mA_c", data=val["mA_c"])
            grp.create_dataset("period", data=val["period"])
            spect_grp = grp.create_group("spectrum")
            spect_grp.create_dataset("mAs", data=val["spectrum"]["mAs"])
            spect_grp.create_dataset("fpts", data=val["spectrum"]["fpts"])
            spect_grp.create_dataset("data", data=val["spectrum"]["data"])
            points_grp = grp.create_group("points")
            points_grp.create_dataset("mAs", data=val["points"]["mAs"])
            points_grp.create_dataset("fpts", data=val["points"]["fpts"])


def load_spects(load_path: str) -> Dict[str, Any]:
    s_spects = {}
    with h5.File(load_path, "r") as f:
        for key in f.keys():
            grp = f[key]
            s_spects.update(
                {
                    key: {
                        "mA_c": grp["mA_c"][()],  # type: ignore
                        "period": grp["period"][()],  # type: ignore
                        "spectrum": {
                            "mAs": grp["spectrum"]["mAs"][()],  # type: ignore
                            "fpts": grp["spectrum"]["fpts"][()],  # type: ignore
                            "data": grp["spectrum"]["data"][()],  # type: ignore
                        },
                        "points": {
                            "mAs": grp["points"]["mAs"][()],  # type: ignore
                            "fpts": grp["points"]["fpts"][()],  # type: ignore
                        },
                    }
                }
            )

    return s_spects
