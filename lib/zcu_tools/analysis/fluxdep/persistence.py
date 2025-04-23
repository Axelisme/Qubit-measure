# Persistence functions (load/save) for flux-dependent analysis

"""Persistence functions for flux-dependent analysis.

This module provides functions for loading and saving results from
flux-dependent spectroscopy analysis.
"""

import json
import os

import h5py as h5
import numpy as np


def dump_result(path, name, params, cflx, period, allows):
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


def load_result(path):
    with open(path, "r") as f:
        data = json.load(f)

    return (
        data["name"],
        np.array([data["params"]["EJ"], data["params"]["EC"], data["params"]["EL"]]),
        data["half flux"],
        data["period"],
        data["allows"],
    )


def dump_spects(save_path, s_spects, mode="x"):
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


def load_spects(load_path):
    s_spects = {}
    with h5.File(load_path, "r") as f:
        for key in f.keys():
            grp = f[key]
            s_spects.update(
                {
                    key: {
                        "mA_c": grp["mA_c"][()],
                        "period": grp["period"][()],
                        "spectrum": {
                            "mAs": grp["spectrum"]["mAs"][()],
                            "fpts": grp["spectrum"]["fpts"][()],
                            "data": grp["spectrum"]["data"][()],
                        },
                        "points": {
                            "mAs": grp["points"]["mAs"][()],
                            "fpts": grp["points"]["fpts"][()],
                        },
                    }
                }
            )

    return s_spects
