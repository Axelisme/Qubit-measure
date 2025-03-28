# Persistence functions (load/save) for flux-dependent analysis

"""Persistence functions for flux-dependent analysis.

This module provides functions for loading and saving results from
flux-dependent spectroscopy analysis.
"""

import json
import os

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
