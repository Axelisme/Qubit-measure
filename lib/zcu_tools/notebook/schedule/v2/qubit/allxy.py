from copy import deepcopy
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v2 import ModularProgramV2, Pulse, make_readout, make_reset

from ...tools import format_sweep1D, sweep2array, sweep2param
from ..template import sweep1D_soft_template


def allxy_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


def measure_allxy(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    gate2pulse_map = dict(
        I=cfg.get("I_pulse"),
        X180=cfg["X180_pulse"],
        Y180=cfg["Y180_pulse"],
        X90=cfg["X90_pulse"],
        Y90=cfg["Y90_pulse"],
    )

    sequence = [
        ("I", "I"),
        ("X180", "X180"),
        ("Y180", "Y180"),
        ("X180", "Y180"),
        ("Y180", "X180"),
        ("X90", "I"),
        ("Y90", "I"),
        ("X90", "Y90"),
        ("Y90", "X90"),
        ("X90", "Y180"),
        ("Y90", "X180"),
        ("X180", "Y90"),
        ("Y180", "X90"),
        ("X90", "X180"),
        ("X180", "X90"),
        ("Y90", "Y180"),
        ("Y180", "Y90"),
        ("X180", "I"),
        ("Y180", "I"),
        ("X90", "X90"),
        ("Y90", "Y90"),
    ]

    def updateCfg(cfg, i, _):
        cfg["current_gates"] = sequence[i]

    def measure_fn(cfg, callback):
        gate1, gate2 = cfg["current_gates"]
        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg["reset"]),
                Pulse(name="first_pulse", pulse=gate2pulse_map[gate1]),
                Pulse(name="second_pulse", pulse=gate2pulse_map[gate2]),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        return prog.acquire(soc, callback=callback)

    liveplotter = LivePlotter1D(
        xlabel="Gate",
        ylabel="Signal",
        title="All XY",
        line_kwargs=[
            {
                "marker": ".",
                "linestyle": "",
                "markersize": 5,
            }
        ],
    )

    ax = liveplotter.axs[0]
    assert isinstance(ax, plt.Axes)

    # set x ticks to gate names, rotated 45 degrees for better visibility
    ax.set_xticks(np.arange(len(sequence)))
    ax.set_xticklabels(
        [f"({gate1}, {gate2})" for gate1, gate2 in sequence], rotation=45, ha="right"
    )

    _, signals = sweep1D_soft_template(
        cfg,
        measure_fn,
        liveplotter,
        xs=np.arange(len(sequence)),
        updateCfg=updateCfg,
        signal2real=allxy_signal2real,
    )

    return sequence, signals
