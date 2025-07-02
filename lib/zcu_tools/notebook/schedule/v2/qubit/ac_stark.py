from copy import deepcopy
from typing import Tuple

import numpy as np
from zcu_tools.liveplot.jupyter import LivePlotter2D
from zcu_tools.notebook.single_qubit.process import minus_background, rotate2real
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    check_no_post_delay,
    make_readout,
    make_reset,
)

from ...tools import sweep2array, sweep2param
from ..template import sweep_hard_template


def qub_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(minus_background(signals, axis=1)).real  # type: ignore


def measure_ac_stark(soc, soccfg, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    check_no_post_delay(cfg["stark_pulse1"], "stark_pulse1")

    # force the order of sweep
    gain_sweep = cfg["sweep"]["gain"]
    freq_sweep = cfg["sweep"]["freq"]
    cfg["sweep"] = {"gain": gain_sweep, "freq": freq_sweep}

    prog = ModularProgramV2(
        soccfg,
        cfg,
        modules=[
            make_reset("reset", reset_cfg=cfg["reset"]),
            Pulse(
                name="stark_pulse1",
                cfg={
                    **cfg["stark_pulse1"],
                    "gain": sweep2param("gain", gain_sweep),
                },
            ),
            Pulse(
                name="stark_pulse2",
                cfg={
                    **cfg["stark_pulse2"],
                    "freq": sweep2param("freq", freq_sweep),
                },
            ),
            make_readout("readout", readout_cfg=cfg["readout"]),
        ],
    )

    amps = sweep2array(gain_sweep)
    freqs = sweep2array(freq_sweep)
    signals = sweep_hard_template(
        cfg,
        lambda _, cb: prog.acquire(soc, progress=True, callback=cb)[0][0].dot([1, 1j]),
        LivePlotter2D("Pulse gain", "Frequency (MHz)"),
        ticks=(amps, freqs),
        signal2real=qub_signal2real,
    )

    # get the actual amplitudes
    amps = prog.get_pulse_param("stark_pulse1", "gain", as_array=True)
    freqs = prog.get_pulse_param("stark_pulse2", "freq", as_array=True)
    assert isinstance(amps, np.ndarray), "amps should be an array"
    assert isinstance(freqs, np.ndarray), "freqs should be an array"

    return amps, freqs, signals
