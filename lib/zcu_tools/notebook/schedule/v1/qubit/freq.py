from copy import deepcopy
from typing import Tuple

from numpy import ndarray
from zcu_tools.liveplot.jupyter import LivePlotter1D
from zcu_tools.notebook.single_qubit.process import rotate2real
from zcu_tools.program.v1 import RFreqTwoToneProgram

from ...tools import sweep2array
from ..template import sweep1D_hard_template


def signal2real(signals: ndarray) -> ndarray:
    return rotate2real(signals).real


def measure_qub_freq(soc, soccfg, cfg) -> Tuple[ndarray, ndarray]:
    cfg = deepcopy(cfg)  # prevent in-place modification

    fpts = sweep2array(cfg["sweep"])

    def measure_fn(cfg, callback) -> Tuple[ndarray, ...]:
        prog = RFreqTwoToneProgram(soccfg, cfg)
        return prog.acquire(soc, progress=False, callback=callback)

    fpts, signals = sweep1D_hard_template(
        cfg,
        measure_fn,
        LivePlotter1D("Frequency (MHz)", "Amplitude"),
        xs=fpts,
    )

    return fpts, signals
