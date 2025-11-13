from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment.utils import sweep2array
from zcu_tools.experiment.v2.runner import HardTask, ResultType, TaskContext
from zcu_tools.experiment.v2.utils import wrap_earlystop_check
from zcu_tools.liveplot import AbsLivePlotter, LivePlotter2DwithLine
from zcu_tools.program.v2 import ModularProgramV2, Pulse, Readout, Reset, sweep2param
from zcu_tools.utils.fitting import fit_qubit_freq
from zcu_tools.utils.process import rotate2real

from ..executor import AutoMeasurement

# class QubitFreqTask(AbsTask):
#     def measure_freq_fn(
#         self, ctx: TaskContext, update_hook: Callable
#     ) -> List[np.ndarray]:
#         cfg = deepcopy(ctx.cfg)

#         cfg["sweep"] = {"detune": self.detune_sweep}
#         cfg["relax_delay"] = 1.0  # no need for freq measurement

#         detune_params = sweep2param("detune", cfg["sweep"]["detune"])

#         prog = ModularProgramV2(
#             self.soccfg,
#             cfg,
#             modules=[
#                 Reset("reset", cfg.get("reset", {"type": "none"})),
#                 Pulse(
#                     name="qub_pulse",
#                     cfg={
#                         **cfg["qub_pulse"],
#                         "freq": cfg["qub_pulse"]["freq"] + detune_params,
#                     },
#                 ),
#                 Readout("readout", cfg["readout"]),
#             ],
#         )
#         return prog.acquire(
#             self.soc,
#             progress=False,
#             callback=wrap_earlystop_check(
#                 prog,
#                 update_hook,
#                 self.earlystop_snr,
#                 signal2real_fn=lambda x: rotate2real(x).real,
#             ),
#         )

#     def run(
#         self, ctx: TaskContext, need_infos: Dict[str, complex]
#     ) -> Dict[str, complex]:
#         self.task.run(ctx)

#         detunes = sweep2array(self.detune_sweep)
#         signals = ctx.get_current_data()

#         real_signals = rotate2real(signals).real
#         detune, freq_err, kappa, *_ = fit_qubit_freq(detunes, real_signals)
#         if freq_err > 0.3 * kappa:
#             fit_freq = np.nan
#         else:
#             fit_freq = detune + ctx.cfg["qub_pulse"]["freq"]

#         return {
#             "qubit_freq": fit_freq,
#             "qubit_detune": detune,
#             "qubit_linewidth": kappa,
#         }


def qubitfreq_fluxdep_signal2real(signals: np.ndarray) -> np.ndarray:
    real_signals = np.full_like(signals, np.nan, dtype=np.float64)

    for i in range(signals.shape[0]):
        if np.any(np.isnan(signals[i, :])):
            continue

        real_signals[i, :] = rotate2real(signals[i, :]).real

        # normalize
        max_val = np.max(real_signals[i, :])
        min_val = np.min(real_signals[i, :])
        real_signals[i, :] = (real_signals[i, :] - min_val) / (max_val - min_val)

    return real_signals


class QubitFreqMeasurement(AutoMeasurement):
    def __init__(self, detune_sweep: dict) -> None:
        self.detune_sweep = detune_sweep

        self.detunes = sweep2array(detune_sweep)

    def num_axes(self) -> Dict[str, int]:
        return {"detune": 2}

    def make_plotter(
        self, name: str, axs: Dict[str, List[plt.Axes]]
    ) -> Dict[str, LivePlotter2DwithLine]:
        return {
            "detune": LivePlotter2DwithLine(
                "Flux device value",
                "Detune (MHz)",
                line_axis=1,
                num_lines=5,
                title=name,
                segment2d_kwargs=dict(flip=True),
                existed_axes=[axs["detune"]],
            )
        }

    def generate_update_kwargs(
        self,
        plotters: Dict[str, AbsLivePlotter],
        flx_values: np.ndarray,
        signals: ResultType,
    ) -> Dict[str, tuple]:
        return {
            "detune": (flx_values, self.detunes, qubitfreq_fluxdep_signal2real(signals))
        }

    def writeback_cfg(self, i: int, ctx: TaskContext, flx_value: float) -> None: ...

    def make_task(self, soccfg, soc) -> HardTask:
        pass

    def save(
        self, filepath: str, result: ResultType, comment: Optional[str], prefix_tag: str
    ) -> None: ...
