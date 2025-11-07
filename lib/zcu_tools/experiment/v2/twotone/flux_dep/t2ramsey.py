from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import AutoBatchTask, Runner, SoftTask, TaskContext
from zcu_tools.experiment.v2.utils import merge_result_list, set_pulse_freq
from zcu_tools.liveplot import LivePlotter2DwithLine, MultiLivePlotter, make_plot_frame
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay_fringe

from .task import (
    MeasureDetuneTask,
    MeasureLenRabiTask,
    MeasureT2RamseyTask,
    detune_signal2real,
    lenrabi_signal2real,
    t2ramsey_signal2real,
)
from .util import check_gains

T2RamseyResultType = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]
]


class T2RamseyExperiment(AbsExperiment[T2RamseyResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        predictor: FluxoniumPredictor,
        activate_detune: float = 0.0,
        ref_flux: float = 0.0,
        drive_oper: Literal["n", "phi"] = "n",
        progress: bool = True,
        earlystop_snr: Optional[float] = None,
    ) -> T2RamseyResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_sweep = cfg["sweep"]["flux"]
        detune_sweep = cfg["sweep"]["detune"]
        rabilen_sweep = cfg["sweep"]["rabi_length"]
        t2rlen_sweep = cfg["sweep"]["t2r_length"]
        del cfg["sweep"]  # delete sweep dicts, it will be added back later

        # predict sweep points
        values = sweep2array(flx_sweep)
        detunes = sweep2array(detune_sweep)
        rabilens = sweep2array(rabilen_sweep)
        t2rlens = sweep2array(t2rlen_sweep)

        # get reference matrix element
        ref_m = predictor.predict_matrix_element(ref_flux, operator=drive_oper)
        ref_qub_gain = cfg["qub_pulse"]["gain"]
        ref_pi2_gain = cfg["pi2_pulse"]["gain"]

        def updateCfg(i: int, ctx: TaskContext, value: float) -> None:
            set_flux_in_dev_cfg(ctx.cfg["dev"], value)

            # calibrate bias by last fitted frequency
            if i > 0:
                last_freq = ctx.get_data(
                    addr_stack=[i - 1, "meta_infos", "detune", "qubit_freq"]
                )
                if not np.isnan(last_freq):
                    bias = predictor.calculate_bias(values[i - 1], last_freq)
                    predictor.update_bias(bias)

            predict_freq = predictor.predict_freq(value)
            predict_m = predictor.predict_matrix_element(value, operator=drive_oper)

            set_pulse_freq(ctx.cfg["qub_pulse"], predict_freq)
            set_pulse_freq(ctx.cfg["pi2_pulse"], predict_freq)
            ctx.cfg["qub_pulse"]["gain"] = check_gains(
                ref_qub_gain * ref_m / predict_m, "qub_pulse"
            )
            ctx.cfg["pi2_pulse"]["gain"] = check_gains(
                ref_pi2_gain * ref_m / predict_m, "pi2_pulse"
            )

        # -- Run Experiment --
        fig, axs = make_plot_frame(n_row=2, n_col=3, figsize=(15, 7))

        with MultiLivePlotter(
            fig,
            dict(
                detune=LivePlotter2DwithLine(
                    "Flux device value",
                    "Detune (MHz)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [[axs[1, 0], axs[0, 0]]]),
                    disable=not progress,
                ),
                t2ramsey=LivePlotter2DwithLine(
                    "Flux device value",
                    "Time (us)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [[axs[1, 1], axs[0, 1]]]),
                    disable=not progress,
                ),
            ),
        ) as viewer:
            plot_map = {
                "detune": (detunes, detune_signal2real),
                "len_rabi": (rabilens, lenrabi_signal2real),
                "t2ramsey": (t2rlens, t2ramsey_signal2real),
            }

            def plot_fn(ctx: TaskContext) -> None:
                if ctx.is_empty_stack():
                    return

                cur_task = ctx.addr_stack[-1]
                if cur_task not in plot_map:
                    return

                signals = merge_result_list(ctx.get_data())

                ys, signal2real_fn = plot_map[cur_task]
                viewer.update(
                    {cur_task: (values, ys, signal2real_fn(signals[cur_task]))}
                )

            results = Runner(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=values,
                    update_cfg_fn=updateCfg,
                    sub_task=AutoBatchTask(
                        tasks=dict(
                            detune=MeasureDetuneTask(
                                soccfg, soc, detune_sweep, earlystop_snr
                            ),
                            len_rabi=MeasureLenRabiTask(
                                soccfg, soc, rabilen_sweep, earlystop_snr
                            ),
                            t2ramsey=MeasureT2RamseyTask(
                                soccfg,
                                soc,
                                t2rlen_sweep,
                                activate_detune=activate_detune,
                                earlystop_snr=earlystop_snr,
                            ),
                        ),
                    ),
                ),
                update_hook=plot_fn,
            ).run(cfg)
            signals_dict = merge_result_list(results)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, detunes, rabilens, t2rlens, signals_dict)

        return values, detunes, rabilens, t2rlens, signals_dict, fig

    def analyze(
        self,
        result: Optional[T2RamseyResultType] = None,
        activate_detune: float = 0.0,
        snr_threshold: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, detunes, _, t2rlens, signals_dict = result
        fit_freqs = signals_dict["meta_infos"]["detune"]["fit_freq"]
        t2r_signals = signals_dict["t2ramsey"]

        real_t2r_signals = t2ramsey_signal2real(t2r_signals)

        t2s = np.full(len(values), np.nan, dtype=np.float64)
        t2errs = np.zeros_like(t2s)
        true_freqs = np.full_like(t2s, np.nan)
        true_freq_errs = np.zeros_like(t2s)

        for i in range(len(values)):
            real_signals = real_t2r_signals[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signals)):
                continue

            t2r, t2err, detune, derr, y_fit, *_ = fit_decay_fringe(
                t2rlens, real_signals
            )

            if t2err > 0.3 * t2r:
                continue

            contrast = np.max(y_fit) - np.min(y_fit) + 1e-9
            snr = contrast / np.mean(np.abs(real_signals - y_fit))
            if snr < snr_threshold:
                continue

            t2s[i] = t2r
            t2errs[i] = t2err
            true_freqs[i] = activate_detune - detune + fit_freqs[i]
            true_freq_errs[i] = derr

        if np.all(np.isnan(t2s)):
            raise ValueError("No valid Fitting T2 found. Please check the data.")

        valid_idxs = ~np.isnan(t2s)
        values = values[valid_idxs]
        t2s = t2s[valid_idxs]
        t2errs = t2errs[valid_idxs]
        true_freqs = true_freqs[valid_idxs]
        true_freq_errs = true_freq_errs[valid_idxs]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)
        fig.suptitle("T2Ramsey over Flux")
        ax1.errorbar(
            values, t2s, yerr=t2errs, label="Fitting T2", elinewidth=1, capsize=1
        )
        ax1.set_ylabel("T2 (us)")
        ax1.set_ylim(bottom=0)
        ax1.grid()
        ax2.errorbar(
            values,
            true_freqs,
            yerr=true_freq_errs,
            label="Fitting frequency",
            elinewidth=1,
            capsize=1,
        )
        ax2.set_ylabel("Detune (MHz)")
        ax2.set_xlabel("Flux value (a.u.)")
        ax2.grid()
        plt.plot()

        return values, t2s, t2errs, true_freqs, true_freq_errs

    def save(
        self,
        filepath: str,
        result: Optional[T2RamseyResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/ge/batch",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, detunes, rabilens, t2rlens, signals_dict = result
        detune_singals = signals_dict["detune"]
        fit_freqs = signals_dict["meta_infos"]["detune"]["fit_freq"]
        rabilen_signals = signals_dict["len_rabi"]
        t2r_signals = signals_dict["t2ramsey"]

        filepath = Path(filepath)

        save_data(
            filepath=filepath.with_stem(filepath.stem + "_detune"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Detune", "unit": "Hz", "values": detunes * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": detune_singals.T},
            comment=comment,
            tag=tag + "/detune",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_stem(filepath.stem + "_fit_freq"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            z_info={"name": "Frequency", "unit": "Hz", "values": fit_freqs * 1e6},
            comment=comment,
            tag=tag + "/fit_freq",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_stem(filepath.stem + "_len_rabi"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Length", "unit": "s", "values": rabilens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": rabilen_signals.T},
            comment=comment,
            tag=tag + "/len_rabi",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_stem(filepath.stem + "_t2ramsey"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Time", "unit": "s", "values": t2rlens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": t2r_signals.T},
            comment=comment,
            tag=tag + "/t2ramsey",
            **kwargs,
        )
