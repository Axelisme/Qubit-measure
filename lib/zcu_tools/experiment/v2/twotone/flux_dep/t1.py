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
from zcu_tools.utils.fitting import fit_decay

from .task import (
    MeasureDetuneTask,
    MeasureLenRabiTask,
    MeasureT1Task,
    detune_signal2real,
    lenrabi_signal2real,
    t1_signal2real,
)
from .util import check_gains

T1ResultType = Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]


class T1Experiment(AbsExperiment[T1ResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        predictor: FluxoniumPredictor,
        ref_flux: float = 0.0,
        drive_oper: Literal["n", "phi"] = "n",
        progress: bool = True,
        earlystop_snr: Optional[float] = None,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_sweep = cfg["sweep"]["flux"]
        detune_sweep = cfg["sweep"]["detune"]
        rabilen_sweep = cfg["sweep"]["rabi_length"]
        t1len_sweep = cfg["sweep"]["t1_length"]
        del cfg["sweep"]  # delete sweep dicts, it will be added back later

        # predict sweep points
        values = sweep2array(flx_sweep)
        detunes = sweep2array(detune_sweep)
        rabilens = sweep2array(rabilen_sweep)
        t1lens = sweep2array(t1len_sweep)

        # get reference matrix element
        ref_m = predictor.predict_matrix_element(ref_flux, operator=drive_oper)
        ref_qub_gain = cfg["qub_pulse"]["gain"]
        ref_pi_gain = cfg["pi_pulse"]["gain"]

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

            # set pulse freq to predicted frequency
            set_pulse_freq(ctx.cfg["qub_pulse"], predict_freq)
            set_pulse_freq(ctx.cfg["pi_pulse"], predict_freq)
            ctx.cfg["qub_pulse"]["gain"] = check_gains(
                ref_qub_gain * ref_m / predict_m, "qub_pulse"
            )
            ctx.cfg["pi_pulse"]["gain"] = check_gains(
                ref_pi_gain * ref_m / predict_m, "pi_pulse"
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
                len_rabi=LivePlotter2DwithLine(
                    "Flux device value",
                    "Rabi length (us)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [[axs[1, 1], axs[0, 1]]]),
                    disable=not progress,
                ),
                t1=LivePlotter2DwithLine(
                    "Flux device value",
                    "Time (us)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [[axs[1, 2], axs[0, 2]]]),
                    disable=not progress,
                ),
            ),
        ) as viewer:
            plot_map = {
                "detune": (detunes, detune_signal2real),
                "len_rabi": (rabilens, lenrabi_signal2real),
                "t1": (t1lens, t1_signal2real),
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
                            t1=MeasureT1Task(soccfg, soc, t1len_sweep, earlystop_snr),
                        ),
                    ),
                ),
                update_hook=plot_fn,
            ).run(cfg)
            signals_dict = merge_result_list(results)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, detunes, rabilens, t1lens, signals_dict)

        return values, detunes, rabilens, t1lens, signals_dict, fig

    def analyze(
        self,
        result: Optional[T1ResultType] = None,
        *,
        start_idx: int = 0,
        snr_threshold: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, _, _, t1lens, signals_dict = result
        fit_freqs = signals_dict["meta_infos"]["detune"]["qubit_freq"]
        t1_signals = signals_dict["t1"]

        t1lens = t1lens[start_idx:]
        t1_signals = t1_signals[:, start_idx:]

        real_t1_signals = t1_signal2real(t1_signals)

        t1s = np.full(len(values), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(values)):
            real_signals = real_t1_signals[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signals)):
                continue

            t1, t1err, y_fit, *_ = fit_decay(t1lens, real_signals)

            if t1err > 0.3 * t1:
                continue

            contrast = np.max(y_fit) - np.min(y_fit) + 1e-9
            snr = contrast / np.mean(np.abs(real_signals - y_fit))
            if snr < snr_threshold:
                continue

            t1s[i] = t1
            t1errs[i] = t1err

        if np.all(np.isnan(t1s)):
            raise ValueError("No valid Fitting T1 found. Please check the data.")

        valid_idxs = ~np.isnan(t1s)
        valid_values = values[valid_idxs]
        valid_freqs = fit_freqs[valid_idxs]
        t1s = t1s[valid_idxs]
        t1errs = t1errs[valid_idxs]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)
        fig.suptitle("T1 over Flux")
        ax1.imshow(
            real_t1_signals.T,
            aspect="auto",
            extent=[values[0], values[-1], t1s[0], t1s[-1]],
            origin="lower",
            interpolation="none",
        )
        ax2.errorbar(
            valid_values, t1s, yerr=t1errs, label="Fitting T1", elinewidth=1, capsize=1
        )
        ax2.set_xlabel("Flux value (a.u.)")
        ax2.set_ylabel("T1 (us)")
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(values[0], values[-1])
        ax2.grid()
        plt.plot()

        return valid_values, t1s, t1errs, valid_freqs

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/ge/batch",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, detunes, rabilens, t1lens, signals_dict = result
        detune_signals = signals_dict["detune"]
        fit_freqs = signals_dict["meta_infos"]["detune"]["qubit_freq"]
        rabilen_signals = signals_dict["len_rabi"]
        t1_signals = signals_dict["t1"]

        filepath = Path(filepath)

        save_data(
            filepath=filepath.with_name(filepath.name + "_detune"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Detune", "unit": "Hz", "values": detunes * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": detune_signals.T},
            comment=comment,
            tag=tag + "/detune",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_fit_freq"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            z_info={"name": "Frequency", "unit": "Hz", "values": fit_freqs * 1e6},
            comment=comment,
            tag=tag + "/fit_freq",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_len_rabi"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Length", "unit": "s", "values": rabilens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": rabilen_signals.T},
            comment=comment,
            tag=tag + "/len_rabi",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_t1"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Time", "unit": "s", "values": t1lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": t1_signals.T},
            comment=comment,
            tag=tag + "/t1",
            **kwargs,
        )
