from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine, MultiLivePlotter, make_plot_frame
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay
from zcu_tools.utils.process import rotate2real

from ...runner import BatchTask, Runner, SoftTask, TaskContext
from ...utils import set_pulse_freq
from .cell import FitLastFreqTask, MeasureDetuneTask, MeasureT1Task
from .util import check_gains

T1ResultType = Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]


def t1_signal2real(signals: np.ndarray) -> np.ndarray:
    real_signals = np.zeros_like(signals, dtype=np.float64)

    flx_len = signals.shape[0]
    for i in range(flx_len):
        real_signals[i, :] = rotate2real(signals[i : min(i + 1, flx_len), :]).real[0]

        if np.any(np.isnan(real_signals[i, :])):
            continue

        # flip to peak up
        max_val = np.max(real_signals[i, :])
        min_val = np.min(real_signals[i, :])
        avg_val = np.mean(real_signals[i, :])
        if max_val + min_val < 2 * avg_val:
            real_signals[i, :] = -real_signals[i, :]
            max_val, min_val = -min_val, -max_val

        # normalize
        real_signals[i, :] = (real_signals[i, :] - min_val) / (max_val - min_val)

    return real_signals


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
        len_sweep = cfg["sweep"]["length"]

        # delete sweep dicts, it will be added back later
        del cfg["sweep"]

        # predict sweep points
        values = sweep2array(flx_sweep)
        detunes = sweep2array(detune_sweep)
        lens = sweep2array(len_sweep)

        # get reference matrix element
        ref_m = predictor.predict_matrix_element(ref_flux, operator=drive_oper)
        ref_qub_gain = cfg["qub_pulse"]["gain"]
        ref_pi_gain = cfg["pi_pulse"]["gain"]

        def updateCfg(i: int, ctx: TaskContext, value: float) -> None:
            set_flux_in_dev_cfg(ctx.cfg["dev"], value)

            if i > 0:
                last_freq = ctx.get_data(addr_stack=[i - 1, "fit_freq"])
                if not np.isnan(last_freq):
                    bias = predictor.calculate_bias(values[i - 1], last_freq)
                    predictor.update_bias(bias)

            predict_freq = predictor.predict_freq(value)
            predict_m = predictor.predict_matrix_element(value, operator=drive_oper)

            set_pulse_freq(ctx.cfg["qub_pulse"], predict_freq)
            set_pulse_freq(ctx.cfg["pi_pulse"], predict_freq)
            ctx.cfg["qub_pulse"]["gain"] = check_gains(
                ref_qub_gain * ref_m / predict_m, "qub_pulse"
            )
            ctx.cfg["pi_pulse"]["gain"] = check_gains(
                ref_pi_gain * ref_m / predict_m, "pi_pulse"
            )

        # -- Define Measure Functions --

        def upwrap_signal(
            results: List[Dict[str, np.ndarray]],
        ) -> Dict[str, np.ndarray]:
            return {name: np.array([r[name] for r in results]) for name in results[0]}

        # -- Run Experiment --
        fig, axs, dh = make_plot_frame(n_row=2, n_col=2, figsize=(8, 7))

        with MultiLivePlotter(
            dict(
                detune=LivePlotter2DwithLine(
                    "Flux device value",
                    "Detune (MHz)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [axs[0]], dh),
                    disable=not progress,
                ),
                t1=LivePlotter2DwithLine(
                    "Flux device value",
                    "Time (us)",
                    line_axis=1,
                    num_lines=5,
                    existed_frames=(fig, [axs[1]], dh),
                    disable=not progress,
                ),
            )
        ) as viewer:

            def plot_fn(ctx: TaskContext) -> None:
                signals = upwrap_signal(ctx.get_data())
                plot_kwargs = dict(
                    detune=(values, detunes, t1_signal2real(signals["detune"])),
                    t1=(values, lens, t1_signal2real(signals["t1"])),
                )
                if not ctx.is_empty_stack():
                    cur_task = ctx.addr_stack[-1]
                    # only update current liveplotter for speed
                    if cur_task in ["detune", "t1"]:
                        plot_kwargs = {cur_task: plot_kwargs[cur_task]}
                    elif cur_task == "fit_freq":
                        return  # do nothing

                viewer.update(plot_kwargs)

            detune_ax = viewer.get_plotter("detune").get_ax("1d")
            t1_ax = viewer.get_plotter("t1").get_ax("1d")

            results = Runner(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=values,
                    update_cfg_fn=updateCfg,
                    sub_task=BatchTask(
                        tasks=dict(
                            detune=MeasureDetuneTask(
                                soccfg,
                                soc,
                                detune_sweep,
                                earlystop_snr=earlystop_snr,
                                snr_ax=detune_ax,
                            ),
                            fit_freq=FitLastFreqTask(
                                line_ax=detune_ax, detunes=detunes
                            ),
                            t1=MeasureT1Task(
                                soccfg,
                                soc,
                                len_sweep,
                                earlystop_snr=earlystop_snr,
                                snr_ax=t1_ax,
                            ),
                        ),
                    ),
                ),
                update_hook=plot_fn,
            ).run(cfg)
            signals_dict = upwrap_signal(results)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, detunes, lens, signals_dict)

        return values, detunes, lens, signals_dict, fig

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

        values, _, ts, signals_dict = result
        t1_signals = signals_dict["t1"]
        fit_freqs = signals_dict["fit_freq"]

        ts = ts[start_idx:]
        t1_signals = t1_signals[:, start_idx:]

        real_t1_signals = t1_signal2real(t1_signals)

        t1s = np.full(len(values), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(values)):
            real_signals = real_t1_signals[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signals)):
                continue

            t1, t1err, y_fit, *_ = fit_decay(ts, real_signals)

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

        values, detunes, ts, signals_dict = result
        detune_signals = signals_dict["detune"]
        fit_freqs = signals_dict["fit_freq"]
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
            filepath=filepath.with_name(filepath.name + "_t1"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Time", "unit": "s", "values": ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": t1_signals.T},
            comment=comment,
            tag=tag + "/t1",
            **kwargs,
        )
