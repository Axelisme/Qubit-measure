from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import (
    AutoBatchTask,
    HardTask,
    Runner,
    SoftTask,
    TaskContext,
)
from zcu_tools.experiment.v2.utils import merge_result_list, set_pulse_freq
from zcu_tools.liveplot import LivePlotter2DwithLine, MultiLivePlotter, make_plot_frame
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.simulate import mA2flx
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor
from zcu_tools.utils.datasaver import save_data

from .task import (
    MeasureDetuneTask,
    MeasureLenRabiTask,
    MeasureMistTask,
    automist_signal2real,
    detune_signal2real,
    lenrabi_signal2real,
)
from .util import check_gains

AutoMistResultType = Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]


class AutoMistExperiment(AbsExperiment[AutoMistResultType]):
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
    ) -> AutoMistResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_sweep = cfg["sweep"]["flux"]
        detune_sweep = cfg["sweep"]["detune"]
        rabilen_sweep = cfg["sweep"]["rabi_length"]
        pdr_sweep = cfg["sweep"]["gain"]
        del cfg["sweep"]  # delete sweep dicts, it will be added back later

        # predict sweep points
        values = sweep2array(flx_sweep)
        detunes = sweep2array(detune_sweep)
        rabilens = sweep2array(rabilen_sweep)
        gains = sweep2array(pdr_sweep)

        # get reference matrix element
        ref_m = predictor.predict_matrix_element(ref_flux, operator=drive_oper)
        ref_qub_gain = cfg["qub_pulse"]["gain"]
        ref_pi_gain = cfg["pi_pulse"]["gain"]
        ref_pilen = cfg["pi_pulse"]["waveform"]["length"]

        def updateCfg(i: int, ctx: TaskContext, value: float) -> None:
            set_flux_in_dev_cfg(ctx.cfg["dev"], value)

            adjust_factor = 1.0
            if i > 0:
                last_freq = ctx.get_data(
                    addr_stack=[i - 1, "meta_infos", "detune", "qubit_freq"]
                )
                if not np.isnan(last_freq):
                    bias = predictor.calculate_bias(values[i - 1], last_freq)
                    predictor.update_bias(bias)
                last_pilen = ctx.get_data(
                    addr_stack=[i - 1, "meta_infos", "len_rabi", "pi_length"]
                )
                if not np.isnan(last_pilen):
                    adjust_factor = last_pilen / ref_pilen

            predict_freq = predictor.predict_freq(value)
            predict_m = predictor.predict_matrix_element(value, operator=drive_oper)

            set_pulse_freq(ctx.cfg["qub_pulse"], predict_freq)
            set_pulse_freq(ctx.cfg["pi_pulse"], predict_freq)
            ctx.cfg["qub_pulse"]["gain"] = check_gains(
                adjust_factor * ref_qub_gain * ref_m / predict_m, "qub_pulse"
            )
            ctx.cfg["pi_pulse"]["gain"] = check_gains(
                adjust_factor * ref_pi_gain * ref_m / predict_m, "pi_pulse"
            )

        # -- Run Experiment --
        fig, axs = make_plot_frame(n_row=2, n_col=4, figsize=(18, 7))

        with MultiLivePlotter(
            fig,
            dict(
                detune=LivePlotter2DwithLine(
                    "Flux device value",
                    "Detune (MHz)",
                    line_axis=1,
                    num_lines=5,
                    title="Detune",
                    segment2d_kwargs=dict(flip=True),
                    existed_frames=(fig, [[axs[0, 0], axs[1, 0]]]),
                    disable=not progress,
                ),
                len_rabi=LivePlotter2DwithLine(
                    "",
                    "Length (us)",
                    line_axis=1,
                    num_lines=5,
                    title="Length Rabi",
                    segment2d_kwargs=dict(flip=True),
                    existed_frames=(fig, [[axs[0, 1], axs[1, 1]]]),
                    disable=not progress,
                ),
                mist_g=LivePlotter2DwithLine(
                    "",
                    "Readout power (a.u.)",
                    line_axis=1,
                    num_lines=5,
                    title="MIST (Ground)",
                    segment2d_kwargs=dict(flip=True),
                    existed_frames=(fig, [[axs[0, 2], axs[1, 2]]]),
                    disable=not progress,
                ),
                mist_e=LivePlotter2DwithLine(
                    "",
                    "Readout power (a.u.)",
                    line_axis=1,
                    num_lines=5,
                    title="MIST (Excited)",
                    segment2d_kwargs=dict(flip=True),
                    existed_frames=(fig, [[axs[0, 3], axs[1, 3]]]),
                    disable=not progress,
                ),
            ),
        ) as viewer:
            detune_line = axs[1, 0].axvline(0.0, color="red", linestyle="--")
            rabi_line = axs[1, 1].axvline(0.0, color="red", linestyle="--")

            def plot_fn(ctx: TaskContext) -> None:
                if ctx.is_empty_stack():
                    return

                cur_task = ctx.addr_stack[-1]

                meta_infos = ctx.get_data(
                    addr_stack=[*ctx.addr_stack[:-1], "meta_infos"]
                )
                if "detune" in meta_infos:
                    detune_line.set_xdata(meta_infos["detune"]["qubit_detune"].real)
                if "len_rabi" in meta_infos:
                    rabi_line.set_xdata(meta_infos["len_rabi"]["pi_length"].real)

                signals = merge_result_list(ctx.get_data())[cur_task]
                if cur_task == "detune":
                    viewer.update(
                        {"detune": (values, detunes, detune_signal2real(signals))}
                    )
                elif cur_task == "len_rabi":
                    viewer.update(
                        {"len_rabi": (values, rabilens, lenrabi_signal2real(signals))}
                    )
                elif cur_task == "mist":
                    mist_signals = automist_signal2real(signals)
                    viewer.update(
                        {
                            "mist_g": (values, gains, mist_signals[..., 0]),
                            "mist_e": (values, gains, mist_signals[..., 1]),
                        }
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
                                soccfg, soc, rabilen_sweep, earlystop_snr=earlystop_snr
                            ),
                            mist=MeasureMistTask(soccfg, soc, pdr_sweep),
                        ),
                    ),
                ),
                update_hook=plot_fn,
            ).run(cfg)
            signals_dict = merge_result_list(results)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, detunes, rabilens, gains, signals_dict)

        return values, detunes, gains, signals_dict, fig

    def analyze(
        self,
        result: Optional[AutoMistResultType] = None,
        *,
        start_idx: int = 0,
        snr_threshold: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, detunes, rabilens, gains, signals_dict = result
        mist_signals: np.ndarray = signals_dict["mist"]

        mist_signals = automist_signal2real(mist_signals)
        g_real_signals = mist_signals[..., 0]
        e_real_signals = mist_signals[..., 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figsize)
        ax1.imshow(
            g_real_signals.T,
            origin="lower",
            interpolation="none",
            aspect="auto",
            extent=[
                values[0],
                values[-1],
                gains[0],
                gains[-1],
            ],
        )
        ax1.set_title("MIST Ground State")
        ax1.set_xlabel("Flux value (a.u.)")
        ax1.set_ylabel("Readout power (a.u.)")

        ax2.imshow(
            e_real_signals.T,
            origin="lower",
            interpolation="none",
            aspect="auto",
            extent=[
                values[0],
                values[-1],
                gains[0],
                gains[-1],
            ],
        )
        ax2.set_title("MIST Excited State")
        ax2.set_xlabel("Flux value (a.u.)")
        ax2.set_ylabel("Readout power (a.u.)")

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[AutoMistResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/mist/batch",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, detunes, lens, gains, signals_dict = result
        detune_signals = signals_dict["detune"]
        fit_freqs = signals_dict["meta_infos"]["detune"]["qubit_freq"]
        rabilen_signals = signals_dict["len_rabi"]
        mist_signals = signals_dict["mist"]

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
            y_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": rabilen_signals.T},
            comment=comment,
            tag=tag + "/len_rabi",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_mist_g"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Power", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": mist_signals[..., 0].T},
            comment=comment,
            tag=tag + "/mist_g",
            **kwargs,
        )

        save_data(
            filepath=filepath.with_name(filepath.name + "_mist_e"),
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Power", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": mist_signals[..., 1].T},
            comment=comment,
            tag=tag + "/mist_e",
            **kwargs,
        )


MistResultType = Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]


def mist_signal2real(signals: np.ndarray) -> np.ndarray:
    avg_len = max(int(0.05 * signals.shape[1]), 1)
    std_len = max(int(0.3 * signals.shape[1]), 5)

    mist_signals = np.abs(
        signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)
    )
    if np.all(np.isnan(mist_signals)):
        return mist_signals
    mist_signals = np.clip(mist_signals, 0, 5 * np.nanstd(mist_signals[:, :std_len]))

    return mist_signals


class MistExperiment(AbsExperiment[MistResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> MistResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_sweep = cfg["sweep"]["flux"]
        cfg["sweep"] = {"gain": cfg["sweep"]["gain"]}

        # predict sweep points
        values = sweep2array(flx_sweep)
        gains = sweep2array(cfg["sweep"]["gain"])

        def updateCfg(i: int, ctx: TaskContext, value: float) -> None:
            set_flux_in_dev_cfg(ctx.cfg["dev"], value)

        Pulse.set_param(
            cfg["probe_pulse"], "gain", sweep2param("gain", cfg["sweep"]["gain"])
        )

        with LivePlotter2DwithLine(
            "Flux device value",
            "Readout power (a.u.)",
            line_axis=1,
            num_lines=5,
            title="MIST over FLux",
            disable=not progress,
        ) as viewer:
            results = Runner(
                task=SoftTask(
                    sweep_name="flux",
                    sweep_values=values,
                    update_cfg_fn=updateCfg,
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    make_reset("reset", reset_cfg=ctx.cfg.get("reset")),
                                    Pulse(
                                        name="init_pulse", cfg=ctx.cfg.get("init_pulse")
                                    ),
                                    Pulse(
                                        name="probe_pulse", cfg=ctx.cfg["probe_pulse"]
                                    ),
                                    make_readout(
                                        "readout", readout_cfg=ctx.cfg["readout"]
                                    ),
                                ],
                            ).acquire(soc, progress=False, callback=update_hook)
                        ),
                        result_shape=(len(gains),),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    values, gains, mist_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(results)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (values, gains, signals)

        return values, gains, signals

    def analyze(
        self,
        result: Optional[MistResultType] = None,
        *,
        mA_c: Optional[float] = None,
        period: Optional[float] = None,
        ac_coeff: Optional[float] = None,
        fig: Optional[go.Figure] = None,
    ) -> go.Figure:
        if result is None:
            result = self.last_result

        dev_values, pdrs, signals = result

        if mA_c is not None and period is not None:
            xs = mA2flx(dev_values, mA_c, period)
        else:
            xs = dev_values

        amp_diff = mist_signal2real(signals)

        if fig is None:
            fig = go.Figure()

        if mA_c is not None and period is not None:
            xlabel = r"$\phi$ (a.u.)"
        else:
            xlabel = r"$A$ (mA)"
        fig.update_xaxes(title_text=xlabel, title_font_size=14, range=[xs[0], xs[-1]])

        if ac_coeff is None:
            ys = pdrs
            ylabel = "probe gain (a.u.)"
        else:
            ys = ac_coeff * pdrs**2
            ylabel = r"$\bar n$"
        fig.update_yaxes(title_text=ylabel, title_font_size=12, range=[ys[0], ys[-1]])

        fig.add_trace(
            go.Heatmap(x=xs, y=ys, z=amp_diff.T, showscale=False, colorscale="Greys")
        )

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[MistResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/mist",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, gains, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Flux value", "unit": "a.u.", "values": values},
            y_info={"name": "Power", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
