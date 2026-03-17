from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Callable, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import make_ge_sweep, sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.liveplot import LivePlotter2D, MultiLivePlotter, make_plot_frame
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import batch_fit_func, fitlor, lorfunc
from zcu_tools.utils.process import rotate2real

# (res_freqs, qub_freqs, signals2D)
CKP_Result: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def ckp_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    amps = rotate2real(signals).real
    max_amp = np.nanmax(amps, axis=2, keepdims=True)
    min_amp = np.nanmin(amps, axis=2, keepdims=True)
    return (amps - min_amp) / np.clip(max_amp - min_amp, 1e-12, None)


def get_resonance_freq(
    xs: NDArray[np.float64], fpts: NDArray[np.float64], amps: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s_xs = []
    s_fpts = []

    prev_freq = np.nan
    for x, amp in zip(xs, amps):
        if np.any(np.isnan(amp)):
            continue

        param, _ = fitlor(fpts, amp)
        curr_freq = param[3]

        if abs(curr_freq - prev_freq) > 0.1 * (fpts[-1] - fpts[0]):
            continue

        prev_freq = curr_freq

        s_xs.append(x)
        s_fpts.append(curr_freq)

    return np.array(s_xs), np.array(s_fpts)


class CKP_ModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    res_pulse: PulseCfg
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class CKP_Cfg(ModularProgramCfg, TaskCfg):
    modules: CKP_ModuleCfg
    sweep: dict[str, SweepCfg]


class CKP_Exp(AbsExperiment[CKP_Result, CKP_Cfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> CKP_Result:
        _cfg = check_type(deepcopy(cfg), CKP_Cfg)
        modules = _cfg["modules"]

        if modules["res_pulse"].get("block_mode", True):
            raise ValueError("Resonator pulse must not in block mode")

        _cfg["sweep"] = {
            "ge": make_ge_sweep(),
            "res_freq": _cfg["sweep"]["res_freq"],
            "qub_freq": _cfg["sweep"]["qub_freq"],
        }

        # uniform in square space
        res_freqs = sweep2array(_cfg["sweep"]["res_freq"])
        qub_freqs = sweep2array(_cfg["sweep"]["qub_freq"])

        Pulse.set_param(
            modules["pi_pulse"], "on/off", sweep2param("ge", _cfg["sweep"]["ge"])
        )
        Pulse.set_param(
            modules["res_pulse"],
            "freq",
            sweep2param("res_freq", _cfg["sweep"]["res_freq"]),
        )
        Pulse.set_param(
            modules["qub_pulse"],
            "freq",
            sweep2param("qub_freq", _cfg["sweep"]["qub_freq"]),
        )

        fig, axs = make_plot_frame(1, 2, figsize=(10, 4))

        with MultiLivePlotter(
            fig,
            dict(
                ground=LivePlotter2D(
                    "Resonator Drive Frequency (MHz)",
                    "Qubit Probe Frequency (MHz)",
                    segment_kwargs=dict(title="Ground State"),
                    existed_axes=[[axs[0][0]]],
                ),
                excited=LivePlotter2D(
                    "Resonator Drive Frequency (MHz)",
                    "Qubit Probe Frequency (MHz)",
                    segment_kwargs=dict(title="Excited State"),
                    existed_axes=[[axs[0][1]]],
                ),
            ),
        ) as viewer:

            def plot_fn(ctx: TaskState):
                amps = ckp_signal2real(ctx.root_data)

                viewer.get_plotter("ground").update(
                    res_freqs, qub_freqs, amps[0], refresh=False
                )
                viewer.get_plotter("excited").update(
                    res_freqs, qub_freqs, amps[1], refresh=False
                )
                viewer.refresh()

            def measure_fn(ctx: TaskState, update_hook: Optional[Callable]):
                modules = ctx.cfg["modules"]
                return ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        Pulse("pi_pulse", modules["pi_pulse"]),
                        Pulse("res_pulse", modules["res_pulse"]),
                        Pulse("qub_pulse", modules["qub_pulse"]),
                        Readout("readout", modules["readout"]),
                    ],
                ).acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(2, len(res_freqs), len(qub_freqs)),
                ),
                init_cfg=_cfg,
                on_update=plot_fn,
            )
        plt.close(fig)

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (res_freqs, qub_freqs, signals)

        return res_freqs, qub_freqs, signals

    def analyze(
        self, result: Optional[CKP_Result] = None
    ) -> tuple[float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        res_freqs, qub_freqs, signals = result
        amps = ckp_signal2real(signals)

        g_res_freqs, g_qub_freqs = get_resonance_freq(res_freqs, qub_freqs, amps[0])
        e_res_freqs, e_qub_freqs = get_resonance_freq(res_freqs, qub_freqs, amps[1])

        # y0, slope, yscale, x0, gamma
        fixedparams: list[Optional[float]] = [None] * 5
        fixedparams[1] = 0.0  # fix slope to 0

        g_params, _ = fitlor(g_res_freqs, g_qub_freqs, fixedparams=fixedparams)
        e_params, _ = fitlor(e_res_freqs, e_qub_freqs, fixedparams=fixedparams)

        y0 = 0.5 * (e_params[0] + g_params[0])
        yscale = 0.5 * (e_params[2] + g_params[2])
        gamma = 0.5 * (e_params[4] + g_params[4])

        (g_params, e_params), (g_Cov, e_Cov) = batch_fit_func(
            [g_res_freqs, e_res_freqs],
            [g_qub_freqs, e_qub_freqs],
            lorfunc,
            list_init_p=[
                (y0, 0.0, yscale, g_params[3], gamma),
                (y0, 0.0, yscale, e_params[3], gamma),
            ],
            shared_idxs=[0, 2, 4],
            fixedparams=[None, 0.0, None, None, None],
        )

        g_freq = g_params[3]
        e_freq = e_params[3]
        chi = abs(e_freq - g_freq) / 2
        kappa = e_params[4] + g_params[4]

        res_freq = (g_freq + e_freq) / 2
        if kappa < 2 * chi:
            res_freq += np.sqrt(chi**2 - (kappa / 2) ** 2) * np.sign(e_freq - g_freq)

        kappa_err = np.sqrt(g_Cov[4, 4] + e_Cov[4, 4])
        chi_err = np.sqrt(g_Cov[3, 3] + e_Cov[3, 3]) / 2

        g_fit_freqs = lorfunc(res_freqs, *g_params)
        e_fit_freqs = lorfunc(res_freqs, *e_params)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        fig.suptitle(
            rf"$\chi/(2\pi): {chi:.3f}\pm {chi_err:.3f}\ MHz,\ \ \kappa/(2\pi): {kappa:.3f}\pm {kappa_err:.3f}\ MHz$",
            fontsize=14,
        )

        factor = 1.0
        if np.max(amps[0]) + np.min(amps[0]) < 2 * np.median(amps[0]):
            factor = -1.0

        ax1.imshow(
            factor * amps[0].T,
            extent=[res_freqs[0], res_freqs[-1], qub_freqs[0], qub_freqs[-1]],
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax1.scatter(g_res_freqs, g_qub_freqs, color="b", s=5)
        ax1.plot(res_freqs, g_fit_freqs, color="b", label="Ground")
        ax1.plot(res_freqs, e_fit_freqs, color="r", label="Excited", linestyle="--")
        ax1.axvline(g_freq, color="b")
        ax1.axvline(e_freq, color="r", linestyle="--")
        ax1.axvline(res_freq, color="k", linestyle=":")
        # ax1.set_xlabel("Resonator Drive Frequency (MHz)")
        ax1.set_ylabel("Qubit Probe Frequency (MHz)")
        ax1.legend()

        ax2.imshow(
            factor * amps[1].T,
            extent=[res_freqs[0], res_freqs[-1], qub_freqs[0], qub_freqs[-1]],
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax2.scatter(e_res_freqs, e_qub_freqs, color="r", s=5)
        ax2.plot(res_freqs, e_fit_freqs, color="r", label="Excited")
        ax2.plot(res_freqs, g_fit_freqs, color="b", label="Ground", linestyle="--")
        ax2.axvline(e_freq, color="r")
        ax2.axvline(g_freq, color="b", linestyle="--")
        ax2.axvline(res_freq, color="k", linestyle=":")
        ax2.set_xlabel("Resonator Drive Frequency (MHz)")
        ax2.set_ylabel("Qubit Probe Frequency (MHz)")
        ax2.legend()

        fig.tight_layout()

        return chi, kappa, res_freq, fig

    def save(
        self,
        filepath: str,
        result: Optional[CKP_Result] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ckp",
        **kwargs,
    ) -> None:
        """Save AC Stark experiment data."""
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        res_freqs, qub_freqs, signals = result

        _filepath = Path(filepath)

        # ground
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_ground")),
            x_info={
                "name": "Resonator Frequency",
                "unit": "Hz",
                "values": 1e6 * res_freqs,
            },
            y_info={"name": "Qubit Frequency", "unit": "Hz", "values": 1e6 * qub_freqs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals[0].T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
        # excited
        save_data(
            filepath=str(_filepath.with_name(_filepath.name + "_excited")),
            x_info={
                "name": "Resonator Frequency",
                "unit": "Hz",
                "values": 1e6 * res_freqs,
            },
            y_info={"name": "Qubit Frequency", "unit": "Hz", "values": 1e6 * qub_freqs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals[1].T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: list[str], **kwargs) -> CKP_Result:
        g_filepath, e_filepath = filepath

        g_signals, res_freqs, qub_freqs = load_data(g_filepath, **kwargs)
        assert qub_freqs is not None
        assert len(res_freqs.shape) == 1 and len(qub_freqs.shape) == 1
        assert g_signals.shape == (len(res_freqs), len(qub_freqs))

        e_signals, *_ = load_data(e_filepath, **kwargs)
        assert e_signals.shape == (len(res_freqs), len(qub_freqs))

        res_freqs = res_freqs * 1e-6  # Hz -> MHz
        qub_freqs = qub_freqs * 1e-6  # Hz -> MHz
        signals = np.stack([g_signals, e_signals], axis=0)

        res_freqs = res_freqs.astype(np.float64)
        qub_freqs = qub_freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (res_freqs, qub_freqs, signals)

        return res_freqs, qub_freqs, signals
