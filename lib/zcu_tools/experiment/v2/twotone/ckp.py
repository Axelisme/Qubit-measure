from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple, List
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array, make_ge_sweep
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter2D, MultiLivePlotter, make_plot_frame
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fitlor, lorfunc
from zcu_tools.utils.process import minus_background


# (res_freqs, qub_freqs, signals2D)
CKP_Result = Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]


def ckp_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=(1, 2)))


def get_resonance_freq(
    xs: NDArray[np.float64], fpts: NDArray[np.float64], amps: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
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


class CKP_Cfg(TaskConfig, ModularProgramCfg):
    pi_pulse: PulseCfg
    res_pulse: PulseCfg
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class CKP_Exp(AbsExperiment[CKP_Result, CKP_Cfg]):
    def run(self, soc, soccfg, cfg: CKP_Cfg) -> CKP_Result:
        cfg = deepcopy(cfg)  # prevent in-place modification

        if cfg["res_pulse"].get("block_mode", True):
            raise ValueError("Resonator pulse must not in block mode")

        assert "sweep" in cfg
        assert isinstance(cfg["sweep"], dict)
        cfg["sweep"] = {
            "ge": make_ge_sweep(),
            "res_freq": cfg["sweep"]["res_freq"],
            "qub_freq": cfg["sweep"]["qub_freq"],
        }

        # uniform in square space
        res_freqs = sweep2array(cfg["sweep"]["res_freq"])
        qub_freqs = sweep2array(cfg["sweep"]["qub_freq"])

        Pulse.set_param(
            cfg["pi_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )
        Pulse.set_param(
            cfg["res_pulse"], "freq", sweep2param("res_freq", cfg["sweep"]["res_freq"])
        )
        Pulse.set_param(
            cfg["qub_pulse"], "freq", sweep2param("qub_freq", cfg["sweep"]["qub_freq"])
        )

        fig, axs = make_plot_frame(2, 1, figsize=(12, 8))

        with MultiLivePlotter(
            fig,
            dict(
                ground=LivePlotter2D(
                    "Resonator Drive Frequency (MHz)",
                    "Qubit Probe Frequency (MHz)",
                    segment_kwargs=dict(title="Ground State"),
                    existed_axes=[axs[0]],
                ),
                excited=LivePlotter2D(
                    "Resonator Drive Frequency (MHz)",
                    "Qubit Probe Frequency (MHz)",
                    segment_kwargs=dict(title="Excited State"),
                    existed_axes=[axs[1]],
                ),
            ),
        ) as viewer:

            def plot_fn(ctx):
                amps = ckp_signal2real(ctx.data)

                viewer.get_plotter("ground").update(
                    res_freqs, qub_freqs, amps[0], refresh=False
                )
                viewer.get_plotter("excited").update(
                    res_freqs, qub_freqs, amps[1], refresh=False
                )
                viewer.refresh()

            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset(
                                    "reset",
                                    ctx.cfg.get("reset", {"type": "none"}),
                                ),
                                Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                Pulse("res_pulse", ctx.cfg["res_pulse"]),
                                Pulse("qub_pulse", ctx.cfg["qub_pulse"]),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, update_hook=update_hook)
                    ),
                    result_shape=(2, len(res_freqs), len(qub_freqs)),
                ),
                init_cfg=cfg,
                update_hook=plot_fn,
            )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (res_freqs, qub_freqs, signals)

        return res_freqs, qub_freqs, signals

    def analyze(
        self,
        *,
        res_freq: float,
        res_gain: float,
        result: Optional[CKP_Result] = None,
        kappa: Optional[float] = None,
    ) -> Tuple[float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "No result found"

        res_freqs, qub_freqs, signals = result
        amps = ckp_signal2real(signals)

        g_res_freqs, g_qub_freqs = get_resonance_freq(res_freqs, qub_freqs, amps[0])
        e_res_freqs, e_qub_freqs = get_resonance_freq(res_freqs, qub_freqs, amps[1])

        # y0, slope, yscale, x0, gamma
        fixedparams: List[Optional[float]] = [None] * 5
        fixedparams[1] = 0.0  # fix slope to 0
        if kappa is not None:
            fixedparams[4] = kappa / 2  # kappa = 2 * gamma

        g_params, _ = fitlor(g_res_freqs, g_qub_freqs, fixedparams=fixedparams)
        e_params, _ = fitlor(e_res_freqs, e_qub_freqs, fixedparams=fixedparams)

        g_freq = g_params[3]
        e_freq = e_params[3]
        chi = abs(e_freq - g_freq)
        kappa = e_params[4] + g_params[4]

        g_delta = lorfunc(np.array(res_freq), *g_params).item() - g_params[0]
        e_delta = lorfunc(np.array(res_freq), *e_params).item() - e_params[0]
        delta_slope = (e_delta - g_delta) / res_gain**2

        # Calculate the Stark shift
        eta = kappa**2 / (kappa**2 + chi**2)
        ac_coeff = abs(delta_slope) / (2 * eta * chi)

        g_fit_freqs = lorfunc(res_freqs, *g_params)
        e_fit_freqs = lorfunc(res_freqs, *e_params)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

        ax1.imshow(
            amps[0].T,
            extent=[res_freqs[0], res_freqs[-1], qub_freqs[0], qub_freqs[-1]],
            aspect="auto",
            origin="lower",
            interpolation="none",
        )
        ax1.scatter(g_res_freqs, g_qub_freqs, color="b", s=2)
        ax1.plot(res_freqs, g_fit_freqs, color="b", label="Ground")
        ax1.plot(res_freqs, e_fit_freqs, color="r", label="Excited", linestyle="--")
        ax1.axvline(g_freq, color="b")
        ax1.axvline(e_freq, color="r", linestyle="--")
        ax1.axvline(res_freq, color="k", linestyle=":")
        ax1.set_xlabel("Resonator Drive Frequency (MHz)")
        ax1.set_ylabel("Qubit Probe Frequency (MHz)")

        ax2.imshow(
            amps[1].T,
            extent=[res_freqs[0], res_freqs[-1], qub_freqs[0], qub_freqs[-1]],
            aspect="auto",
            origin="lower",
            interpolation="none",
        )
        ax2.scatter(e_res_freqs, e_qub_freqs, color="r", s=2)
        ax2.plot(res_freqs, e_fit_freqs, color="r", label="Excited")
        ax2.plot(res_freqs, g_fit_freqs, color="b", label="Ground", linestyle="--")
        ax2.axvline(e_freq, color="r")
        ax2.axvline(g_freq, color="b", linestyle="--")
        ax2.axvline(res_freq, color="k", linestyle=":")
        ax2.set_xlabel("Resonator Drive Frequency (MHz)")
        # ax2.set_ylabel("Qubit Probe Frequency (MHz)")

        fig.tight_layout()

        return chi, kappa, ac_coeff, fig

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
            tag=tag + "_g",
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
            tag=tag + "_e",
            **kwargs,
        )

    def load(self, filepath: List[str], **kwargs) -> CKP_Result:
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
