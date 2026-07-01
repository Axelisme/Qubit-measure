from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D, MultiLivePlot, make_plot_frame
from zcu_tools.program.v2 import (
    Branch,
    Join,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import batch_fit_func, fitlor, lorfunc
from zcu_tools.utils.process import rotate2real


def _default_initial_states() -> NDArray[np.int64]:
    return np.array([0, 1], dtype=np.int64)


@dataclass(frozen=True)
class CKP_Result:
    res_freqs: NDArray[np.float64]
    qub_freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    initial_states: NDArray[np.int64] = field(default_factory=_default_initial_states)
    cfg_snapshot: CKP_Cfg | None = None


def ckp_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    amps = rotate2real(signals).real
    max_amp = np.nanmax(amps, axis=2, keepdims=True)
    min_amp = np.nanmin(amps, axis=2, keepdims=True)
    return (amps - min_amp) / np.clip(max_amp - min_amp, 1e-12, None)


def get_resonance_freq(
    xs: NDArray[np.float64], freqs: NDArray[np.float64], amps: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s_xs = []
    s_freqs = []

    prev_freq = np.nan
    for x, amp in zip(xs, amps):
        if np.any(np.isnan(amp)):
            continue

        param, _ = fitlor(freqs, amp)
        curr_freq = param[3]

        if abs(curr_freq - prev_freq) > 0.1 * (freqs[-1] - freqs[0]):
            continue

        prev_freq = curr_freq

        s_xs.append(x)
        s_freqs.append(curr_freq)

    return np.array(s_xs), np.array(s_freqs)


class CKPModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi_pulse: PulseCfg
    res_pulse: PulseCfg
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class CKPSweepCfg(ConfigBase):
    res_freq: SweepCfg
    qub_freq: SweepCfg


class CKP_Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: CKPModuleCfg
    sweep: CKPSweepCfg


class CKP_Exp(PersistableExperiment[CKP_Result, CKP_Cfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "qub_freqs", "Qubit Frequency", "Hz", scale=MHZ_TO_HZ, dtype=np.float64
            ),
            Axis(
                "res_freqs",
                "Resonator Frequency",
                "Hz",
                scale=MHZ_TO_HZ,
                dtype=np.float64,
            ),
            Axis(
                "initial_states",
                "Initial State",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=CKP_Result,
        cfg_type=CKP_Cfg,
        tag="twotone/ge/ckp",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: CKP_Cfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> CKP_Result:
        orig_cfg = deepcopy(cfg)

        setup_devices(cfg, progress=True)
        modules = cfg.modules

        res_freq_sweep = cfg.sweep.res_freq
        qub_freq_sweep = cfg.sweep.qub_freq

        res_freqs = sweep2array(
            res_freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.res_pulse.ch},
        )
        qub_freqs = sweep2array(
            qub_freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        fig, axs = make_plot_frame(1, 2, plot_instant=True, figsize=(10, 4))

        with MultiLivePlot(
            fig,
            dict(
                ground=LivePlot2D(
                    "Resonator Drive Frequency (MHz)",
                    "Qubit Probe Frequency (MHz)",
                    segment_kwargs=dict(title="Ground State"),
                    existed_axes=[[axs[0][0]]],
                ),
                excited=LivePlot2D(
                    "Resonator Drive Frequency (MHz)",
                    "Qubit Probe Frequency (MHz)",
                    segment_kwargs=dict(title="Excited State"),
                    existed_axes=[[axs[0][1]]],
                ),
            ),
        ) as viewer:

            def plot_fn(data: NDArray[np.complex128]) -> None:
                real_signal = ckp_signal2real(data)

                viewer.get_plotter("ground").update(
                    res_freqs, qub_freqs, real_signal[0], refresh=False
                )
                viewer.get_plotter("excited").update(
                    res_freqs, qub_freqs, real_signal[1], refresh=False
                )
                viewer.refresh()

            signals_buffer = SignalBuffer(
                (2, len(res_freqs), len(qub_freqs)),
                on_update=plot_fn,
            )
            with Schedule(cfg, signals_buffer) as sched:
                modules = sched.cfg.modules
                res_freq_param = sweep2param("res_freq", sched.cfg.sweep.res_freq)
                qub_freq_param = sweep2param("qub_freq", sched.cfg.sweep.qub_freq)
                modules.res_pulse.set_param("freq", res_freq_param)
                modules.qub_pulse.set_param("freq", qub_freq_param)

                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add(
                        Reset("reset", cfg=modules.reset),
                        Branch("ge", [], Pulse("pi_pulse", cfg=modules.pi_pulse)),
                        Join(
                            Pulse("res_pulse", cfg=modules.res_pulse),
                            Pulse("qub_pulse", cfg=modules.qub_pulse),
                        ),
                        Readout("readout", cfg=modules.readout),
                    )
                    .declare_sweep("ge", 2)
                    .declare_sweep("res_freq", sched.cfg.sweep.res_freq)
                    .declare_sweep("qub_freq", sched.cfg.sweep.qub_freq)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )
                signals = signals_buffer.array
        plt.close(fig)

        return CKP_Result(
            res_freqs=res_freqs,
            qub_freqs=qub_freqs,
            signals=signals,
            cfg_snapshot=orig_cfg,
        )

    @retrieve_result
    def analyze(
        self, result: CKP_Result | None = None
    ) -> tuple[float, float, float, Figure]:
        assert result is not None, "no result found"

        res_freqs = result.res_freqs
        qub_freqs = result.qub_freqs
        signals = result.signals
        amps = ckp_signal2real(signals)

        g_res_freqs, g_qub_freqs = get_resonance_freq(res_freqs, qub_freqs, amps[0])
        e_res_freqs, e_qub_freqs = get_resonance_freq(res_freqs, qub_freqs, amps[1])

        # y0, slope, yscale, x0, gamma
        fixedparams: list[float | None] = [None] * 5
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
