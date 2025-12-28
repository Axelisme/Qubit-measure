from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
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
from zcu_tools.utils.fitting import fit_transition_rates
from zcu_tools.experiment.v2.utils import round_zcu_time

from ..util import calc_populations

# (times, signals)
T1ResultType = Tuple[NDArray[np.float64], NDArray[np.float64]]


def calc_transition_rate(g_p: float, e_p: float, t1: float) -> Tuple[float, float]:
    """Calculate transition rates from T1 times and steady populations."""
    if np.isclose(t1, 0.0, atol=1e-1) or not np.isfinite(t1):
        return np.nan, np.nan

    # Using detailed balance: p_g * gamma_ge = p_e * gamma_eg
    # And total rate: gamma_total = gamma_ge + gamma_eg = 1 / t1

    gamma_ge = (e_p / (g_p + e_p)) / t1
    gamma_eg = (g_p / (g_p + e_p)) / t1

    return gamma_ge, gamma_eg


class T1WithToneTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: NotRequired[PulseCfg]
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneExp(AbsExperiment[T1ResultType, T1WithToneTaskConfig]):
    def run(self, *args, uniform: bool = False, **kwargs):
        if uniform:
            return self._run_uniform(*args, **kwargs)
        else:
            return self._run_non_uniform(*args, **kwargs)

    def _run_uniform(
        self,
        soc,
        soccfg,
        cfg: T1WithToneTaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        Pulse.set_param(
            cfg["test_pulse"], "length", sweep2param("length", cfg["sweep"]["length"])
        )

        ts = sweep2array(cfg["sweep"]["length"])
        ts = round_zcu_time(ts, soccfg, gen_ch=cfg["test_pulse"]["ch"])

        with LivePlotter1D(
            "Time (us)",
            "Amplitude",
            segment_kwargs=dict(
                num_lines=3,
                line_kwargs=[
                    dict(label="Ground"),
                    dict(label="Excited"),
                    dict(label="Other"),
                ],
            ),
        ) as viewer:
            populations = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                                Pulse("pi_pulse", ctx.cfg.get("pi_pulse")),
                                Pulse("test_pulse", ctx.cfg["test_pulse"]),
                                Readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(
                            soc,
                            progress=False,
                            callback=update_hook,
                            g_center=g_center,
                            e_center=e_center,
                            population_radius=radius,
                        )
                    ),
                    raw2signal_fn=lambda raw: raw[0][0],
                    result_shape=(len(ts), 2),
                    dtype=np.float64,
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(ts, calc_populations(ctx.data).T),
            )

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, populations)

        return ts, populations

    def _run_non_uniform(
        self,
        soc,
        soccfg,
        cfg: T1WithToneTaskConfig,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]
        del cfg["sweep"]

        if isinstance(len_sweep, dict):
            ts = (
                np.linspace(
                    len_sweep["start"] ** (1 / 2),
                    len_sweep["stop"] ** (1 / 2),
                    len_sweep["expts"],
                )
                ** 2
            )
        else:
            ts = np.asarray(len_sweep)
        ts = round_zcu_time(ts, soccfg, gen_ch=cfg["test_pulse"]["ch"])
        ts = np.unique(ts)

        def measure_fn(ctx, update_hook):
            rounds = ctx.cfg.pop("rounds", 1)
            ctx.cfg["rounds"] = 1

            acc_populations = np.zeros((len(ts), 2), dtype=np.float64)
            for ir in range(rounds):
                for i, t1_delay in enumerate(ts):
                    Pulse.set_param(ctx.cfg["test_pulse"], "length", t1_delay)
                    raw_i = ModularProgramV2(
                        soccfg,
                        ctx.cfg,
                        modules=[
                            Reset("reset", ctx.cfg.get("reset", {"type": "none"})),
                            Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                            Pulse("test_pulse", ctx.cfg["test_pulse"]),
                            Readout("readout", ctx.cfg["readout"]),
                        ],
                    ).acquire(
                        soc,
                        progress=False,
                        g_center=g_center,
                        e_center=e_center,
                        population_radius=radius,
                    )

                    acc_populations[i] += raw_i[0][0]

                update_hook(ir, acc_populations / (ir + 1))

            return acc_populations / rounds

        with LivePlotter1D(
            "Time (us)",
            "Amplitude",
            segment_kwargs=dict(
                num_lines=3,
                line_kwargs=[
                    dict(label="Ground"),
                    dict(label="Excited"),
                    dict(label="Other"),
                ],
            ),
        ) as viewer:
            populations = run_task(
                task=HardTask(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: raw,
                    result_shape=(len(ts), 2),
                    dtype=np.float64,
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    ts, calc_populations(np.asarray(ctx.data)).T
                ),
            )
        populations = np.asarray(populations)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, populations)

        return ts, populations

    def analyze(
        self,
        result: Optional[T1ResultType] = None,
        *,
        confusion_matrix: Optional[NDArray[np.float64]] = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, populations = result

        populations = calc_populations(populations)

        if confusion_matrix is not None:  # readout correction
            populations = populations @ np.linalg.inv(confusion_matrix)
            populations = np.clip(populations, 0.0, 1.0)

        rates, _, fit_pops, _ = fit_transition_rates(lens, populations)

        Rs = []
        for i in range(30, len(lens)):
            rate, *_ = fit_transition_rates(lens[:i], populations[:i])
            Rs.append(rate)
        Rs = np.array(Rs)

        fig, ax = plt.subplots(figsize=config.figsize)
        for i in range(Rs.shape[1]):
            ax.plot(lens[30:], Rs[:, i])
        plt.show(fig)

        T_g = rates[0] + rates[5]
        T_e = rates[1] + rates[2]
        t1 = 1.0 / (T_g + T_e)

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.set_title(f"T_1 = {t1:.1f} μs")
        ax.plot(lens, fit_pops[:, 0], color="blue", ls="--", label="Ground Fit")
        ax.plot(lens, fit_pops[:, 1], color="red", ls="--", label="Excited Fit")
        ax.plot(lens, fit_pops[:, 2], color="green", ls="--", label="Other Fit")
        plot_kwargs = dict(ls="-", marker=".", markersize=3)
        ax.plot(lens, populations[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
        ax.plot(lens, populations[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
        ax.plot(lens, populations[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Population")
        # ax.set_ylim(0.0, 1.0)
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1_with_tone",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            y_info={"name": "GE population", "unit": "a.u.", "values": [0, 1]},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> T1ResultType:
        populations, Ts, y_values = load_data(filepath, **kwargs)
        assert Ts is not None and y_values is not None
        assert len(Ts.shape) == 1 and len(y_values.shape) == 1
        assert populations.shape == (len(y_values), len(Ts))

        Ts = Ts * 1e6  # s -> us
        populations = populations.T  # transpose back

        Ts = Ts.astype(np.float64)
        populations = populations.astype(np.float64)

        self.last_cfg = None
        self.last_result = (Ts, populations)

        return Ts, populations
