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
from zcu_tools.utils.fitting.multi_decay import (
    fit_transition_rates,
    calc_lambda_and_amplitude,
)
from zcu_tools.experiment.v2.utils import round_zcu_time

from ..util import calc_populations

# (times, signals)
T1WithToneResult = Tuple[NDArray[np.float64], NDArray[np.float64]]


class T1WithToneCfg(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: NotRequired[PulseCfg]
    test_pulse: PulseCfg
    readout: ReadoutCfg


class T1WithToneExp(AbsExperiment[T1WithToneResult, T1WithToneCfg]):
    def run(self, *args, uniform: bool = False, **kwargs):
        if uniform:
            return self._run_uniform(*args, **kwargs)
        else:
            return self._run_non_uniform(*args, **kwargs)

    def _run_uniform(
        self,
        soc,
        soccfg,
        cfg: T1WithToneCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1WithToneResult:
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
        cfg: T1WithToneCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> T1WithToneResult:
        cfg = deepcopy(cfg)

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        len_sweep = cfg["sweep"]["length"]
        del cfg["sweep"]

        if isinstance(len_sweep, dict):
            ts = np.geomspace(len_sweep["start"], len_sweep["stop"], len_sweep["expts"])
        else:
            ts = np.asarray(len_sweep)
        ts = round_zcu_time(ts, soccfg, gen_ch=cfg["test_pulse"]["ch"])
        ts = np.unique(ts)

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

            def make_prog(cfg: T1WithToneCfg, t1_delay: float) -> ModularProgramV2:
                cfg = deepcopy(cfg)
                Pulse.set_param(cfg["test_pulse"], "length", t1_delay)
                return ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        Reset("reset", cfg.get("reset", {"type": "none"})),
                        Pulse("pi_pulse", cfg.get("pi_pulse")),
                        Pulse("test_pulse", cfg["test_pulse"]),
                        Readout("readout", cfg["readout"]),
                    ],
                )

            def measure_fn(ctx, update_hook):
                cfg = deepcopy(ctx.cfg)
                rounds = cfg.pop("rounds", 1)
                cfg["rounds"] = 1

                progs = [make_prog(cfg, t1_delay) for t1_delay in ts]

                acc_populations = np.zeros((len(ts), 2), dtype=np.float64)
                for ir in range(rounds):
                    for i, prog in enumerate(progs):
                        raw_i = prog.acquire(
                            soc,
                            progress=False,
                            g_center=g_center,
                            e_center=e_center,
                            population_radius=radius,
                        )

                        acc_populations[i] += raw_i[0][0]

                    update_hook(ir + 1, acc_populations / (ir + 1))

                return acc_populations / rounds

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
        result: Optional[T1WithToneResult] = None,
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

        from zcu_tools.utils.fitting.multi_decay import fit_with_vadality

        fit_with_vadality(lens, populations)

        rates, _, fit_pops, (pOpt, _) = fit_transition_rates(lens, populations)

        lambdas, _ = calc_lambda_and_amplitude(tuple(pOpt))

        t1 = 1.0 / lambdas[2]
        t1_b = 1.0 / lambdas[1]

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.set_title(f"T_1 = {t1:.1f} μs, T_1b = {t1_b:.1f} μs")
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
        result: Optional[T1WithToneResult] = None,
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

    def load(self, filepath: str, **kwargs) -> T1WithToneResult:
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
