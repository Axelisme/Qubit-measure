from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import zcu_tools.utils.fitting as ft
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D, LivePlotter2D
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_decay, fit_dual_decay
from zcu_tools.utils.process import rotate2real

from ...runner import HardTask, Runner


def t1_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real  # type: ignore


T1ResultType = Tuple[np.ndarray, np.ndarray]  # (times, signals)


class T1Experiment(AbsExperiment[T1ResultType]):
    """T1 relaxation time measurement.

    Applies a π pulse and then waits for a variable time before readout
    to measure the qubit's energy relaxation.
    """

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        ts = sweep2array(cfg["sweep"]["length"])

        with LivePlotter1D(
            "Time (us)",
            "Amplitude",
            segment_kwargs={"title": "T1 relaxation"},
            disable=not progress,
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        ModularProgramV2(
                            soccfg,
                            ctx.cfg,
                            modules=[
                                make_reset("reset", ctx.cfg.get("reset")),
                                Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                Delay(
                                    name="t1_delay",
                                    delay=sweep2param(
                                        "length", ctx.cfg["sweep"]["length"]
                                    ),
                                ),
                                make_readout("readout", ctx.cfg["readout"]),
                            ],
                        ).acquire(soc, progress=False, callback=update_hook)
                    ),
                    result_shape=(len(ts),),
                ),
                update_hook=lambda ctx: viewer.update(
                    ts, t1_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(
        self,
        result: Optional[T1ResultType] = None,
        *,
        plot: bool = True,
        max_contrast: bool = True,
        dual_exp: bool = False,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        if max_contrast:
            real_signals = rotate2real(signals).real
        else:
            real_signals = np.abs(signals)

        if dual_exp:
            t1, t1err, t1b, t1berr, y_fit, (pOpt, _) = fit_dual_decay(xs, real_signals)
        else:
            t1, t1err, y_fit, (pOpt, _) = fit_decay(xs, real_signals)

        if plot:
            t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
            if dual_exp:
                t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"

            fig, ax = plt.subplots(figsize=config.figsize)
            ax.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
            ax.plot(xs, y_fit, label="fit")
            if dual_exp:
                ax.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
                ax.set_title(f"T1 = {t1_str}, T1b = {t1b_str}", fontsize=15)
            else:
                ax.set_title(f"T1 = {t1_str}", fontsize=15)
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
            ax.legend()
            fig.tight_layout()
            plt.show()

        return t1, t1err

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )


class T1WithToneExperiment(AbsExperiment[T1ResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> T1ResultType:
        cfg = deepcopy(cfg)

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        cfg["test_pulse"]["waveform"]["length"] = sweep2param(
            "length", cfg["sweep"]["length"]
        )

        ts = sweep2array(cfg["sweep"]["length"])

        signals = Runner(
            task=HardTask(
                measure_fn=lambda ctx, update_hook: (
                    ModularProgramV2(
                        soccfg,
                        ctx.cfg,
                        modules=[
                            make_reset("reset", reset_cfg=cfg.get("reset")),
                            Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                            Pulse(name="test_pulse", cfg=cfg["test_pulse"]),
                            make_readout("readout", readout_cfg=cfg["readout"]),
                        ],
                    ).acquire(soc, progress=progress, callback=update_hook)
                ),
                result_shape=(len(ts),),
            ),
            liveplotter=LivePlotter1D(
                "Time (us)",
                "Amplitude",
                segment_kwargs={"title": "T1 relaxation"},
                disable=not progress,
            ),
            update_hook=lambda viewer, ctx: viewer.update(
                ts, t1_signal2real(np.asarray(ctx.get_data()))
            ),
        ).run(cfg)
        signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (ts, signals)

        return ts, signals

    def analyze(
        self,
        result: Optional[T1ResultType] = None,
        *,
        plot: bool = True,
        max_contrast: bool = True,
        dual_exp: bool = False,
    ) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        xs, signals = result

        if max_contrast:
            real_signals = rotate2real(signals).real
        else:
            real_signals = np.abs(signals)

        if dual_exp:
            t1, t1err, t1b, t1berr, y_fit, (pOpt, _) = fit_dual_decay(xs, real_signals)
        else:
            t1, t1err, y_fit, (pOpt, _) = fit_decay(xs, real_signals)

        if plot:
            t1_str = f"{t1:.2f}us ± {t1err:.2f}us"
            if dual_exp:
                t1b_str = f"{t1b:.2f}us ± {t1berr:.2f}us"

            fig, ax = plt.subplots(figsize=config.figsize)
            ax.plot(xs, real_signals, label="meas", ls="-", marker="o", markersize=3)
            ax.plot(xs, y_fit, label="fit")
            if dual_exp:
                ax.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
                ax.set_title(f"T1 = {t1_str}, T1b = {t1b_str}", fontsize=15)
            else:
                ax.set_title(f"T1 = {t1_str}", fontsize=15)
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Signal Real (a.u.)" if max_contrast else "Magnitude (a.u.)")
            ax.legend()
            fig.tight_layout()
            plt.show()

        return t1, t1err

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
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )


# (values, times, signals)
T1SweepResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


class T1WithToneSweepExperiment(AbsExperiment[T1SweepResultType]):
    SWEEP_MAP = {
        "gain": {"name": "Gain (a.u.)", "param_key": "gain"},
        "freq": {"name": "Frequency (MHz)", "param_key": "freq"},
    }

    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
    ) -> T1SweepResultType:
        cfg = deepcopy(cfg)

        len_sweep = cfg["sweep"].pop("length")

        # extract sweep parameters
        x_key = list(cfg["sweep"].keys())[0]
        if x_key not in self.SWEEP_MAP:
            raise ValueError(f"Unsupported sweep key: {x_key}")
        x_info = self.SWEEP_MAP[x_key]

        cfg["sweep"] = {
            x_info["param_key"]: cfg["sweep"][x_key],
            "length": len_sweep,
        }

        cfg["test_pulse"]["waveform"]["length"] = sweep2param("length", len_sweep)
        cfg["test_pulse"][x_info["param_key"]] = sweep2param(
            x_info["param_key"], cfg["sweep"][x_key]
        )

        values = sweep2array(cfg["sweep"][x_key])  # predicted
        ts = sweep2array(cfg["sweep"]["length"])  # predicted times

        signals = Runner(
            task=HardTask(
                measure_fn=lambda ctx, update_hook: (
                    ModularProgramV2(
                        soccfg,
                        ctx.cfg,
                        modules=[
                            make_reset("reset", reset_cfg=cfg.get("reset")),
                            Pulse(name="pi_pulse", cfg=cfg["pi_pulse"]),
                            Pulse(name="test_pulse", cfg=cfg["test_pulse"]),
                            make_readout("readout", readout_cfg=cfg["readout"]),
                        ],
                    ).acquire(soc, progress=progress, callback=update_hook)
                ),
                result_shape=(len(values), len(ts)),
            ),
            liveplotter=LivePlotter2D(
                x_info["name"],
                "Time (us)",
                disable=not progress,
            ),
            update_hook=lambda viewer, ctx: viewer.update(
                values, ts, t1_signal2real(np.asarray(ctx.get_data()))
            ),
        ).run(cfg)
        signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (values, ts, signals)

        return values, ts, signals

    def analyze(self, result: Optional[T1ResultType] = None) -> Tuple[float, float]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, ts, signals = result

        signals = gaussian_filter(signals, sigma=1)
        real_signals = rotate2real(signals).real

        t1s = np.full(len(values), np.nan, dtype=np.float64)
        t1errs = np.zeros_like(t1s)

        for i in range(len(values)):
            real_signal = real_signals[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signal)):
                continue

            t1, t1err, *_ = fit_decay(ts, real_signal)

            if t1err > 0.3 * t1:
                continue

            t1s[i] = t1
            t1errs[i] = t1err

        if np.all(np.isnan(t1s)):
            raise ValueError("No valid Fitting T1 found. Please check the data.")

        valid_idxs = ~np.isnan(t1s)
        valid_values = values[valid_idxs]
        t1s = t1s[valid_idxs]
        t1errs = t1errs[valid_idxs]

        _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)
        ax1.set_ylabel("T1 over sweep value")
        ax1.imshow(
            real_signals.T,
            aspect="auto",
            extent=[values[0], values[-1], t1s[0], t1s[-1]],
            origin="lower",
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

        return valid_values, t1s, t1errs

    def save(
        self,
        filepath: str,
        result: Optional[T1ResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/t1_with_tone_sweep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, Ts, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Sweep Value", "unit": "a.u.", "values": values},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
