from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    Pulse,
    make_readout,
    make_reset,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.process import rotate2real
from zcu_tools.utils.fitting import fit_decay_fringe

from ...template import sweep_hard_template
from .util import check_flux_pulse

T2RamseyResultType = Tuple[np.ndarray, np.ndarray, np.ndarray]


def t2ramsey_signal2real(signals: np.ndarray) -> np.ndarray:
    return rotate2real(signals).real


class T2RamseyExperiment(AbsExperiment[T2RamseyResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        detune: float = 0.0,
        progress: bool = True,
    ) -> T2RamseyResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        flx_pulse = cfg["flx_pulse"]

        flx_pulse.setdefault("nqz", 1)
        flx_pulse.setdefault("freq", 0.0)
        flx_pulse.setdefault("phase", 0.0)
        flx_pulse.setdefault("post_delay", 0.0)

        check_flux_pulse(flx_pulse, check_delay=False)

        flx_sweep = cfg["sweep"]["flux"]
        len_sweep = cfg["sweep"]["length"]

        # Flux sweep be outer loop
        cfg["sweep"] = {
            "flux": flx_sweep,
            "length": len_sweep,
        }

        # predict sweep points
        gains = sweep2array(flx_sweep)
        ts = sweep2array(len_sweep)

        # Frequency is swept by FPGA (hard sweep)
        flx_pulse["gain"] = sweep2param("flux", flx_sweep)
        flx_pulse["length"] = sweep2param("length", len_sweep)

        prog = ModularProgramV2(
            soccfg,
            cfg,
            modules=[
                make_reset("reset", reset_cfg=cfg.get("reset")),
                Pulse(name="pi2_pulse1", cfg=cfg["pi2_pulse"]),
                Pulse(name="flux_pulse", cfg=cfg["flx_pulse"]),
                Pulse(
                    name="pi2_pulse2",
                    cfg={  # activate detune
                        **cfg["pi2_pulse"],
                        "phase": cfg["pi2_pulse"]["phase"]
                        + 360 * detune * flx_pulse["length"],
                    },
                ),
                make_readout("readout", readout_cfg=cfg["readout"]),
            ],
        )

        def measure_fn(
            cfg: Dict[str, Any], cb: Optional[Callable[..., None]]
        ) -> np.ndarray:
            return prog.acquire(soc, progress=progress, callback=cb)[0][0].dot([1, 1j])

        # Run 2D soft-hard sweep (flux soft, length hard)
        signals2D = sweep_hard_template(
            cfg,
            measure_fn,
            LivePlotter2D(
                "Local flux gain (a.u.)",
                "Time (us)",
                disable=not progress,
            ),
            ticks=(gains, ts),
            signal2real=t2ramsey_signal2real,
        )

        # Get the actual frequency points used by FPGA
        real_gains = prog.get_pulse_param("flux_pulse", "gain", as_array=True)
        real_ts = prog.get_pulse_param("flux_pulse", "length", as_array=True)
        assert isinstance(real_gains, np.ndarray), "fpts should be an array"
        assert isinstance(real_ts, np.ndarray), "fpts should be an array"
        real_ts += ts[0] - real_ts[0]  # correct absolute offset

        # Cache results
        self.last_cfg = cfg
        self.last_result = (real_gains, real_ts, signals2D)

        return real_gains, real_ts, signals2D

    def analyze(
        self,
        result: Optional[T2RamseyResultType] = None,
        *,
        start_idx: int = 0,
        t2r_cutoff: float = np.inf,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, ts, signals2D = result

        if start_idx > len(ts) - 5:
            raise ValueError(
                f"not enough data to analyze under start_idx={start_idx}, while total length={len(ts)}"
            )

        # start from start_idx
        ts = ts[start_idx:]
        signals2D = signals2D[:, start_idx:]

        real_signals2D = rotate2real(signals2D).real

        t2s = np.full(len(gains), np.nan, dtype=np.float64)
        t2errs = np.zeros_like(t2s)
        detunes = np.full_like(t2s, np.nan)
        detune_errs = np.zeros_like(t2s)

        fit_params = None
        for i in range(len(gains)):
            real_signals = real_signals2D[i, :]

            # skip if have nan data
            if np.any(np.isnan(real_signals)):
                continue

            t2r, t2err, detune, derr, _, (pOpt, _) = fit_decay_fringe(
                ts, real_signals, fit_params=fit_params
            )

            if t2r > t2r_cutoff or t2err > 0.5 * t2r:
                continue

            fit_params = tuple(pOpt)

            t2s[i] = t2r
            t2errs[i] = t2err
            detunes[i] = detune
            detune_errs[i] = derr

        if np.all(np.isnan(t2s)):
            raise ValueError("No valid Fitting T2 found. Please check the data.")

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)
        ax1.errorbar(gains, t2s, yerr=t2errs, label="Fitting T2")
        ax1.set_ylabel("T2 (us)")
        ax1.set_ylim(bottom=0)
        ax2.errorbar(gains, detunes, yerr=detune_errs, label="Fitting detune")
        ax2.set_ylabel("Detune (MHz)")
        ax2.set_xlabel("Flux pulse gain (a.u.)")
        plt.plot()

        return t2s, t2errs, detunes, detune_errs

    def save(
        self,
        filepath: str,
        result: Optional[T2RamseyResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/ge/t2ramsey",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, Ts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Flux pulse gain", "unit": "a.u.", "values": gains},
            y_info={"name": "Time", "unit": "s", "values": Ts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
