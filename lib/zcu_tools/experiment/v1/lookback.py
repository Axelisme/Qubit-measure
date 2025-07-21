from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.liveplot import AbsLivePlotter, LivePlotter1D
from zcu_tools.program.v1 import OneToneProgram, TwoToneProgram
from zcu_tools.utils.datasaver import save_data

from ..flux import set_flux

LookbackResultType = Tuple[np.ndarray, np.ndarray]


class LookbackExperiment(AbsExperiment[LookbackResultType]):
    def run(
        self,
        soc,
        soccfg,
        cfg: Dict[str, Any],
        *,
        progress: bool = True,
        liveplotter: Optional[AbsLivePlotter] = None,
        qub_pulse: bool = False,
    ) -> LookbackResultType:
        cfg = deepcopy(cfg)

        cfg.setdefault("reps", 1)
        if cfg["reps"] != 1:
            warnings.warn("reps is not 1 in config, this will be ignored.")
            cfg["reps"] = 1

        set_flux(cfg["dev"]["flux_dev"], cfg["dev"]["flux"], progress=True)

        MAX_LEN = 3.32  # us
        ro_cfg = cfg["readout"]["ro_cfg"]

        def run_once(cfg, progress: bool = True) -> LookbackResultType:
            prog = (
                TwoToneProgram(soccfg, cfg)
                if qub_pulse
                else OneToneProgram(soccfg, cfg)
            )
            (Is, Qs), _ = prog.acquire(soc, progress=progress)
            signals = np.array(Is) + 1j * np.array(Qs)

            Ts = prog.get_time_axis(ro_index=0)
            Ts += cfg["readout"]["ro_cfg"]["trig_offset"]

            return Ts, signals

        if ro_cfg["ro_length"] <= MAX_LEN:
            Ts, signals = run_once(cfg, progress=progress)
        else:
            # measure multiple times
            trig_offset = ro_cfg["trig_offset"]
            total_len = trig_offset + ro_cfg["ro_length"]
            ro_cfg["ro_length"] = MAX_LEN

            bar = tqdm(
                total=int((total_len - trig_offset) / MAX_LEN + 0.999),
                desc="Readout",
                smoothing=0,
                disable=not progress,
            )

            Ts = []
            signals = []
            if liveplotter is None:
                liveplotter = LivePlotter1D(
                    "Time (us)", "Amplitude", title="Readout", disable=not progress
                )

            with liveplotter as viewer:
                while trig_offset < total_len:
                    ro_cfg["trig_offset"] = trig_offset

                    Ts_, singals_ = run_once(cfg, progress=False)

                    Ts.append(Ts_)
                    signals.append(singals_)

                    viewer.update(np.concatenate(Ts), np.concatenate(signals))

                    trig_offset += MAX_LEN
                    bar.update()

                bar.close()
                Ts = np.concatenate(Ts)
                signals = np.concatenate(signals)

                sort_idxs = np.argsort(Ts, kind="stable")
                Ts = Ts[sort_idxs]
                signals = signals[sort_idxs]

                viewer.update(Ts, np.abs(signals))

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (Ts, signals)

        return Ts, signals

    def analyze(
        self,
        result: Optional[LookbackResultType] = None,
        *,
        ratio: float = 0.3,
        ro_cfg: Optional[dict] = None,
        plot_fit: bool = True,
    ) -> float:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        Ts, signals = result
        y = np.abs(signals)

        # find first idx where y is larger than ratio * max_y
        offset = Ts[np.argmax(y > ratio * np.max(y))]

        plt.figure(figsize=config.figsize)
        plt.plot(Ts, signals.real, label="I value")
        plt.plot(Ts, signals.imag, label="Q value")
        plt.plot(Ts, y, label="mag")
        if plot_fit:
            plt.axvline(offset, color="r", linestyle="--", label="predict_offset")
        if ro_cfg is not None:
            trig_offset = ro_cfg["trig_offset"]
            ro_length = ro_cfg["ro_length"]
            plt.axvline(trig_offset, color="g", linestyle="--", label="ro start")
            plt.axvline(
                trig_offset + ro_length, color="g", linestyle="--", label="ro end"
            )

        plt.xlabel("Time (us)")
        plt.ylabel("a.u.")
        plt.legend()
        plt.show()

        return offset

    def save(
        self,
        filepath: str,
        result: Optional[LookbackResultType] = None,
        comment: Optional[str] = None,
        tag: str = "lookback",
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
