from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typeguard import check_type
from typing_extensions import Any, NotRequired, Optional, TypeAlias, TypedDict

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.tracker import PCATracker
from zcu_tools.experiment.v2.utils import make_ge_sweep, snr_as_signal, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

LengthResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class LengthModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    qub_pulse: PulseCfg
    readout: PulseReadoutCfg


class LengthCfg(ModularProgramCfg, TaskCfg):
    modules: LengthModuleCfg
    sweep: dict[str, SweepCfg]


class LengthExp(AbsExperiment[LengthResult, LengthCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> LengthResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")
        _cfg = check_type(deepcopy(cfg), LengthCfg)

        length_sweep = _cfg["sweep"]["length"]

        # replace length sweep with ge sweep, and use soft loop for length
        _cfg["sweep"] = {"ge": make_ge_sweep()}

        # set with / without pi gain for qubit pulse
        modules = _cfg["modules"]
        Pulse.set_param(
            modules["qub_pulse"], "on/off", sweep2param("ge", _cfg["sweep"]["ge"])
        )

        readout_cfg = modules["readout"]
        lengths = sweep2array(
            length_sweep,
            "time",
            {
                "soccfg": soccfg,
                "gen_ch": readout_cfg["pulse_cfg"]["ch"],
                "ro_ch": readout_cfg["ro_cfg"]["ro_ch"],
            },
        )

        # set initial readout length and adjust pulse length
        length_params = sweep2param("length", length_sweep)
        PulseReadout.set_param(modules["readout"], "ro_length", length_params)
        PulseReadout.set_param(modules["readout"], "length", lengths.max() + 0.11)

        with LivePlotter1D("Readout Length (us)", "SNR") as viewer:

            def measure_fn(ctx, update_hook):
                modules = ctx.cfg["modules"]
                prog = ModularProgramV2(
                    soccfg,
                    ctx.cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        Pulse("qub_pulse", modules["qub_pulse"]),
                        PulseReadout("readout", modules["readout"]),
                    ],
                )
                tracker = PCATracker()
                avg_d = prog.acquire(
                    soc,
                    progress=False,
                    callback=lambda i, avg_d: update_hook(
                        i, (avg_d, [tracker.covariance], [tracker.rough_median])
                    ),
                    statistic_trackers=[tracker],
                )
                return avg_d, [tracker.covariance], [tracker.rough_median]

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    dtype=np.float64,
                ).scan(
                    "length",
                    lengths.tolist(),
                    before_each=lambda _, ctx, length: PulseReadout.set_param(
                        ctx.cfg["modules"]["readout"], "ro_length", length
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(lengths, np.abs(ctx.root_data)),
            )
            signals = np.asarray(signals)

        # record the last cfg and result
        self.last_cfg = _cfg
        self.last_result = (lengths, signals)

        return lengths, signals

    def analyze(
        self, result: Optional[LengthResult] = None, *, t0: Optional[float] = None
    ) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lengths, signals = result

        snrs = np.abs(signals)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)

        if t0 is None:
            max_id = np.argmax(snrs)
        else:
            max_id = np.argmax(snrs / np.sqrt(lengths + t0))

        max_length = float(lengths[max_id])
        max_snr = float(snrs[max_id])

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(lengths, snrs)
        ax.axvline(max_length, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Readout Length (us)")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return max_length, fig

    def save(
        self,
        filepath: str,
        result: Optional[LengthResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lengths, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Readout Length", "unit": "s", "values": lengths * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> LengthResult:
        signals, lengths, _ = load_data(filepath, **kwargs)
        assert lengths is not None
        assert len(lengths.shape) == 1 and len(signals.shape) == 1
        assert lengths.shape == signals.shape

        lengths = lengths * 1e6  # s -> us

        lengths = lengths.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (lengths, signals)

        return lengths, signals
