from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from typing_extensions import Any, Optional, TypeAlias

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
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
from zcu_tools.utils.datasaver import load_data, save_data

PowerResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class PowerModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class PowerSweepCfg(ConfigBase):
    gain: SweepCfg


class PowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerModuleCfg
    sweep: PowerSweepCfg


class PowerExp(AbsExperiment[PowerResult, PowerCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: PowerCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> PowerResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        def measure_fn(ctx: TaskState[NDArray[np.float64], Any, PowerCfg], update_hook):
            cfg = ctx.cfg
            modules = cfg.modules

            gain_sweep = cfg.sweep.gain
            gain_param = sweep2param("gain", gain_sweep)
            modules.readout.set_param("gain", gain_param)

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Branch("ge", [], Pulse("qub_pulse", modules.qub_pulse)),
                    Readout("readout", modules.readout),
                ],
                sweep=[
                    ("ge", 2),
                    ("gain", gain_sweep),
                ],
            )
            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                round_hook=lambda i, avg_d: update_hook(i, [tracker]),
                trackers=[tracker],
                **(acquire_kwargs or {}),
            )
            return [tracker]

        with LivePlot1D("Readout Power", "SNR") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    result_shape=(len(gains),),
                    dtype=np.float64,
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(gains, np.abs(ctx.root_data)),
            )

        # record the last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (gains, signals)

        return gains, signals

    def analyze(
        self, result: Optional[PowerResult] = None, penalty_ratio: float = 0.0
    ) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        powers, snrs = result
        snrs = np.abs(snrs)

        # fill NaNs with zeros
        snrs[np.isnan(snrs)] = 0.0

        snrs = gaussian_filter1d(snrs, 1)
        penaltized_snrs = snrs * np.exp(-powers * penalty_ratio)

        max_id = np.argmax(penaltized_snrs)
        max_power = float(powers[max_id])
        max_snr = float(snrs[max_id])

        fig, ax = plt.subplots(figsize=config.figsize)

        ax.plot(powers, snrs)
        ax.axvline(max_power, color="r", ls="--", label=f"max SNR = {max_snr:.2f}")
        ax.set_xlabel("Readout Power")
        ax.set_ylabel("SNR (a.u.)")
        ax.legend()
        ax.grid(True)

        return max_power, fig

    def save(
        self,
        filepath: str,
        result: Optional[PowerResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/ro_optimize/gain",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Probe Power", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerResult:
        signals, gains, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert gains is not None
        assert len(gains.shape) == 1 and len(signals.shape) == 1
        assert gains.shape == signals.shape

        gains = gains.astype(np.float64)
        signals = signals.astype(np.float64)

        if comment is not None:

            cfg, _, _ = parse_comment(comment)

            if cfg is not None:

                self.last_cfg = PowerCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (gains, signals)

        return gains, signals
