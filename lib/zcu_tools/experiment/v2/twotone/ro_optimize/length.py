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
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data

LengthResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class LengthModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    qub_pulse: PulseCfg
    readout: PulseReadoutCfg


class LengthSweepCfg(ConfigBase):
    length: SweepCfg


class LengthCfg(ProgramV2Cfg, ExpCfgModel):
    modules: LengthModuleCfg
    sweep: LengthSweepCfg


class LengthExp(AbsExperiment[LengthResult, LengthCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: LengthCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> LengthResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        readout_cfg = modules.readout
        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "ro_ch": readout_cfg.ro_cfg.ro_ch},
        )
        modules.readout.set_param("length", lengths.max() + 0.11)

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, LengthCfg], update_hook
        ):
            cfg = ctx.cfg
            modules = cfg.modules

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Branch("ge", [], Pulse("qub_pulse", modules.qub_pulse)),
                    PulseReadout("readout", modules.readout),
                ],
                sweep=[("ge", 2)],
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

        def average_signals(
            signals: list[list[NDArray[np.float64]]],
        ) -> NDArray[np.float64]:
            return np.mean([s[0] for s in signals], axis=0)

        with LivePlot1D("Readout Length (us)", "SNR") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    dtype=np.float64,
                    pbar_n=cfg.rounds,
                ).scan(
                    "length",
                    lengths.tolist(),
                    before_each=lambda _, ctx, length: (
                        ctx.cfg.modules.readout.set_param("ro_length", length)
                    ),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(lengths, np.abs(ctx.root_data)),
            )
            signals = np.asarray(signals)

        # record the last cfg and result
        self.last_cfg = deepcopy(cfg)
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

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Readout Length", "unit": "s", "values": lengths * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> LengthResult:
        signals, lengths, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert lengths is not None
        assert len(lengths.shape) == 1 and len(signals.shape) == 1
        assert lengths.shape == signals.shape

        lengths = lengths * 1e6  # s -> us

        lengths = lengths.astype(np.float64)
        signals = signals.astype(np.float64)

        if comment is not None:

            cfg, _, _ = parse_comment(comment)

            if cfg is not None:

                self.last_cfg = LengthCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (lengths, signals)

        return lengths, signals
