from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Mapping, Optional, TypeAlias, cast

from zcu_tools.config import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import (
    make_comment,
    parse_comment,
    set_power_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.experiment.v2.utils.tracker import MomentTracker
from zcu_tools.liveplot import LivePlotScatter
from zcu_tools.program import SweepCfg
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
)
from zcu_tools.utils.datasaver import load_data, save_data

PowerResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class PowerModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class PowerSweepCfg(ConfigBase):
    jpa_power: SweepCfg


class PowerCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerModuleCfg
    dev: Mapping[str, DeviceInfo] = ...
    sweep: PowerSweepCfg


class PowerExp(AbsExperiment[PowerResult, PowerCfg]):
    def run(self, soc, soccfg, cfg: PowerCfg) -> PowerResult:
        jpa_powers = sweep2array(cfg.sweep.jpa_power, allow_array=True)
        np.random.shuffle(jpa_powers[1:-1])

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any, PowerCfg],
            update_hook: Optional[Callable[[int, list[MomentTracker]], None]],
        ) -> list[MomentTracker]:
            cfg = ctx.cfg
            setup_devices(cfg, progress=False)
            modules = cfg.modules

            assert update_hook is not None

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Branch("ge", [], Pulse("pi_pulse", modules.pi_pulse)),
                    Readout("readout", modules.readout),
                ],
                sweep=[("ge", 2)],
            )

            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                round_hook=lambda i, _avg_d: update_hook(i, [tracker]),
                trackers=[tracker],
            )
            return [tracker]

        with LivePlotScatter("Power (dBm)", "Signal Difference") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    dtype=np.float64,
                    pbar_n=cfg.rounds,
                ).scan(
                    "power (dBm)",
                    jpa_powers.tolist(),
                    before_each=lambda _, ctx, gain: set_power_in_dev_cfg(
                        ctx.cfg.dev, gain, label="jpa_rf_dev"
                    ),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(jpa_powers, np.abs(ctx.root_data)),
            )
            signals = np.asarray(signals)

        self.last_cfg = deepcopy(cfg)
        self.last_result = (jpa_powers, signals)
        return jpa_powers, signals

    def analyze(self, result: Optional[PowerResult] = None) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_powers, signals = result
        snrs = np.abs(signals)

        max_idx = np.nanargmax(snrs)
        best_jpa_power = jpa_powers[max_idx]

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.scatter(jpa_powers, snrs, label="signal difference", s=1)
        ax.axvline(
            best_jpa_power,
            color="r",
            ls="--",
            label=f"best JPA power = {best_jpa_power:.2g} dBm",
        )
        ax.set_xlabel("JPA Frequency (MHz)")
        ax.set_ylabel("Signal Difference (a.u.)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return float(best_jpa_power), fig

    def save(
        self,
        filepath: str,
        result: Optional[PowerResult] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/power",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_powers, signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Power", "unit": "dBm", "values": jpa_powers},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerResult:
        signals, jpa_powers, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert jpa_powers is not None
        assert len(jpa_powers.shape) == 1 and len(signals.shape) == 1
        assert jpa_powers.shape == signals.shape

        jpa_powers = jpa_powers.astype(np.float64)
        signals = signals.astype(np.float64)

        if comment is not None:

            cfg, _, _ = parse_comment(comment)

            if cfg is not None:

                self.last_cfg = PowerCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (jpa_powers, signals)

        return jpa_powers, signals
