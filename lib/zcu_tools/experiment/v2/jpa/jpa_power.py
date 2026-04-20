from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import (
    format_sweep1D,
    set_power_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.tracker import MomentTracker
from zcu_tools.experiment.v2.utils import snr_as_signal, sweep2array
from zcu_tools.liveplot import LivePlotScatter
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
)
from zcu_tools.utils.datasaver import load_data, save_data

PowerResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]


class PowerModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class PowerCfg(ModularProgramCfg, TaskCfg):
    modules: PowerModuleCfg
    sweep: dict[str, SweepCfg]


class PowerExp(AbsExperiment[PowerResult, PowerCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> PowerResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_power")
        _cfg = check_type(deepcopy(cfg), PowerCfg)

        jpa_powers = sweep2array(_cfg["sweep"]["jpa_power"], allow_array=True)
        np.random.shuffle(jpa_powers[1:-1])

        def measure_fn(
            ctx: TaskState[NDArray[np.float64], Any],
            update_hook: Optional[Callable[[int, list[MomentTracker]], None]],
        ) -> list[MomentTracker]:
            cfg: PowerCfg = cast(PowerCfg, ctx.cfg)
            setup_devices(cfg, progress=False)
            modules = cfg["modules"]

            assert update_hook is not None

            prog = ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    Branch("ge", [], Pulse("pi_pulse", modules["pi_pulse"])),
                    Readout("readout", modules["readout"]),
                ],
                sweep=[("ge", 2)],
            )

            tracker = MomentTracker()
            prog.acquire(
                soc,
                progress=False,
                callback=lambda i, avg_d: update_hook(i, [tracker]),
                statistic_trackers=[tracker],
            )
            return [tracker]

        with LivePlotScatter("Power (dBm)", "Signal Difference") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    dtype=np.float64,
                    pbar_n=_cfg["rounds"],
                ).scan(
                    "power (dBm)",
                    jpa_powers.tolist(),
                    before_each=lambda i, ctx, gain: set_power_in_dev_cfg(
                        ctx.cfg["dev"], gain, label="jpa_rf_dev"
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(jpa_powers, np.abs(ctx.root_data)),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
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

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Power", "unit": "dBm", "values": jpa_powers},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerResult:
        signals, jpa_powers, _ = load_data(filepath, **kwargs)
        assert jpa_powers is not None
        assert len(jpa_powers.shape) == 1 and len(signals.shape) == 1
        assert jpa_powers.shape == signals.shape

        jpa_powers = jpa_powers.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (jpa_powers, signals)

        return jpa_powers, signals
