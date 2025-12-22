from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import (
    format_sweep1D,
    make_ge_sweep,
    set_power_in_dev_cfg,
    sweep2array,
)
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.experiment.v2.utils import snr_as_signal
from zcu_tools.liveplot import LivePlotterScatter
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

JPAPowerResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class JPAFreqTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg

    dev: Mapping[str, DeviceInfo]


class JPAPowerExp(AbsExperiment[JPAPowerResultType, JPAFreqTaskConfig]):
    def run(self, soc, soccfg, cfg: JPAFreqTaskConfig) -> JPAPowerResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_power")

        jpa_powers = sweep2array(cfg["sweep"]["jpa_power"], allow_array=True)
        np.random.shuffle(jpa_powers[1:-1])

        cfg["sweep"] = {"ge": make_ge_sweep()}
        Pulse.set_param(
            cfg["pi_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        with LivePlotterScatter("Power (dBm)", "Signal Difference") as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="power (dBm)",
                    sweep_values=jpa_powers.tolist(),
                    update_cfg_fn=lambda i, ctx, pdr: set_power_in_dev_cfg(
                        ctx.cfg["dev"], pdr, label="jpa_rf_dev"
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            (
                                prog := ModularProgramV2(
                                    soccfg,
                                    ctx.cfg,
                                    modules=[
                                        Reset(
                                            "reset",
                                            ctx.cfg.get("reset", {"type": "none"}),
                                        ),
                                        Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                        Readout("readout", ctx.cfg["readout"]),
                                    ],
                                )
                            )
                            and (
                                prog.acquire(
                                    soc,
                                    progress=False,
                                    callback=update_hook,
                                    record_statistic=True,
                                ),
                                prog.get_covariance(),
                                prog.get_median(),
                            )
                        ),
                        raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    jpa_powers, np.abs(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (jpa_powers, signals)

        return jpa_powers, signals

    def analyze(
        self, result: Optional[JPAPowerResultType] = None
    ) -> Tuple[float, Figure]:
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
        result: Optional[JPAPowerResultType] = None,
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

    def load(self, filepath: str, **kwargs) -> JPAPowerResultType:
        signals, jpa_powers, _ = load_data(filepath, **kwargs)
        assert jpa_powers is not None
        assert len(jpa_powers.shape) == 1 and len(signals.shape) == 1
        assert jpa_powers.shape == signals.shape

        jpa_powers = jpa_powers.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (jpa_powers, signals)

        return jpa_powers, signals
