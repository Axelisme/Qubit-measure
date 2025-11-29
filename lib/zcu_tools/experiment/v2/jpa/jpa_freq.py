from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import NotRequired
from scipy.ndimage import gaussian_filter1d

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import (
    format_sweep1D,
    make_ge_sweep,
    set_freq_in_dev_cfg,
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
from zcu_tools.utils.datasaver import save_data

JPAFreqResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


class JPAFreqTaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class JPAFreqExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: JPAFreqTaskConfig) -> JPAFreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_freq")

        jpa_freqs = sweep2array(cfg["sweep"]["jpa_freq"], allow_array=True)
        np.random.shuffle(jpa_freqs[1:-1])  # randomize permutation

        cfg["sweep"] = {"ge": make_ge_sweep()}
        Pulse.set_param(
            cfg["pi_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        with LivePlotterScatter("JPA Frequency (MHz)", "Signal Difference") as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="JPA Frequency",
                    sweep_values=jpa_freqs.tolist(),
                    update_cfg_fn=lambda i, ctx, freq: set_freq_in_dev_cfg(
                        ctx.cfg["dev"], freq * 1e6, label="jpa_rf_dev"
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
                                    record_stderr=True,
                                ),
                                prog.get_stderr(),
                            )
                        ),
                        raw2signal_fn=lambda raw: snr_as_signal(raw, ge_axis=0),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    jpa_freqs, np.abs(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (jpa_freqs, signals)

        return jpa_freqs, signals

    def analyze(
        self, result: Optional[JPAFreqResultType] = None
    ) -> Tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_freqs, signals = result

        real_signals = np.abs(signals)

        max_idx = np.nanargmax(real_signals)
        best_jpa_freq = jpa_freqs[max_idx]

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.scatter(jpa_freqs, real_signals, label="signal difference", s=2)
        ax.axvline(
            best_jpa_freq,
            color="r",
            ls="--",
            label=f"best JPA frequency = {best_jpa_freq:.2g} MHz",
        )
        ax.set_xlabel("JPA Frequency (MHz)")
        ax.set_ylabel("Signal Difference (a.u.)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return float(best_jpa_freq), fig

    def save(
        self,
        filepath: str,
        result: Optional[JPAFreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_freqs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Frequency", "unit": "Hz", "values": jpa_freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
