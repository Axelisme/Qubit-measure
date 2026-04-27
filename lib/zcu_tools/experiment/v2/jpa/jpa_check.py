from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Mapping, Optional, TypeAlias

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import (
    make_comment,
    parse_comment,
    set_output_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

CheckResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def check_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class CheckModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    readout: PulseReadoutCfg


class CheckSweepCfg(ConfigBase):
    freq: SweepCfg


class CheckCfg(ProgramV2Cfg, ExpCfgModel):
    modules: CheckModuleCfg
    dev: Mapping[str, DeviceInfo] = ...
    sweep: CheckSweepCfg


class CheckExp(AbsExperiment[CheckResult, CheckCfg]):
    OUTPUT_MAP = {0: "off", 1: "on"}

    def run(self, soc, soccfg, cfg: CheckCfg) -> CheckResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        outputs = np.array([0, 1])
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.pulse_cfg.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, CheckCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            setup_devices(cfg, progress=False)
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.readout.set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    PulseReadout("readout", modules.readout),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(soc, progress=False, round_hook=update_hook)

        with LivePlot1D(
            "Frequency (MHz)", "Magnitude", segment_kwargs=dict(num_lines=2)
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs),),
                    pbar_n=cfg.rounds,
                ).scan(
                    "JPA on/off",
                    outputs.tolist(),
                    before_each=lambda _, ctx, output: (
                        (dev := ctx.cfg.dev) is not None
                        and set_output_in_dev_cfg(
                            dev,
                            self.OUTPUT_MAP[output],  # type: ignore
                            label="jpa_rf_dev",
                        )
                    ),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, check_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        self.last_cfg = deepcopy(cfg)
        self.last_result = (outputs, freqs, signals)
        return outputs, freqs, signals

    def analyze(self, result: Optional[CheckResult] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        outputs, freqs, signals2D = result
        real_signals = check_signal2real(signals2D)

        fig, ax = plt.subplots(figsize=config.figsize)
        for i, output in enumerate(outputs):
            ax.plot(
                freqs,
                real_signals[i, :],
                label=f"JPA {self.OUTPUT_MAP[output]}",
                marker="o",
                markersize=4,
                linestyle="-",
            )

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Magnitude (a.u.)")
        ax.legend()
        ax.grid(True)
        return fig

    def save(
        self,
        filepath: str,
        result: Optional[CheckResult] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/check",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        cfg = self.last_cfg
        assert cfg is not None

        outputs, freqs, signals2D = result
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            y_info={"name": "JPA Output", "unit": "a.u.", "values": outputs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> CheckResult:
        signals2D, freqs, outputs, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert freqs is not None and outputs is not None
        assert len(freqs.shape) == 1 and len(outputs.shape) == 1
        assert signals2D.shape == (len(outputs), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz
        outputs = outputs.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = CheckCfg.validate_or_warn(cfg, source=filepath)

        self.last_result = (outputs, freqs, signals2D)
        return outputs, freqs, signals2D
