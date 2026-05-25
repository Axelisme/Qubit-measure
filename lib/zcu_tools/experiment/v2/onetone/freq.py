from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Literal, Optional, cast

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
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
from zcu_tools.utils.fitting import HangerModel, TransmissionModel, get_proper_model


@dataclass(frozen=True)
class FreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: Optional[FreqCfg] = None


class FreqModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    readout: PulseReadoutCfg


class FreqSweepCfg(ConfigBase):
    freq: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg


def freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FreqExp(AbsExperiment[FreqResult, FreqCfg]):
    def run(self, soc, soccfg, cfg: FreqCfg) -> FreqResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        # Predicted frequency points (before mapping to ADC domain)
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": modules.readout.pulse_cfg.ch,
                "ro_ch": modules.readout.ro_cfg.ro_ch,
            },
        )

        def measure_fn(
            ctx: TaskState, update_hook: Optional[Callable]
        ) -> list[NDArray[np.float64]]:
            cfg = cast(FreqCfg, ctx.cfg)
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
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
            )

        # run experiment
        with LivePlot1D("Frequency (MHz)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, freq_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = FreqResult(
            freqs=freqs, signals=signals, cfg_snapshot=self.last_cfg
        )

        return self.last_result

    def analyze(
        self,
        result: Optional[FreqResult] = None,
        *,
        model_type: Literal["hm", "t", "auto"] = "auto",
        edelay: Optional[float] = None,
        fit_bg_slope: bool = False,
    ) -> tuple[float, float, dict[str, Any], Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs = result.freqs
        signals = result.signals

        # remove first and last point, sometimes they have problems
        freqs = freqs[1:-1]
        signals = signals[1:-1]

        if model_type == "hm":
            model = HangerModel()
        elif model_type == "t":
            model = TransmissionModel()
        elif model_type == "auto":
            model = get_proper_model(freqs, signals)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        param_dict = model.fit(freqs, signals, edelay, fit_bg_slope=fit_bg_slope)
        fig = model.visualize_fit(freqs, signals, param_dict)  # type: ignore

        return (
            float(param_dict["freq"]),
            float(param_dict["fwhm"]),
            dict(param_dict),
            fig,
        )

    def save(
        self,
        filepath: str,
        result: Optional[FreqResult] = None,
        comment: Optional[str] = None,
        tag: str = "onetone/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs = result.freqs
        signals = result.signals

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        signals, freqs, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert len(freqs.shape) == 1 and len(signals.shape) == 1
        assert freqs.shape == signals.shape

        freqs = freqs * 1e-6  # Hz -> MHz

        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        cfg_snapshot = None
        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                cfg_snapshot = FreqCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = FreqResult(
            freqs=freqs, signals=signals, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
