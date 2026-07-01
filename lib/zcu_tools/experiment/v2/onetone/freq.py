from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import HangerModel, TransmissionModel, get_proper_model


@dataclass(frozen=True)
class FreqResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FreqCfg | None = None


class FreqModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    readout: PulseReadoutCfg


class FreqSweepCfg(ConfigBase):
    freq: SweepCfg


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg


def freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    # freq stores Hz on disk -> scale=MHZ_TO_HZ (disk = memory * 1e6)
    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="onetone/freq",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        orig_cfg = deepcopy(cfg)
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

        setup_devices(cfg, progress=True)

        with LivePlot1D("Frequency (MHz)", "Amplitude") as viewer:
            signals_buffer = SignalBuffer(
                (len(freqs),),
                on_update=lambda data: viewer.update(freqs, freq_signal2real(data)),
            )

            with Schedule(cfg, signals_buffer) as sched:
                cfg = sched.cfg
                modules = cfg.modules

                freq_sweep = cfg.sweep.freq
                modules.readout.set_param("freq", sweep2param("freq", freq_sweep))

                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add(PulseReadout("readout", modules.readout))
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )

            return FreqResult(
                freqs=freqs, signals=signals_buffer.array, cfg_snapshot=orig_cfg
            )

    @retrieve_result
    def analyze(
        self,
        result: FreqResult | None = None,
        *,
        model_type: Literal["hm", "t", "auto"] = "auto",
        edelay: float | None = None,
        fit_bg_slope: bool = False,
    ) -> tuple[float, float, dict[str, Any], Figure]:
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
