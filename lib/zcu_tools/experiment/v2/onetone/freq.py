from __future__ import annotations

from copy import deepcopy

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    Literal,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import HangerModel, TransmissionModel, get_proper_model

# (freqs, signals)
FreqResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


class FreqModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    readout: PulseReadoutCfg


class FreqCfg(ModularProgramCfg, TaskCfg):
    modules: FreqModuleCfg
    sweep: dict[str, SweepCfg]


def freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FreqExp(AbsExperiment[FreqResult, FreqCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> FreqResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        _cfg = check_type(deepcopy(cfg), FreqCfg)
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        # Predicted frequency points (before mapping to ADC domain)
        freqs = sweep2array(
            _cfg["sweep"]["freq"],
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": modules["readout"].pulse_cfg.ch,
                "ro_ch": modules["readout"].ro_cfg.ro_ch,
            },
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: FreqCfg = cast(FreqCfg, ctx.cfg)
            modules = cfg["modules"]

            freq_sweep = cfg["sweep"]["freq"]
            freq_param = sweep2param("freq", freq_sweep)
            modules["readout"].set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    PulseReadout("readout", modules["readout"]),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(soc, progress=False, round_hook=update_hook)

        # run experiment
        with LivePlot1D("Frequency (MHz)", "Amplitude") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs),),
                    pbar_n=_cfg["rounds"],
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, freq_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (freqs, signals)

        return freqs, signals

    def analyze(
        self,
        result: Optional[FreqResult] = None,
        *,
        model_type: Literal["hm", "t", "auto"] = "auto",
        edelay: Optional[float] = None,
    ) -> tuple[float, float, dict[str, Any], Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, signals = result

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

        param_dict = model.fit(freqs, signals, edelay)
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

        freqs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FreqResult:
        signals, freqs, _ = load_data(filepath, **kwargs)
        assert len(freqs.shape) == 1 and len(signals.shape) == 1
        assert freqs.shape == signals.shape

        freqs = freqs * 1e-6  # Hz -> MHz

        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (freqs, signals)

        return freqs, signals
