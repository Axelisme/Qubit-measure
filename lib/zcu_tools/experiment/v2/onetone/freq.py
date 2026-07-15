from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import model_validator

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
    TablePulseReadout,
    sweep2param,
)
from zcu_tools.program.v2.utils import readout_freq_words
from zcu_tools.utils.fitting import HangerModel, TransmissionModel, get_proper_model

SamplingMode = Literal["linear", "homophasal"]


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


class HomophasalSamplingCfg(ConfigBase):
    r_f: float
    rf_w: float
    theta0: float

    @model_validator(mode="after")
    def _validate_positive_fit_params(self) -> HomophasalSamplingCfg:
        if self.r_f <= 0.0:
            raise ValueError(f"r_f must be positive, got {self.r_f}")
        if self.rf_w <= 0.0:
            raise ValueError(f"rf_w must be positive, got {self.rf_w}")
        return self

    @property
    def q_l(self) -> float:
        return self.r_f / self.rf_w


class FreqCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FreqModuleCfg
    sweep: FreqSweepCfg
    sampling_mode: SamplingMode = "linear"
    homophasal: HomophasalSamplingCfg | None = None

    @model_validator(mode="after")
    def _validate_sampling_cfg(self) -> FreqCfg:
        if self.sampling_mode == "homophasal":
            if self.homophasal is None:
                raise ValueError(
                    "sampling_mode='homophasal' requires homophasal fit parameters"
                )
        elif self.homophasal is not None:
            raise ValueError(
                "homophasal fit parameters require sampling_mode='homophasal'"
            )
        return self


def freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


def homophasal_freqs_from_sweep(
    sweep: SweepCfg,
    params: HomophasalSamplingCfg,
) -> NDArray[np.float64]:
    def theta_from_freq(freq: float) -> float:
        return params.theta0 + 2.0 * np.arctan(
            2.0 * params.q_l * (1.0 - freq / params.r_f)
        )

    theta_start = theta_from_freq(sweep.start)
    theta_stop = theta_from_freq(sweep.stop)
    thetas = np.linspace(theta_start, theta_stop, sweep.expts, endpoint=True)
    freqs = params.r_f * (
        1.0 - np.tan((thetas - params.theta0) / 2.0) / (2.0 * params.q_l)
    )
    return np.asarray(freqs, dtype=np.float64)


def _ensure_distinct_homophasal_freqs(freqs: NDArray[np.float64]) -> None:
    if len(freqs) <= 1:
        return
    diffs = np.diff(freqs)
    if np.any(np.isclose(diffs, 0.0, rtol=1e-9, atol=1e-12)):
        raise ValueError(
            "homophasal frequency points collapse after hardware quantization. "
            "Increase the sweep span or reduce expts so adjacent points remain "
            "distinct on the hardware grid."
        )


def homophasal_sweep2array(
    sweep: SweepCfg,
    params: HomophasalSamplingCfg,
    round_info: dict[str, Any],
) -> NDArray[np.float64]:
    freqs = homophasal_freqs_from_sweep(sweep, params)
    rounded = sweep2array(freqs, "freq", round_info, allow_array=True)
    _ensure_distinct_homophasal_freqs(rounded)
    return rounded


class FreqExp(PersistableExperiment[FreqResult, FreqCfg]):
    # freq stores Hz on disk -> scale=MHZ_TO_HZ (disk = memory * 1e6)
    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        tag="onetone/freq",
    )

    def _round_info(self, cfg: FreqCfg, soccfg) -> dict[str, Any]:
        modules = cfg.modules
        return {
            "soccfg": soccfg,
            "gen_ch": modules.readout.pulse_cfg.ch,
            "ro_ch": modules.readout.ro_cfg.ro_ch,
        }

    def _run_uniform(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        orig_cfg = deepcopy(cfg)

        # Predicted frequency points (before mapping to ADC domain)
        freqs = sweep2array(cfg.sweep.freq, "freq", self._round_info(cfg, soccfg))

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

    def _run_nonuniform(
        self,
        soc,
        soccfg,
        cfg: FreqCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FreqResult:
        orig_cfg = deepcopy(cfg)
        params = cfg.homophasal
        assert params is not None, "homophasal sampling requires fit parameters"

        freqs = homophasal_sweep2array(
            cfg.sweep.freq, params, self._round_info(cfg, soccfg)
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
                modules.readout.set_param("freq", float(freqs[0]))
                pulse_cfg = modules.readout.pulse_cfg
                ro_cfg = modules.readout.ro_cfg
                freq_words, ro_freq_words = readout_freq_words(
                    soccfg,
                    freqs,
                    gen_ch=pulse_cfg.ch,
                    ro_ch=ro_cfg.ro_ch,
                    mixer_freq=pulse_cfg.mixer_freq,
                    nqz=pulse_cfg.nqz,
                )

                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add(
                        TablePulseReadout(
                            "readout",
                            modules.readout,
                            idx_reg="freq",
                            freq_words=freq_words,
                            ro_freq_words=ro_freq_words,
                        ),
                    )
                    .declare_sweep("freq", len(freqs))
                    .build_and_acquire(
                        **(acquire_kwargs or {}),
                    )
                )

            return FreqResult(
                freqs=freqs, signals=signals_buffer.array, cfg_snapshot=orig_cfg
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
        if cfg.sampling_mode == "linear":
            return self._run_uniform(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)
        if cfg.sampling_mode == "homophasal":
            return self._run_nonuniform(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)
        raise ValueError(f"Invalid sampling_mode: {cfg.sampling_mode}")

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
