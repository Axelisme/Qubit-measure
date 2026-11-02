from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    PulseReadout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)

from zcu_lab.v2.onetone._shared import (
    FreqResult as FreqResult,
)
from zcu_lab.v2.onetone._shared import (
    linear_freqs,
    spectrum_buffer,
)
from zcu_lab.v2.support.resonator import (
    ResonanceOptions as FreqAnalyzeOptions,
)
from zcu_lab.v2.support.resonator import (
    fit_resonance,
)

SamplingMode = Literal["linear", "homophasal"]


@dataclass(frozen=True)
class FreqAnalysis:
    freq: float
    fwhm: float
    params: dict[str, Any]


class HomophasalSamplingCfg(ConfigBase):
    r_f: Annotated[float, FieldMeta(label="Resonance frequency", unit="MHz")]
    rf_w: Annotated[float, FieldMeta(label="Linewidth", unit="MHz")]
    theta0: Annotated[float, FieldMeta(label="Phase offset", unit="rad")]

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


class FreqCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="填入 resonator readout 設定。先用 lookback 取得 TOF，再以 onetone 取得 kappa，確認 trigger offset 與 waveform duration，單位 us。",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="填入 absolute MHz 起訖值。range_find 不會從已知中心自動設定掃描窗。",
        ),
    ]
    sampling_mode: Annotated[SamplingMode, FieldMeta(label="Sampling mode")] = "linear"
    homophasal: Annotated[
        HomophasalSamplingCfg | None, FieldMeta(label="Homophasal")
    ] = None

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0

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


class FreqExp(Experiment[FreqResult, FreqCfg]):
    Options: ClassVar[type[FreqAnalyzeOptions]] = FreqAnalyzeOptions

    # freq stores Hz on disk -> scale=1e6 (disk = memory * 1e6)
    AXES_SPEC: ClassVar[AxesSpec[FreqResult, FreqCfg]] = AxesSpec(
        axes=(AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="3.0",
        tag="onetone/freq",
    )

    def _round_info(self, cfg: FreqCfg, soccfg) -> dict[str, Any]:
        return {
            "soccfg": soccfg,
            "gen_ch": cfg.readout.gen_ch,
            "ro_ch": cfg.readout.ro_ch,
        }

    def run(self, config: FreqCfg, *, context: RunContext) -> FreqResult:
        """Measure the resonator response with linear or equal-phase frequency spacing.

        Reset, then pulse and read out at each hardware-swept frequency. Return
        hardware-rounded frequencies in MHz and mean complex IQ at each point;
        fitting this response gives the resonance frequency and linewidth.
        Homophasal sampling requires positive fit parameters and rejects points
        that collapse onto the same hardware frequency.
        """
        cfg = deepcopy(config)
        soccfg = context.soccfg
        if cfg.sampling_mode == "linear":
            freqs = linear_freqs(cfg.freq, cfg.readout, context=context)
        elif cfg.sampling_mode == "homophasal":
            params = cfg.homophasal
            if params is None:
                raise ValueError("homophasal sampling requires fit parameters")
            freqs = homophasal_sweep2array(
                cfg.freq, params, self._round_info(cfg, soccfg)
            )
        else:
            raise ValueError(f"Invalid sampling_mode: {cfg.sampling_mode}")

        signals = spectrum_buffer(freqs, plots=context.plots)
        with Schedule(signals, scope=context.scope) as sched:
            builder = TimelineBuilder(
                context.soccfg,
                reps=cfg.reps,
                relax_delay=cfg.relax_delay,
            )
            freq = (
                builder.add_sweep("freq", cfg.freq)
                if cfg.sampling_mode == "linear"
                else builder.add_table_sweep("freq", freqs.tolist())
            )
            if cfg.reset is not None:
                builder.emit(Reset("reset", cfg.reset))
            builder.emit(PulseReadout("readout", cfg.readout, freq=freq))
            sched.run_acquire(builder.build(), context.soc, rounds=cfg.rounds)
        return FreqResult(freqs=freqs, signals=signals.array)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        fig, _ = plots.subplots(num="fit", figsize=(9, 8))
        answer = fit_resonance(source.result.freqs, source.result.signals, options, fig)
        return FreqAnalysis(
            freq=answer.freq, fwhm=answer.fwhm, params=dict(answer.params)
        )
