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
    SweepCfg,
    TablePulseReadout,
    sweep2param,
)
from zcu_tools.program.v2.utils import readout_freq_words

from zcu_lab.v2._support.resonator import (
    ResonanceOptions as FreqAnalyzeOptions,
)
from zcu_lab.v2._support.resonator import (
    fit_resonance,
)
from zcu_lab.v2.onetone._shared import (
    FreqModuleCfg as FreqModuleCfg,
)
from zcu_lab.v2.onetone._shared import (
    FreqResult as FreqResult,
)
from zcu_lab.v2.onetone._shared import (
    FreqSweepCfg as FreqSweepCfg,
)
from zcu_lab.v2.onetone._shared import (
    LinearFreqCfg,
    linear_freqs,
    spectrum_buffer,
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


class FreqCfg(LinearFreqCfg):
    sampling_mode: Annotated[SamplingMode, FieldMeta(label="Sampling mode")] = "linear"
    homophasal: Annotated[
        HomophasalSamplingCfg | None, FieldMeta(label="Homophasal")
    ] = None

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
        axes=(AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FreqCfg,
        cfg_schema_version="2.0",
        tag="onetone/freq",
    )

    def _round_info(self, cfg: FreqCfg, soccfg) -> dict[str, Any]:
        modules = cfg.modules
        return {
            "soccfg": soccfg,
            "gen_ch": modules.readout.gen_ch,
            "ro_ch": modules.readout.ro_ch,
        }

    def run(self, config: FreqCfg, *, context: RunContext) -> FreqResult:
        """Measure the resonator response with linear or equal-phase frequency spacing.

        Reset, then pulse and read out at each hardware-swept frequency. Return
        hardware-rounded frequencies in MHz and mean complex IQ at each point;
        fitting this response gives the resonance frequency and linewidth.
        Homophasal sampling requires positive fit parameters and rejects points
        that collapse onto the same hardware frequency.
        """
        # 1. Prepare the frequency grid without changing the caller's cfg.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        if cfg.sampling_mode == "linear":
            freqs = linear_freqs(cfg, context=context)
        elif cfg.sampling_mode == "homophasal":
            params = cfg.homophasal
            if params is None:
                raise ValueError("homophasal sampling requires fit parameters")
            freqs = homophasal_sweep2array(
                cfg.sweep.freq, params, self._round_info(cfg, soccfg)
            )
        else:
            raise ValueError(f"Invalid sampling_mode: {cfg.sampling_mode}")

        signals_buffer = spectrum_buffer(freqs, plots=context.plots)

        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            modules = cfg.modules

            # 2. Reset, read out and sweep frequencies on the FPGA.
            if cfg.sampling_mode == "linear":
                freq_sweep = cfg.sweep.freq
                modules.readout.set_param("freq", sweep2param("freq", freq_sweep))
                _ = (
                    sched.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add(PulseReadout("readout", modules.readout))
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire()
                )
            else:
                # Homophasal sampling uses the calibrated DDS table.
                modules.readout.set_param("freq", float(freqs[0]))
                readout_cfg = modules.readout
                freq_words, ro_freq_words = readout_freq_words(
                    soccfg,
                    freqs,
                    gen_ch=readout_cfg.gen_ch,
                    ro_ch=readout_cfg.ro_ch,
                    mixer_freq=readout_cfg.mixer_freq,
                    nqz=readout_cfg.nqz,
                )

                # Update DDS words after each point, wrapping for the next rep;
                # the table readout preserves the native sweep's phase timing.
                builder = (
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
                )

                # 3. Average reps/rounds and retain completed data on cancellation.
                _ = builder.build_and_acquire()

        return FreqResult(freqs=freqs, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[FreqCfg, FreqResult],
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        fig, _ = plots.subplots("fit", figsize=(9, 8))
        answer = fit_resonance(source.result.freqs, source.result.signals, options, fig)
        return FreqAnalysis(
            freq=answer.freq, fwhm=answer.fwhm, params=dict(answer.params)
        )
