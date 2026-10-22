"""Thermal B11 common-frequency × duration Rabi chevron."""

from typing import Annotated, Self

import numpy as np
from pydantic import model_validator
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    US_TO_S,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import SweepCfg

from .._drive import DualLineCfg
from .._rabi_scan import (
    DurationScanCfg,
    RabiAnalysis,
    RabiScanResult,
    fit_duration_rates,
    joint_return,
    plot_duration_rates,
    plot_return_data,
    run_duration_scan,
)


class ChevronCfg(DurationScanCfg):
    """Common MHz frequency scan with fixed signed gains and relative phase.

    Both frequencies change together; drive waveforms and timing remain source
    values except for the explicit common duration axis. No automatic gate or
    frequency selection is inferred from a single population minimum.
    """

    frequency: Annotated[
        SweepCfg, FieldMeta(unit="MHz", label="Common drive frequency")
    ]

    @model_validator(mode="after")
    def validate_frequency(self) -> Self:
        """Require finite common frequency coordinates."""
        if not np.all(np.isfinite(sweep2array(self.frequency))):
            raise ValueError("Common frequencies must be finite")
        return self


class ChevronExp(PersistableExperiment[RabiScanResult, ChevronCfg]):
    """Stateless common-frequency acquisition and native persistence."""

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("tone_indices", "Tone", "", IDENTITY, dtype=np.int64),
            Axis("shot_indices", "Shot", "", IDENTITY, dtype=np.int64),
            Axis("durations", "Drive duration", "s", US_TO_S),
            Axis("coordinates", "Common drive frequency", "Hz", MHZ_TO_HZ),
        ),
        z=ZSpec("signals", "Raw IQ", "a.u."),
        result_type=RabiScanResult,
        cfg_type=ChevronCfg,
        cfg_schema_version="1.0",
        tag="ftf/chevron",
    )

    def run(self, config: ChevronCfg, *, context: RunContext) -> RabiScanResult:
        """Acquire B11 → dual drive → double Xπ → mux for frequency/duration/round.

        Preserve all raw shots and full tones, including partial cancellation.
        Caller must check context.scope.stop.raise_if_error(). No input mutation.
        """
        frequencies = sweep2array(config.frequency)
        drives = [
            DualLineCfg(
                f1=config.drive.f1.with_updates(freq=frequency),
                f2=config.drive.f2.with_updates(freq=frequency),
            )
            for frequency in frequencies
        ]
        return run_duration_scan(config, frequencies, drives, context=context)

    def analyze(
        self,
        source: RunRecord[ChevronCfg, RabiScanResult],
        options: None,
        *,
        plots: Plots,
    ) -> RabiAnalysis:
        """Classify complete points and plot return using saved cfg only.

        Fit complete duration traces with the undamped cosine, excluding rates
        below 1/saved duration span. Return diagnostic rates and fit_quality,
        never an automatic frequency/duration selection or pulse proposal.
        Missing source cfg, invalid raw or malformed axes raise ValueError.
        Incomplete points keep raw but are not classified or replaced by zero.
        """
        del options
        if source.cfg is None:
            raise ValueError(
                "Rabi analysis requires saved cfg; current Form/library is not a substitute"
            )
        answer = joint_return(
            source.cfg, source.result, sweep2array(source.cfg.frequency)
        )
        plot_return_data(
            source.result,
            answer,
            coordinate_label="Common drive frequency (MHz)",
            map_name="chevron",
            plots=plots,
        )
        rates = fit_duration_rates(source.result, answer)
        plot_duration_rates(
            source.result,
            rates,
            coordinate_label="Common drive frequency (MHz)",
            plots=plots,
        )
        return RabiAnalysis(answer, rates)
