from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils import set_flux
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import minus_background


@dataclass(frozen=True)
class FreqFluxResult:
    """Flux-device values, frequencies in MHz, and mean IQ shaped (Nflux, Nfreq)."""

    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def freqflux_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class FreqFluxSweepCfg(ConfigBase):
    flux: Annotated[SweepCfg, FieldMeta(label="Flux device value", decimals=6)]
    freq: Annotated[SweepCfg, FieldMeta(label="Freq", unit="MHz")]


class FreqFluxModuleCfg(ConfigBase):
    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[PulseCfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[PulseCfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[ReadoutCfg, FieldMeta(label="Readout")]


class FreqFluxCfg(ProgramV2Cfg, ExpCfgModel):
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Working unit for the selected driver; no conversion.",
        ),
    ]  # External flux driver unit; no conversion.
    modules: FreqFluxModuleCfg
    flux_dev: Annotated[
        str,
        FieldMeta(
            label="Flux Device",
            description="Name of the connected flux-bias device driven by the sweep.",
        ),
    ]
    sweep: FreqFluxSweepCfg
    fail_retry: int = 0


class FreqFluxExp(PersistableExperiment[FreqFluxResult, FreqFluxCfg]):
    # inner freqs stores MHz on disk -> scale=MHZ_TO_HZ; outer values raw -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("values", "Flux device value", "a.u.", scale=IDENTITY),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqFluxResult,
        cfg_type=FreqFluxCfg,
        tag="twotone/flux_dep/freq",
        cfg_schema_version="3.0",
    )

    def run(
        self,
        cfg: FreqFluxCfg,
        *,
        context: RunContext,
    ) -> FreqFluxResult:
        """Map qubit spectroscopy versus the selected device's flux bias.

        At each host-set flux value, reset, apply the optional init pulse,
        sweep the qubit probe frequency on the FPGA, then read out. Return
        mean complex IQ on the (flux, frequency) grid; no line is selected.
        ``flux_dev`` must name a context device accepting ``flux_unit``;
        values are passed in that unit without conversion. ``fail_retry``
        controls acquisition retries at each flux point.
        """
        # Prepare the device-value axis and the probe's hardware frequency grid.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules

        value_sweep = cfg.sweep.flux
        freq_sweep = cfg.sweep.freq

        dev_values = sweep2array(value_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch}
        )

        viewer = context.plots.liveplot_2d_with_line(
            "measurement",
            "Flux device value",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
        )
        signals_buffer = SignalBuffer(
            (len(dev_values), len(freqs)),
            on_update=lambda data: viewer.update(
                dev_values, freqs, freqflux_signal2real(data)
            ),
        )

        # Set the external flux bias before acquiring each complete spectrum.
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for _, step in sched.scan("flux", dev_values.tolist()):
                cfg = step.cfg
                set_flux(
                    context.devices[cfg.flux_dev],
                    step.value,
                    unit=cfg.flux_unit,
                    stop=context.cancel_signal,
                    progress=False,
                )
                modules = cfg.modules

                # Reset and initialize before each frequency-swept probe and readout.
                freq_sweep = cfg.sweep.freq
                modules.qub_pulse.set_param("freq", sweep2param("freq", freq_sweep))

                _ = (
                    step.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add_pulse("init_pulse", modules.init_pulse)
                    .add_pulse("qubit_pulse", modules.qub_pulse)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(retry=cfg.fail_retry)
                )

        return FreqFluxResult(
            values=dev_values,
            freqs=freqs,
            signals=signals_buffer.array,
        )
