from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.utils.process import minus_background

from zcu_lab.v2.support import live_heatmap_with_line
from zcu_lab.v2.support.device import set_flux


@dataclass(frozen=True)
class FreqFluxResult:
    """Flux-device values, frequencies in MHz, and mean IQ shaped (Nflux, Nfreq)."""

    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def freqflux_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class FreqFluxCfg(ConfigBase):
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
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Working unit for the selected driver; no conversion.",
        ),
    ]  # External flux driver unit; no conversion.
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Init Pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Probe Pulse")]
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]
    flux_dev: Annotated[
        DeviceName,
        FieldMeta(
            label="Flux Device",
            description="Name of the connected flux-bias device driven by the sweep.",
        ),
    ]
    flux: Annotated[SweepCfg, FieldMeta(label="Flux device value", decimals=6)]
    freq: Annotated[SweepCfg, FieldMeta(label="Frequency", unit="MHz")]
    fail_retry: int = 0


class FreqFluxExp(Experiment[FreqFluxResult, FreqFluxCfg]):
    # inner freqs stores MHz on disk -> scale=1e6; outer values raw -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[FreqFluxResult, FreqFluxCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),
            AxesSpec.Axis("values", "Flux device value", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
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

        value_sweep = cfg.flux
        freq_sweep = cfg.freq

        dev_values = sweep2array(value_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch}
        )

        viewer = live_heatmap_with_line(
            context.plots,
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
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("flux", dev_values.tolist()):
                cfg = step.cfg
                set_flux(
                    context.devices[cfg.flux_dev],
                    step.value,
                    unit=cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )

                # Reset and initialize before each frequency-swept probe and readout.
                freq_sweep = cfg.freq
                cfg.qub_pulse.set_param("freq", sweep2param("freq", freq_sweep))

                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add_reset("reset", cfg.reset)
                    .add_pulse("init_pulse", cfg.init_pulse)
                    .add_pulse("qubit_pulse", cfg.qub_pulse)
                    .add_readout("readout", cfg.readout)
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(
                        options=AcquireOptions(retry=cfg.fail_retry), rounds=cfg.rounds
                    )
                )

        return FreqFluxResult(
            values=dev_values,
            freqs=freqs,
            signals=signals_buffer.array,
        )
