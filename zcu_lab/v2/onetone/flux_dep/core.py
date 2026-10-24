from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import PulseReadout, Reset, SweepCfg, sweep2param

from zcu_lab.v2.support import live_heatmap_with_line
from zcu_lab.v2.support.device import set_flux


@dataclass(frozen=True)
class FluxDepResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FluxDepCfg(ConfigBase):
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
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    flux_dev: Annotated[
        str,
        FieldMeta(
            label="Flux Device",
            description="Name of the connected flux-bias device driven by the sweep.",
        ),
    ]
    flux_unit: Annotated[
        Literal["A", "V", "native"],
        FieldMeta(
            label="Flux unit",
            description="Working unit for the selected flux driver; no conversion.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Frequency",
            unit="MHz",
            description="Readout frequency sweep.",
        ),
    ]
    flux: Annotated[
        SweepCfg,
        FieldMeta(
            label="Flux device value",
            description="Flux device sweep in flux_unit coordinates.",
        ),
    ]


class FluxDepExp(Experiment[FluxDepResult, FluxDepCfg]):
    # inner axis (fastest-varying) = freqs (MHz in memory, Hz on disk);
    # outer axis = flux device values (a.u.).
    AXES_SPEC: ClassVar[AxesSpec[FluxDepResult, FluxDepCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),
            AxesSpec.Axis("values", "Flux device value", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=FluxDepResult,
        cfg_type=FluxDepCfg,
        tag="onetone/flux_dep",
        cfg_schema_version="3.0",
    )

    def run(
        self,
        config: FluxDepCfg,
        *,
        context: RunContext,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> FluxDepResult:
        """Map the resonator response versus flux bias and readout frequency.

        At each host-side flux value, apply optional reset and sweep the
        readout tone frequency on the board. Return flux-device values in
        cfg.flux_unit, frequencies in MHz, and mean complex IQ shaped
        (Nflux, Nfreq). The map reveals resonance tracks; line selection and
        flux calibration are separate analysis steps.

        cfg.flux_dev must name a connected device supporting cfg.flux_unit.
        acquire_kwargs forwards optional acquisition controls to the builder.
        """

        # Prepare the device coordinates and the matched generator/ADC grid.
        soc, soccfg = context.soc, context.soccfg
        cfg = deepcopy(config)
        modules = cfg
        freq_sweep = cfg.freq
        flux_sweep = cfg.flux

        dev_values = sweep2array(flux_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": modules.readout.gen_ch,
                "ro_ch": modules.readout.ro_ch,
            },
        )

        # Move to the first bias with progress before collecting the map.
        set_flux(
            context.devices[cfg.flux_dev],
            dev_values[0],
            unit=cfg.flux_unit,
            stop=context.scope.stop,
            progress=True,
        )

        # Display magnitude, but retain complex IQ for each flux/frequency point.
        viewer = live_heatmap_with_line(
            context.plots,
            "measurement",
            "Flux device value",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=10,
            uniform=False,
        )
        signals_buffer = SignalBuffer(
            (len(dev_values), len(freqs)),
            on_update=lambda data: viewer.update(
                dev_values,
                freqs,
                fluxdep_signal2real(data),
            ),
        )
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
                modules = cfg

                freq_sweep = cfg.freq
                modules.readout.set_param("freq", sweep2param("freq", freq_sweep))

                # Reset at each frequency, then excite and measure the resonator.
                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add_reset("reset", modules.reset)
                    .add(PulseReadout("readout", modules.readout))
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(**(acquire_kwargs or {}), rounds=cfg.rounds)
                )

        # Publish the final map without treating an acquisition error as success.
        signals_buffer.trigger_update(flush=True)
        sched.stop.raise_if_error()
        return FluxDepResult(
            values=dev_values,
            freqs=freqs,
            signals=signals_buffer.array,
        )
