from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, TimelineBuilder
from zcu_tools.utils.process import minus_background

from zcu_lab.v2.support import live_heatmap, live_line
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
    fail_retry: int = 0


class FreqFluxExp(Experiment[FreqFluxResult, FreqFluxCfg]):
    # inner freqs stores MHz on disk -> scale=1e6; outer values raw -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[FreqFluxResult, FreqFluxCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("values", label="Flux device value", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
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

        fig, _ = context.plots.subplots(1, 2, num='measurement')
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel='Flux device value', ylabel='Frequency (MHz)')
        line_ax.set_xlabel('Frequency (MHz)')
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        heat_ax.xaxis.set_major_formatter(formatter)
        heat_ax.yaxis.set_major_formatter(formatter)
        heatmap = live_heatmap(heat_ax, uniform=True)
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        line_ax.grid()
        lines = live_line(line_ax, num_lines=2)
        for artist in line_ax.lines[:-1]:
            artist.set(marker="None", alpha=0.3, color="red")
        line_ax.lines[-1].set(label="current line", marker=".", color="C0")
        line_ax.legend()
        
        @context.plots.on_plot_thread
        def update(data):
            real = freqflux_signal2real(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((2, len(freqs)), np.nan)
            for offset in range(min(2, current + 1)):
                history[-offset - 1] = real[current - offset]
            artists = heatmap.set(dev_values, freqs, real)
            artists = (*artists, *lines.set(freqs, history))
            return artists
        
        fig.show()
        signals_buffer = SignalBuffer(
            (len(dev_values), len(freqs)),
            on_update=update,
        )

        # Set the external flux bias before acquiring each complete spectrum.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("flux", dev_values.tolist()):
                set_flux(
                    context.devices[cfg.flux_dev],
                    step.value,
                    unit=cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )

                # Reset and initialize before each frequency-swept probe and readout.
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                freq_sweep = cfg.freq
                freq_param = builder.add_sweep("freq", freq_sweep)

                if cfg.reset is not None:
                    builder.emit(cfg.reset.build("reset"))
                builder.emit(Pulse("init_pulse", cfg.init_pulse))
                builder.emit(Pulse("qub_pulse", cfg.qub_pulse, freq=freq_param))
                builder.emit(Readout("readout", cfg.readout))
                prog = builder.build()
                step.run_acquire(
                    prog,
                    soc,
                    options=RunOptions(retry=cfg.fail_retry),
                    rounds=cfg.rounds,
                )

        return FreqFluxResult(
            values=dev_values,
            freqs=freqs,
            signals=signals_buffer.array,
        )
