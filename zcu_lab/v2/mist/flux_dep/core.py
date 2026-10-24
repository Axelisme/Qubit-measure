from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param
from zcu_tools.simulate import value2flux

from zcu_lab.v2.support import live_heatmap_with_line
from zcu_lab.v2.support.device import set_flux


@dataclass(frozen=True)
class FluxDepResult:
    """Flux device values, probe gains, and mean complex IQ [flux, gain]."""

    values: NDArray[np.float64]
    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]


def mist_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = np.abs(
        signals - np.mean(signals[:, :avg_len], axis=1, keepdims=True)
    )
    if np.all(np.isnan(mist_signals)):
        return mist_signals

    ref_signals = np.sort(mist_signals.flatten())[: int(0.5 * mist_signals.size)]
    return np.clip(mist_signals, 0, 10 * np.nanmedian(ref_signals))


@dataclass(frozen=True)
class FluxDepAnalyzeOptions:
    flux_half: float | None = None
    flux_period: float | None = None
    ac_coeff: float | None = None
    secondary_xaxis: bool = True
    auto_range: bool = True


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
    flux_unit: Literal["A", "V", "native"]  # External flux driver unit; no conversion.
    reset: Reset.Cfg | None = None
    init_pulse: Pulse.Cfg | None = None
    probe_pulse: Pulse.Cfg
    readout: Readout.Cfg
    flux_dev: Annotated[
        str,
        FieldMeta(
            label="Flux device",
            description="Connected flux device driven by the sweep.",
        ),
    ]
    flux: SweepCfg
    gain: SweepCfg


class FluxDepExp(Experiment[FluxDepResult, FluxDepCfg]):
    Options: ClassVar[type[FluxDepAnalyzeOptions]] = FluxDepAnalyzeOptions

    AXES_SPEC: ClassVar[AxesSpec[FluxDepResult, FluxDepCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("gains", "Power", "a.u."),
            AxesSpec.Axis("values", "Flux value", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=FluxDepResult,
        cfg_type=FluxDepCfg,
        tag="mist/flux_dep",
        cfg_schema_version="3.0",
    )

    def run(
        self,
        config: FluxDepCfg,
        *,
        context: RunContext,
    ) -> FluxDepResult:
        """Map the MIST readout response versus flux bias and probe gain.

        Step flux on the host. At each bias, reset, apply the optional init
        pulse, apply the gain-swept probe, then read out. The board sweeps
        probe gain. Return flux device values, hardware gain coordinates,
        and mean complex IQ shaped [flux, gain]. Changes in IQ reveal the
        probe-induced response; they are not calibrated state populations.

        ``flux_dev`` must name a device in ``context.devices``;
        ``flux_unit`` selects its working unit without conversion.
        """
        # 1. Prepare external flux values and the board's probe-gain coordinates.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg

        values = sweep2array(cfg.flux, allow_array=True)
        gains = sweep2array(
            cfg.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.probe_pulse.ch},
        )

        viewer = live_heatmap_with_line(
            context.plots,
            "measurement",
            "Flux device value",
            "Readout power (a.u.)",
            line_axis=1,
            num_lines=5,
            title="MIST over FLux",
        )
        signals_buffer = SignalBuffer(
            (len(values), len(gains)),
            on_update=lambda data: viewer.update(values, gains, mist_signal2real(data)),
        )

        # 2. Set each external bias before acquiring its full gain trace.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for flux, step in sched.scan("flux", values.tolist()):
                set_flux(
                    context.devices[step.cfg.flux_dev],
                    flux,
                    unit=step.cfg.flux_unit,
                    stop=context.scope.stop,
                    progress=False,
                )

                # 3. Reset, initialize, perturb with the probe, then measure IQ.
                modules = step.cfg
                modules.probe_pulse.set_param(
                    "gain", sweep2param("gain", step.cfg.gain)
                )
                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", modules.reset),
                        Pulse("init_pulse", modules.init_pulse),
                        Pulse("probe_pulse", modules.probe_pulse),
                        Readout("readout", modules.readout),
                    )
                    .declare_sweep("gain", step.cfg.gain)
                    .build_and_acquire(rounds=cfg.rounds)
                )

        # 4. Preserve raw complex response; baseline subtraction is display-only.
        return FluxDepResult(values=values, gains=gains, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[FluxDepCfg, FluxDepResult],
        options: FluxDepAnalyzeOptions,
        *,
        plots: Plots,
    ) -> None:
        """Plot the baseline-subtracted MIST magnitude map from ``source``.

        Each flux row uses its first 5% of gains (at least one) as a mean
        IQ reference; display clipping limits strong outliers. ``ac_coeff``
        maps gain squared to photon number when supplied. Flux conversion
        uses ``flux_half`` and ``flux_period`` together; requesting the
        secondary device-value axis without both raises ValueError.
        No state-population fit or calibration value is returned.
        """
        result = source.result
        flux_half, flux_period = options.flux_half, options.flux_period
        if options.secondary_xaxis and (flux_half is None or flux_period is None):
            raise ValueError("Secondary flux axis requires flux_half and flux_period")

        dev_values, gains, signals = result.values, result.gains, result.signals
        if flux_half is not None and flux_period is not None:
            xs = np.asarray(value2flux(dev_values, flux_half, flux_period))
            xlabel = r"$\phi$ (a.u.)"
        else:
            xs = dev_values
            xlabel = "Flux device value"

        amp_diff = mist_signal2real(signals)
        if options.ac_coeff is None:
            ys = gains
            ylabel = "probe gain (a.u.)"
        else:
            ys = options.ac_coeff * gains**2
            ylabel = r"$\bar n$"
        ys = np.asarray(ys)

        _, ax = plots.subplots("fit", figsize=config.figsize)
        ax.pcolormesh(xs, ys, amp_diff.T, shading="nearest", cmap="Greys")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)

        if options.secondary_xaxis:
            # Match the source samples and scientific labels of the former overlay.
            count = len(xs)
            if count <= 12:
                tick_indices = np.arange(count)
            else:
                tick_indices = np.unique(
                    np.round(np.linspace(0, count - 1, 12)).astype(int)
                )
            secondary = ax.secondary_xaxis("top")
            secondary.set_xticks(
                xs[tick_indices],
                labels=[f"{value:.1e}" for value in dev_values[tick_indices]],
            )

        if options.auto_range:
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])
