"""Single-carrier, two-photon g/f scans using acquired reference and signal IQ."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from qick import QickConfig
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.core.waveform_format import decode_npz
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_freq, round_zcu_gain
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import GainSweep, Pulse, PulseReadout, SweepCfg
from zcu_tools.program.v2.modules.waveform import ArbWaveformCfg

from zcu_lab.v2.support.singleshot.util import raw_shots_to_signal


@dataclass(frozen=True)
class RamanGFResult:
    """Mean-sample ADC IQ, with no simulator state information.

    signals: (points,shots) complex IQ in detuning-major, gain-minor order.
    references: (3,shots) IQ for g/no pulse, e/pi01, f/pi03 then pi32.
    detunings: (points,) actual two-photon detuning in MHz, positive above center.
    areas: (points,) gain squared times sampled envelope intensity integral, us.
    gains: (points,) actual nonnegative DAC gains. Areas are not rotation angles.
    Partial/unacquired rows remain NaN. References describe preparations, not purity.
    """

    signals: NDArray[np.complex128]
    references: NDArray[np.complex128]
    detunings: NDArray[np.float64]
    areas: NDArray[np.float64]
    gains: NDArray[np.float64]


class RamanGFCfg(ConfigBase):
    """One charge-drive carrier with a sampled sqrt(cos)-type envelope.

    carrier_mhz is operator-supplied (f03-f32)/2, not a model frequency.
    detuning = 2*f_drive - 2*carrier_mhz, in MHz (cycles/us, not angular units).
    Frequency mode uses gain.start; area mode uses detuning.start; grid uses both.
    Gain sweeps change the two-photon area gain^2*integral(|envelope|^2 dt).
    The drive's freq/gain are supplied by the scan; its channel, NQZ, waveform,
    phase and delays remain operator inputs. pi01/pi03/pi32 calibrate references.
    """

    carrier_mhz: Annotated[
        float,
        FieldMeta(
            label="Carrier center",
            unit="MHz",
            description="Enter (measured f03 - measured f32)/2. Never inferred from the model.",
        ),
    ] = Field(gt=0, allow_inf_nan=False)
    scan_mode: Annotated[
        Literal["frequency", "area", "frequency_area"],
        FieldMeta(
            label="Scan mode",
            description="Frequency holds gain at its start; area holds detuning at its start; frequency_area scans both.",
        ),
    ] = "frequency_area"
    detuning: Annotated[SweepCfg, FieldMeta(label="Two-photon detuning", unit="MHz")]
    gain: Annotated[GainSweep, FieldMeta(label="Area via drive gain", unit="a.u.")]
    drive: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Raman drive",
            description="Select a sampled sqrt(cos) arbitrary waveform. Frequency and gain come from the scan.",
        ),
    ]
    pi01: Annotated[Pulse.Cfg, FieldMeta(label="Pi 01")]
    pi03: Annotated[Pulse.Cfg, FieldMeta(label="Pi 03")]
    pi32: Annotated[Pulse.Cfg, FieldMeta(label="Pi 32")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    shots: Annotated[int, FieldMeta(label="Shots per point")] = Field(
        default=1024, ge=6
    )
    relax_delay: Annotated[float, FieldMeta(label="Relax delay", unit="us")] = Field(
        default=1, ge=0
    )
    initial_delay: Annotated[float, FieldMeta(label="Initial delay", unit="us")] = (
        Field(default=1, ge=0)
    )


def scan_points(
    cfg: RamanGFCfg, soccfg: QickConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Return flat carrier MHz, DAC gains and envelope intensity integral in us.

    Points are detuning-major and gain-minor. Quantize scalar carriers/gains
    independently using the selected generator and mixer. Reject a non-arbitrary
    or zero-intensity envelope, and negative area-scan gains, before acquisition.
    The integral describes source samples, not a calibrated rotation angle.
    """
    if not isinstance(cfg.drive.waveform, ArbWaveformCfg):
        raise ValueError(
            "Raman drive requires a sampled arbitrary sqrt(cos)-type envelope"
        )
    if min(cfg.gain.start, cfg.gain.stop) < 0:
        raise ValueError("Raman area scan requires nonnegative drive gains")
    samples = decode_npz(Path(cfg.drive.waveform.path).read_bytes()).samples
    intensity = samples.idata**2
    if samples.qdata is not None:
        intensity = intensity + samples.qdata**2
    integral = float(np.trapezoid(intensity, samples.time))
    if integral <= 0:
        raise ValueError("Raman envelope must have nonzero integrated intensity")
    detunings = np.linspace(cfg.detuning.start, cfg.detuning.stop, cfg.detuning.expts)
    gains = np.linspace(cfg.gain.start, cfg.gain.stop, cfg.gain.expts)
    if cfg.scan_mode == "area":
        detunings = detunings[:1]
    elif cfg.scan_mode == "frequency":
        gains = gains[:1]
    frequencies = round_zcu_freq(
        cfg.carrier_mhz + detunings / 2,
        soccfg,
        cfg.drive.ch,
        mixer_freq=cfg.drive.mixer_freq,
    )
    gains = round_zcu_gain(gains, soccfg, cfg.drive.ch)
    return (
        np.repeat(frequencies, len(gains)),
        np.tile(gains, len(frequencies)),
        integral,
    )


def restore_result(data: GroupedLoadData[RamanGFCfg]) -> RamanGFResult:
    """Restore all acquired IQ and actual coordinates without requiring a valid cfg."""
    return RamanGFResult(
        signals=np.asarray(data.variable("signal").z, dtype=np.complex128),
        references=np.asarray(data.variable("reference").z, dtype=np.complex128),
        detunings=np.asarray(data.variable("detuning").z, dtype=np.float64),
        areas=np.asarray(data.variable("area").z, dtype=np.float64),
        gains=np.asarray(data.variable("gain").z, dtype=np.float64),
    )


class RamanGFExp(Experiment[RamanGFResult, RamanGFCfg]):
    """Acquire three references and scan a single-tone two-photon g/f drive."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[RamanGFResult, RamanGFCfg]] = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "signal",
                (
                    VariableAxisSpec.generated_arange("Shot", "None"),
                    VariableAxisSpec.generated_arange("Point", "None"),
                ),
                VariableZSpec("signals", "Signal IQ", "ADC", dtype=np.complex128),
            ),
            VariableSpec(
                "reference",
                (
                    VariableAxisSpec.generated_arange("Shot", "None"),
                    VariableAxisSpec.generated_arange("Preparation", "None"),
                ),
                VariableZSpec("references", "Reference IQ", "ADC", dtype=np.complex128),
            ),
            *(
                VariableSpec(
                    name,
                    (VariableAxisSpec.generated_arange("Point", "None"),),
                    VariableZSpec(field, label, unit, dtype=np.float64),
                )
                for name, field, label, unit in (
                    ("detuning", "detunings", "Two-photon detuning", "MHz"),
                    ("area", "areas", "Two-photon area", "gain^2 us"),
                    ("gain", "gains", "Drive gain", "a.u."),
                )
            ),
        ),
        result_type=RamanGFResult,
        cfg_type=RamanGFCfg,
        result_builder=restore_result,
        tag="twotone/raman_gf",
        cfg_schema_version="3.0",
        labber_layout="independent",
    )

    def run(self, cfg: RamanGFCfg, *, context: RunContext) -> RamanGFResult:
        """Acquire references, then one Raman pulse and readout per scan point.

        Positive two-photon detuning raises the carrier by detuning/2 MHz.
        Each point starts after relaxation; there is no Lambda pair or reset.
        Use one round of raw shots through legacy Module/Schedule acquisition.
        Require an arbitrary sampled envelope and nonnegative gains before I/O.
        Returned areas use the source samples' intensity integral; QICK timing
        interpolation/quantization is not an experimental rotation calibration.
        """
        cfg = deepcopy(cfg)
        point_frequencies, point_gains, integral = scan_points(cfg, context.soccfg)
        signals = SignalBuffer((len(point_gains), cfg.shots))
        references = SignalBuffer((3, cfg.shots))

        # Reference labels are requested preparations, never simulator populations.
        with Schedule(cfg, references, scope=context.scope) as sched:
            for label, step in sched.scan("preparation", [0, 1, 2]):
                builder = step.prog_builder(
                    context.soc,
                    context.soccfg,
                    reps=cfg.shots,
                    initial_delay=cfg.initial_delay,
                    relax_delay=cfg.relax_delay,
                )
                if label == 1:
                    builder.add(Pulse("pi01", cfg.pi01))
                elif label == 2:
                    builder.add(Pulse("pi03", cfg.pi03), Pulse("pi32", cfg.pi32))
                program = builder.add(PulseReadout("readout", cfg.readout)).build()
                try:
                    program.acquire(context.soc, progress=True, cancel_flag=step.stop)
                except StoppedPartialAcquireError:
                    step.set_stop()
                    break
                references[step].set(raw_shots_to_signal(program))

        # One carrier and one envelope, immediately followed by the same readout.
        with Schedule(cfg, signals, scope=context.scope) as sched:
            for point, step in sched.scan("point", range(len(point_gains))):
                cfg.drive.freq = float(point_frequencies[point])
                cfg.drive.gain = float(point_gains[point])
                program = (
                    step.prog_builder(
                        context.soc,
                        context.soccfg,
                        reps=cfg.shots,
                        initial_delay=cfg.initial_delay,
                        relax_delay=cfg.relax_delay,
                    )
                    .add(
                        Pulse("raman", cfg.drive), PulseReadout("readout", cfg.readout)
                    )
                    .build()
                )
                try:
                    program.acquire(context.soc, progress=True, cancel_flag=step.stop)
                except StoppedPartialAcquireError:
                    step.set_stop()
                    break
                signals[step].set(raw_shots_to_signal(program))

        result = RamanGFResult(
            signals=signals.array,
            references=references.array,
            detunings=2 * (point_frequencies - cfg.carrier_mhz),
            areas=point_gains**2 * integral,
            gains=point_gains,
        )
        context.plots.subplots(
            "measurement", ncols=2, figsize=(10, 4), layout="constrained"
        )

        def draw(figure):
            ax, reference_ax = figure.axes
            ax.plot(
                np.arange(len(result.signals)),
                result.signals.mean(axis=1).real,
                "o-",
                label="Mean I",
            )
            ax.plot(
                np.arange(len(result.signals)),
                result.signals.mean(axis=1).imag,
                "o-",
                label="Mean Q",
            )
            ax.set(
                xlabel="Point (detuning-major, area-minor)", ylabel="Mean-sample ADC IQ"
            )
            ax.legend()
            for label, row in zip(
                ("g: no pulse", "e: pi01", "f: pi03 then pi32"),
                result.references,
                strict=True,
            ):
                reference_ax.scatter(row.real, row.imag, s=2, alpha=0.3, label=label)
            reference_ax.set(
                xlabel="I (ADC)", ylabel="Q (ADC)", title="Acquired references"
            )
            reference_ax.legend()

        context.plots.draw("measurement", draw)
        return result
