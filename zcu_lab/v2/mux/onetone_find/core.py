"""Single-tone absolute-frequency survey on a mux readout generator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Self

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import StrictInt, field_validator, model_validator
from qick import QickConfig
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_freq, sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg


@dataclass(frozen=True)
class MuxOnetoneFindResult:
    """One spectrum, including partially acquired scans.

    freqs is float64 (P,) of quantized absolute frequencies in MHz, including
    unmeasured points. signals is complex128 (P,) of averaged I+jQ in a.u.,
    using the program's ADC-length normalization. Unacquired entries remain
    NaN; acquired points survive cancellation.
    """

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


DEFAULT_MIN_SEPARATION_MHZ = 1.0


@dataclass
class MuxOnetoneFindOptions:
    """Choose frequency-ordered names and minimum spacing for coarse dip detection.

    resonators must be a nonempty unique list of nonblank component-name strings.
    Names map to selected dips in ascending MHz order, not to original tones.
    min_separation is finite positive MHz. The default 1 MHz separates typical
    MHz readout linewidths; increase it for broader resonators. Validation occurs
    on Analyze, allowing an empty initial GUI list to be edited before use.
    """

    resonators: list[str]
    min_separation: float = DEFAULT_MIN_SEPARATION_MHZ

    def validate(self) -> None:
        """Reject empty/duplicate/non-string names and nonpositive/nonfinite MHz."""
        _validate_options(self.resonators, self.min_separation)


def _validate_options(resonators: object, min_separation: object) -> None:
    # GUI editors and notebook callers can supply values outside annotations.
    if (
        not isinstance(resonators, list)
        or not resonators
        or any(not isinstance(name, str) or not name.strip() for name in resonators)
    ):
        raise ValueError("resonators must be a nonempty list of nonblank strings")
    if len(set(resonators)) != len(resonators):
        raise ValueError("resonators must contain unique names")
    if (
        isinstance(min_separation, bool)
        or not isinstance(min_separation, (int, float))
        or not math.isfinite(min_separation)
        or min_separation <= 0
    ):
        raise ValueError("min_separation must be finite and > 0 MHz")


@dataclass(frozen=True)
class ResonatorDip:
    """One named dip: freq is absolute MHz; depth is magnitude prominence in a.u.

    name comes from Options in frequency order, not from a tone label. depth
    uses the background-subtracted, Gaussian-smoothed magnitude, not a fit.
    """

    name: str
    freq: float
    depth: float


def analyze(
    source: RunRecord[MuxOnetoneFindCfg, MuxOnetoneFindResult],
    options: MuxOnetoneFindOptions,
    *,
    plots: Plots,
) -> tuple[ResonatorDip, ...]:
    """Remove linear magnitude background and select N most prominent interior dips.

    source supplies matching (P,) absolute MHz and complex IQ arrays, P >= 11;
    cfg may be None. Partial/nonfinite scans and duplicate frequencies raise
    ValueError. Descending scans are sorted. No current Form or capture is read.

    Background joins median magnitude/frequency in each outer 10% of points
    (at least two per edge). Gaussian smoothing uses sigma=1 sample. Prominence
    must exceed five adjacent-difference noise estimates (MAD/0.67449/sqrt(2)),
    with a 1e-6 magnitude-range floor. Greedy descending prominence enforces
    min_separation in actual MHz; frequency-sorted dips receive the given names.
    Edge minima are excluded. depth is prominence in a.u., not linewidth.

    plots receives raw/background and corrected curves, labeled name/freq/depth.
    Return all requested dips or raise ValueError naming the available count;
    no partial success or proposals are produced here.
    """
    options.validate()
    freqs = np.asarray(source.result.freqs, dtype=np.float64)
    signals = np.asarray(source.result.signals, dtype=np.complex128)
    if freqs.ndim != 1 or signals.shape != freqs.shape or freqs.size < 11:
        raise ValueError(
            "Analysis needs matching (P,) frequency/IQ arrays with P >= 11"
        )
    if not np.all(np.isfinite(freqs)) or not np.all(np.isfinite(signals)):
        raise ValueError(
            "Analysis needs a complete finite scan; partial/NaN points remain"
        )
    order = np.argsort(freqs)
    freqs, magnitude = freqs[order], np.abs(signals[order])
    if np.any(np.diff(freqs) <= 0):
        raise ValueError("Analysis needs distinct absolute frequencies")

    # Edge medians leave the interior dips out of the linear background estimate.
    edge = max(2, freqs.size // 10)
    x0, x1 = float(np.median(freqs[:edge])), float(np.median(freqs[-edge:]))
    y0, y1 = float(np.median(magnitude[:edge])), float(np.median(magnitude[-edge:]))
    background = y0 + (y1 - y0) * (freqs - x0) / (x1 - x0)
    corrected = magnitude - background
    difference = np.diff(corrected)
    noise = float(np.median(np.abs(difference - np.median(difference)))) / (
        0.67449 * np.sqrt(2)
    )
    threshold = max(5 * noise, 1e-6 * float(np.ptp(magnitude)), np.finfo(float).eps)
    smoothed = gaussian_filter1d(corrected, sigma=1.0)
    candidates, properties = find_peaks(-smoothed, prominence=threshold)
    prominences = properties["prominences"]

    # Enforce actual MHz separation rather than distance in samples.
    selected: list[int] = []
    for candidate in np.argsort(-prominences, kind="stable"):
        index = int(candidates[candidate])
        if all(
            abs(freqs[index] - freqs[other]) >= options.min_separation
            for other in selected
        ):
            selected.append(index)
            if len(selected) == len(options.resonators):
                break
    if len(selected) < len(options.resonators):
        raise ValueError(
            f"Found {len(selected)} significant separated dips; need {len(options.resonators)} "
            f"(min_separation={options.min_separation:g} MHz, prominence >= {threshold:.6g} a.u.). "
            "Check scan range, resolution, power and requested names."
        )
    depth_by_index = {
        int(index): float(depth)
        for index, depth in zip(candidates, prominences, strict=True)
    }
    dips = tuple(
        ResonatorDip(name, float(freqs[index]), depth_by_index[index])
        for name, index in zip(options.resonators, sorted(selected), strict=True)
    )

    plots.subplots("fit", nrows=2, sharex=True, figsize=(9, 7))

    def draw(figure: Figure) -> None:
        raw_ax, dip_ax = figure.axes
        raw_ax.plot(freqs, magnitude, label="Measured magnitude")
        raw_ax.plot(freqs, background, label="Linear edge background")
        raw_ax.set_ylabel("Magnitude (a.u.)")
        raw_ax.legend()
        dip_ax.plot(freqs, smoothed, label="Background removed")
        for dip in dips:
            dip_ax.axvline(
                dip.freq,
                linestyle="--",
                label=f"{dip.name}: {dip.freq:.6g} MHz, depth {dip.depth:.4g}",
            )
        dip_ax.set_xlabel("Frequency (MHz)")
        dip_ax.set_ylabel("Corrected magnitude (a.u.)")
        dip_ax.legend()
        figure.tight_layout()

    plots.draw("fit", draw)
    return dips


class MuxOnetoneFindModuleCfg(ConfigBase):
    """Full mux readout settings, retaining inactive tones for saved provenance.

    readout contains 1..8 ordered tones with absolute MHz frequency, normalized
    gain, phase in degrees and logical ro_ch. Only the selected tone is played.
    Shared playback/ADC timing and wiring use MuxReadoutCfg's validation.
    """

    readout: Annotated[MuxReadoutCfg, FieldMeta(label="Readout")]


class MuxOnetoneFindSweepCfg(ConfigBase):
    """Absolute-frequency host scan in MHz, with finite start, stop and step.

    freq is SweepCfg(start, stop, expts, step). SweepCfg requires positive expts
    and consistent endpoints; a single point has start=stop and step=0.
    Invalid/nonfinite values raise at construction or assignment.
    """

    freq: Annotated[SweepCfg, FieldMeta(label="Frequency", unit="MHz")]

    @field_validator("freq")
    @classmethod
    def _finite_frequency(cls, value: SweepCfg) -> SweepCfg:
        if not all(math.isfinite(v) for v in (value.start, value.stop, value.step)):
            raise ValueError("freq start, stop and step must be finite MHz values")
        return value


class MuxOnetoneFindCfg(ProgramV2Cfg, ExpCfgModel):
    """Sweep one original mux tone through absolute frequencies.

    modules.readout retains the full ordered tone list. tone is its original
    zero-based index, defaults to 0, and must be a strict integer in 0..N-1.
    The selected tone supplies gain, phase and ro_ch; sweep.freq replaces only
    its frequency at each point. Other tones are never declared or played.

    Inherited reps and rounds default to 1 and average repetitions/software
    rounds per point. initial_delay and relax_delay default to 1 us. No reset
    or external devices are used. Hardware-band validation is performed by
    prepare_frequencies before acquisition, and by the GUI preflight hook.
    """

    modules: MuxOnetoneFindModuleCfg
    sweep: MuxOnetoneFindSweepCfg
    tone: Annotated[
        StrictInt,
        FieldMeta(
            label="Tone",
            description="Original readout tone index; selects gain, phase and ADC.",
        ),
    ] = 0

    @model_validator(mode="after")
    def _valid_tone(self) -> Self:
        if not 0 <= self.tone < len(self.modules.readout.tones):
            raise ValueError(
                f"tone must be in 0..{len(self.modules.readout.tones) - 1} "
                "of the full readout"
            )
        return self


def prepare_frequencies(
    cfg: MuxOnetoneFindCfg, *, soccfg: QickConfig
) -> NDArray[np.float64]:
    """Validate the mux band and return quantized absolute MHz with shape (P,).

    cfg supplies the full readout, original tone index and requested sweep.
    soccfg is the cached QICK hardware configuration, not a connected SoC.
    The allowed closed interval is the selected-ADC-rounded mixer frequency
    plus/minus half the generator DDS bandwidth. Out-of-band requested points
    raise ValueError naming sweep.freq, the first offending point and allowed
    MHz interval; Nyquist images are not accepted as substitutes. QICK performs
    DAC/ADC grid matching and propagates hardware/quantization errors.

    This pure preparation performs no acquisition or source mutation. The
    adapter calls it in validate_run_request; run calls it before building
    any programs. The returned axis includes unacquired points after a stop.
    """
    readout = cfg.modules.readout
    tone = readout.tones[cfg.tone]
    freqs = sweep2array(cfg.sweep.freq, "none")
    mixer = soccfg.calc_mixer_freq(
        readout.gen_ch, readout.mixer_freq, readout.nqz, tone.ro_ch
    )["rounded"]
    half_bandwidth = soccfg["gens"][readout.gen_ch]["f_dds"] / 2
    lower, upper = mixer - half_bandwidth, mixer + half_bandwidth
    outside = np.flatnonzero((freqs < lower) | (freqs > upper))
    if outside.size:
        index = int(outside[0])
        raise ValueError(
            f"sweep.freq[{index}]={freqs[index]:.12g} MHz is outside generator "
            f"{readout.gen_ch} mux band [{lower:.12g}, {upper:.12g}] MHz "
            f"(rounded mixer {mixer:.12g} MHz ± {half_bandwidth:.12g} MHz)"
        )
    return round_zcu_freq(
        freqs,
        soccfg,
        readout.gen_ch,
        tone.ro_ch,
        mixer_freq=readout.mixer_freq,
    )


def _single_tone_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder initializes its own module copy, not the caller's MuxReadout.
    for module in prog.modules:
        if module.name == "readout" and isinstance(module, MuxReadout):
            iq = module.collect_iq(prog, acquired, read_index=0)
            return np.asarray(iq[0, 0] + 1j * iq[0, 1], dtype=np.complex128)
    raise ValueError("Mux program is missing its named readout MuxReadout")


class MuxOnetoneFindExp(PersistableExperiment[MuxOnetoneFindResult, MuxOnetoneFindCfg]):
    """Measure a broad spectrum with exactly one mux tone and its ADC per point.

    Each host frequency builds a new readout-only program. Native persistence
    stores the complete cfg and complex IQ; the frequency axis is MHz in memory
    and Hz on disk. analyze identifies coarse dips with typed Options;
    writeback destinations belong to the GUI adapter.
    """

    Options = MuxOnetoneFindOptions

    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=MuxOnetoneFindResult,
        cfg_type=MuxOnetoneFindCfg,
        cfg_schema_version="1.0",
        tag="mux/onetone_find",
    )

    def run(
        self, config: MuxOnetoneFindCfg, *, context: RunContext
    ) -> MuxOnetoneFindResult:
        """Play the selected tone and acquire its ADC at every absolute MHz point.

        config is unchanged. context supplies soc/soccfg, plots and cancellation;
        no devices are required. Invalid band settings fail before any I/O.
        The live measurement plot has one magnitude curve against quantized MHz.
        Return (P,) complex IQ averaged over reps and rounds, with acquired
        points preserved and unacquired points NaN after cancellation. The caller
        calls context.cancel_signal.raise_if_error() to distinguish acquisition
        failure from a stopped/successful return.
        """
        # Prepare the entire hardware grid before starting acquisition.
        cfg = deepcopy(config)
        freqs = prepare_frequencies(cfg, soccfg=context.soccfg)
        base_readout = cfg.modules.readout
        selected = base_readout.tones[cfg.tone]

        viewer = context.plots.liveplot_1d(
            "measurement", "Frequency (MHz)", "Magnitude (a.u.)"
        )
        signals_buffer = SignalBuffer(
            freqs.shape, on_update=lambda data: viewer.update(freqs, np.abs(data))
        )

        # Redeclare exactly one DAC tone and its original ADC at each frequency.
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for freq, step in sched.scan("freq", freqs.tolist()):
                readout = base_readout.with_updates(
                    tones=[selected.with_updates(freq=freq)]
                )
                builder = step.prog_builder(context.soc, context.soccfg).add(
                    MuxReadout("readout", readout)
                )
                prog = builder.build()
                _ = builder.run_program(
                    prog, raw2signal_fn=partial(_single_tone_signal, prog)
                )

        signals_buffer.trigger_update(flush=True)
        return MuxOnetoneFindResult(freqs=freqs, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[MuxOnetoneFindCfg, MuxOnetoneFindResult],
        options: MuxOnetoneFindOptions,
        *,
        plots: Plots,
    ) -> tuple[ResonatorDip, ...]:
        """Find N coarse dips after linear-background removal and Gaussian smoothing.

        source supplies matching (P,) absolute MHz and complex IQ arrays in a.u.,
        with P >= 11; cfg may be None. options supplies frequency-ordered names
        and positive minimum separation in MHz. Return one ResonatorDip per
        name, ordered by ascending frequency, with freq in MHz and prominence
        depth in a.u. Publish raw/background and annotated corrected curves to
        plots as 'fit'. No hardware, current Form, capture or writeback is used;
        source and options are unchanged, and no analysis state is retained.

        ValueError rejects invalid options, mismatched/short arrays, partial or
        nonfinite scans, duplicate frequencies and fewer than N separated dips.
        The module-level analyze documents the shared dip-selection policy.
        """
        return analyze(source, options, plots=plots)
