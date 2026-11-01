"""Calibration acquisition owned by workflows.

Each runner borrows context's SoC, plots and cancellation scope, copies the native
cfg and returns core-compatible Result data without fitting or saving. retry is
the workflow's transient acquisition retry count. Cooperative cancellation returns
partial data. Producer errors propagate unchanged after retry is exhausted.
Converter and update errors propagate without retry. The cancellation token does
not store errors.
"""

from copy import deepcopy

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal, snr_checker
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    TimelineBuilder,
    TimelineProgramV2,
    sweep2param,
)
from zcu_tools.utils.process import minus_background, rotate2real

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqResult
from zcu_lab.v2.twotone.rabi.amp_rabi.core import AmpRabiCfg, AmpRabiResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiCfg, LenRabiResult
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainCfg, FreqGainResult


def qubfreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return magnitude contrast after component-wise median background removal.

    Accept a nonempty complex128 spectrum of shape (N,). Preserve NaNs in partial
    spectra without changing the input. Raise ValueError for another shape or
    dtype, an empty spectrum, or infinities.
    """
    if signals.ndim != 1 or signals.size == 0 or signals.dtype != np.complex128:
        raise ValueError("Expected a nonempty complex128 spectrum of shape (N,)")
    if np.any(np.isinf(signals)):
        raise ValueError("Frequency spectrum must not contain infinities")
    return np.abs(minus_background(signals))


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Rotate complex IQ to its principal real axis, preserving shape and NaNs."""
    return rotate2real(signals).real


def run_freq(
    cfg: FreqCfg,
    *,
    context: RunContext,
    retry: int,
) -> FreqResult:
    """Locate a qubit transition by sweeping the drive frequency.

    Apply reset, optional init_pulse, the frequency-swept qubit probe,
    then readout. None init_pulse emits no pulse. Return mean complex IQ
    at each frequency in MHz, not a fitted transition frequency. The live
    preview always uses background-subtracted magnitude (``lor``), independent
    of the model selected later for analyze.
    """
    # Prepare the frequency axis and display the background-subtracted contrast.
    cfg = deepcopy(cfg)
    soc, soccfg = context.soc, context.soccfg

    # predicted sweep points before FPGA coercion
    freqs = sweep2array(
        cfg.freq, "freq", {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch}
    )

    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Frequency (MHz)", ylabel="Amplitude")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid()
    viewer = live_line(ax)

    @context.plots.on_plot_thread
    def update(data):
        return viewer.set(freqs, qubfreq_signal2real(data))

    fig.show()
    signals_buffer = SignalBuffer(
        (len(freqs),),
        on_update=update,
    )

    with Schedule(signals_buffer, scope=context.scope) as sched:
        builder = TimelineBuilder(soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay)
        freq_sweep = cfg.freq
        freq_param = builder.add_sweep("freq", freq_sweep)

        # Reset and initialize before the FPGA-swept probe, then read out.

        if cfg.reset is not None:
            builder.emit(cfg.reset.build("reset"))
        builder.emit(Pulse("init_pulse", cfg.init_pulse))
        builder.emit(Pulse("qub_pulse", cfg.qub_pulse, freq=freq_param))
        builder.emit(Readout("readout", cfg.readout))
        prog = builder.build()
        sched.run_acquire(
            prog,
            soc,
            rounds=cfg.rounds,
            options=RunOptions(
                retry=retry,
                stop_condition=snr_checker(
                    signals_buffer.at(),
                    cfg.earlystop_snr,
                    lambda data: rotate2real(data).real,
                ),
            ),
        )

    return FreqResult(
        freqs=freqs,
        signals=signals_buffer.array,
    )


def run_len_rabi(
    cfg: LenRabiCfg,
    *,
    context: RunContext,
    retry: int,
) -> LenRabiResult:
    """Measure Rabi oscillations by sweeping drive length at fixed gain.

    Apply optional reset and initialization, then the length-swept qubit
    pulse and readout. Const/flat-top pulses sweep on the FPGA; other
    waveforms scan on the host and reuse each length's compiled program.
    Return quantized lengths in us and mean complex IQ at each length,
    retaining acquired data on cancellation and NaN for unmeasured points.
    retry sets the transient acquisition retry count.
    Reject native arb with ValueError before preparing coordinates or plots:
    its asset-derived length cannot be swept by this experiment.
    """
    if cfg.qub_pulse.waveform.style == "arb":
        raise ValueError(
            "Len Rabi cannot sweep an arb waveform's asset-derived length. "
            "Select a waveform with an editable length instead."
        )

    # 1. Prepare length coordinates without changing the caller's cfg.
    cfg = deepcopy(cfg)
    soccfg = context.soccfg
    lengths = sweep2array(
        cfg.length,
        "time",
        {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
    )

    if cfg.qub_pulse.waveform.style in ["const", "flat_top"]:
        return _run_len_rabi_fpga(cfg, lengths, context=context, retry=retry)
    return _run_len_rabi_host(cfg, np.unique(lengths), context=context, retry=retry)


def _run_len_rabi_fpga(
    cfg: LenRabiCfg, lengths: NDArray[np.float64], *, context: RunContext, retry: int
) -> LenRabiResult:
    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Length (us)", ylabel="Signal")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid()
    viewer = live_line(ax)

    @context.plots.on_plot_thread
    def update(data):
        return viewer.set(lengths, rabi_signal2real(data))

    fig.show()
    signals = SignalBuffer(
        (len(lengths),),
        on_update=update,
    )
    with Schedule(signals, scope=context.scope) as sched:
        builder = TimelineBuilder(
            context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
        )
        length = builder.add_sweep("length", cfg.length)
        if cfg.reset is not None:
            builder.emit(cfg.reset.build("reset"))
        builder.emit(Pulse("init_pulse", cfg.init_pulse))
        builder.emit(Pulse("qub_pulse", cfg.qub_pulse, length=length))
        builder.emit(Readout("readout", cfg.readout))
        prog = builder.build()
        sched.run_acquire(
            prog,
            context.soc,
            rounds=cfg.rounds,
            options=RunOptions(
                retry=retry,
                stop_condition=snr_checker(
                    signals.at(), cfg.earlystop_snr, rabi_signal2real
                ),
            ),
        )
    return LenRabiResult(lengths=lengths, signals=signals.array)


def _run_len_rabi_host(
    cfg: LenRabiCfg, lengths: NDArray[np.float64], *, context: RunContext, retry: int
) -> LenRabiResult:
    programs: dict[float, TimelineProgramV2] = {}

    def average_round(signals: NDArray[np.complex128]) -> NDArray[np.complex128]:
        mask = np.any(~np.isnan(signals), axis=0)
        mean = np.full(signals.shape[1], np.nan, dtype=np.complex128)
        mean[mask] = np.nanmean(signals[:, mask], axis=0)
        return mean

    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Length (us)", ylabel="Signal")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid()
    viewer = live_line(ax)

    @context.plots.on_plot_thread
    def update(data):
        return viewer.set(lengths, rabi_signal2real(average_round(data)))

    fig.show()
    signals = SignalBuffer(
        (cfg.rounds, len(lengths)),
        on_update=update,
    )
    check_snr = snr_checker(
        signals.at(),
        cfg.earlystop_snr,
        lambda data: rabi_signal2real(average_round(data)),
    )
    with Schedule(signals, scope=context.scope) as sched:
        for _, rep in sched.repeat("round", cfg.rounds):
            for length, step in rep.scan("length", lengths.tolist()):
                length_key = float(length)
                if length_key not in programs:
                    builder = TimelineBuilder(
                        context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                    )
                    if cfg.reset is not None:
                        builder.emit(cfg.reset.build("reset"))
                    builder.emit(Pulse("init_pulse", cfg.init_pulse))
                    builder.emit(Pulse("qub_pulse", cfg.qub_pulse, length=length))
                    builder.emit(Readout("readout", cfg.readout))
                    programs[length_key] = builder.build()
                step.run_acquire(
                    programs[length_key],
                    context.soc,
                    rounds=1,
                    options=RunOptions(retry=retry),
                )
            if check_snr():
                break
    return LenRabiResult(lengths=lengths, signals=average_round(signals.array))


def run_amp_rabi(
    cfg: AmpRabiCfg,
    *,
    context: RunContext,
    retry: int,
) -> AmpRabiResult:
    """Measure Rabi oscillations versus qubit drive gain at fixed duration.

    Reset, apply the optional init pulse, play the gain-swept qubit
    pulse, then read out. Return gains and mean complex IQ; subsequent
    analysis uses the oscillation period to infer pi and pi/2 gains.
    retry retries transient acquisition failures; earlystop_snr stops
    averaging at completed-round boundaries. None disables SNR early stop.
    """
    # Prepare the hardware gain grid and display the mean IQ projection.
    cfg = deepcopy(cfg)
    soc, soccfg = context.soc, context.soccfg

    gains = sweep2array(
        cfg.gain,
        "gain",
        {"soccfg": soccfg, "gen_ch": cfg.qub_pulse.ch},
    )

    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Pulse gain", ylabel="Amplitude")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid()
    viewer = live_line(ax)

    @context.plots.on_plot_thread
    def update(data):
        return viewer.set(gains, rabi_signal2real(data))

    fig.show()
    signals_buffer = SignalBuffer(
        (len(gains),),
        on_update=update,
    )

    # Keep pulse duration fixed while the FPGA sweeps its rotation amplitude.
    with Schedule(signals_buffer, scope=context.scope) as sched:
        builder = TimelineBuilder(soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay)
        gain_sweep = cfg.gain
        gain_param = builder.add_sweep("gain", gain_sweep)

        if cfg.reset is not None:
            builder.emit(cfg.reset.build("reset"))
        builder.emit(Pulse("init_pulse", cfg.init_pulse))
        builder.emit(Pulse("qub_pulse", cfg.qub_pulse, gain=gain_param))
        builder.emit(Readout("readout", cfg.readout))
        prog = builder.build()
        sched.run_acquire(
            prog,
            soc,
            rounds=cfg.rounds,
            options=RunOptions(
                retry=retry,
                stop_condition=snr_checker(
                    signals_buffer.at(), cfg.earlystop_snr, rabi_signal2real
                ),
            ),
        )

    return AmpRabiResult(amps=gains, signals=signals_buffer.array)


def run_freq_gain(
    cfg: FreqGainCfg,
    *,
    context: RunContext,
    retry: int,
) -> FreqGainResult:
    """Measure readout SNR over a frequency-by-gain grid.

    At each point, apply the optional reset, omit or apply qub_pulse,
    then read out. Configure qub_pulse for the intended excitation.
    Return frequencies in MHz, gains in a.u. within [-1, 1] and moment-based SNR
    shaped (Nfreq, Ngain), optionally reduced by skew_penalty for distorted
    IQ distributions. This score is not assignment fidelity.
    retry sets the transient acquisition retry count.
    """
    # Prepare both readout axes on the pulse generator's hardware grid.
    soc, soccfg = context.soc, context.soccfg
    cfg = deepcopy(cfg)

    freqs = sweep2array(
        cfg.freq,
        "freq",
        {"soccfg": soccfg, "gen_ch": cfg.readout.gen_ch},
    )
    gains = sweep2array(
        cfg.gain,
        "gain",
        {"soccfg": soccfg, "gen_ch": cfg.readout.gen_ch},
    )

    # Retain one separation score per frequency/gain pair.
    fig, ax = context.plots.subplots(num="measurement")
    ax.set(xlabel="Frequency (MHz)", ylabel="Gain (a.u.)")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    heatmap = live_heatmap(ax)

    @context.plots.on_plot_thread
    def update(data: NDArray[np.float64]) -> tuple[Artist, ...]:
        return heatmap.set(freqs, gains, np.abs(data))

    fig.show()
    signals_buffer = SignalBuffer(
        (len(freqs), len(gains)),
        dtype=np.float64,
        on_update=update,
    )

    # Compare pulse-off/on branches while sweeping both readout knobs in hardware.
    with Schedule(signals_buffer, scope=context.scope) as sched:
        freq_sweep = cfg.freq
        freq_param = sweep2param("freq", freq_sweep)
        cfg.readout.set_param("freq", freq_param)

        gain_sweep = cfg.gain
        gain_param = sweep2param("gain", gain_sweep)
        cfg.readout.set_param("gain", gain_param)
        tracker = MomentTracker()

        builder = TimelineBuilder(soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay)
        builder.add_loop("ge", 2)
        builder.add_sweep("freq", freq_sweep)
        builder.add_sweep("gain", gain_sweep)
        if cfg.reset is not None:
            builder.emit(Reset("reset", cfg.reset))
        with (
            builder.branch(by="ge", align_timeline=False) as cases,
            cases.case(1),
        ):
            builder.emit(Pulse("qub_pulse", cfg.qub_pulse))
        builder.emit(Readout("readout", cfg.readout))
        program = builder.build()
        sched.run_acquire(
            program,
            soc,
            rounds=cfg.rounds,
            options=RunOptions(
                retry=retry,
                converter=lambda _raw: snr_as_signal(
                    [tracker],
                    ge_axis=1,
                    skew_penalty=cfg.skew_penalty,
                ),
            ),
            trackers=[tracker],
        )

    # Return the moment-derived score; raw shots and IQ moments are not retained.
    signals = signals_buffer.array

    return FreqGainResult(freqs, gains, signals)
