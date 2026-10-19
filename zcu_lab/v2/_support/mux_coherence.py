"""Common admission, source data and staged acquisition for mux Ramsey/Echo."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import Join, Module, MuxReadout

from zcu_lab.v2.twotone.time_domain.detune import actual_detune

from .mux_liveplot import draw_mux_magnitudes
from .mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    validate_mux_scan,
)


class MuxCoherenceToneCfg(ConfigBase):
    """One named original tone's required x90 drive and MHz detune.

    name is its nonempty capture-generated qubit identity. drive is the fixed
    pi/2 source pulse. detune ramps the final pi/2 phase, not its carrier.
    Full-list admission validates every pulse, detune and independent channel,
    including nonparticipants, and rejects duplicate names.
    """

    name: Annotated[str, FieldMeta(readonly=True)]
    drive: PulseCfg
    detune: float = 0.0


def validate_coherence_scan(
    cfg: ProgramV2Cfg,
    modules: MuxDriveModuleCfg,
    tones: Sequence[MuxCoherenceToneCfg],
    delay: SweepCfg,
    participants: Sequence[str],
) -> None:
    """Reject invalid full tones, fixed pulses, channels or common delay (us).

    Reuse mux numeric admission with the same shared delay for every tone; no
    independent per-tone sweep is stored. Require finite nonnegative delay bounds
    and finite detunes. Raises ValueError without hardware access or mutation.
    """
    if not all(math.isfinite(v) for v in (delay.start, delay.stop, delay.step)):
        raise ValueError("delay start/stop/step must be finite")
    if min(delay.start, delay.stop) < 0:
        raise ValueError("delay must be nonnegative us")
    for index, tone in enumerate(tones):
        if not math.isfinite(tone.detune):
            raise ValueError(f"tones[{index}].detune must be finite MHz")
    validate_mux_scan(
        cfg,
        modules,
        [MuxToneCfg(name=t.name, drive=t.drive, sweep=delay) for t in tones],
        participants,
    )


@dataclass(frozen=True)
class MuxCoherenceResult:
    """Explicit Ramsey/Echo source in original active tone order.

    tone_indices: int64 (K,), original full readout indices, unique/nonnegative.
    times: float64 (P,), common requested total free-evolution delay in us, not
    compiled actual delay. Echo splits this into two native half-delay waits.
    signals: complex128 (K,P), ADC-length-normalized average I+jQ, NaN if unwritten.
    actual_detunes: float64 (K,), MHz phase-ramp rates from actual_detune owner.
    May contain NaN when nonzero detune has no positive rounded delay; preserve
    the owner's diagnostics, never substitute zero/nominal detune.
    source_frequencies: float64 (K,), MHz from each source x90 pulse before ramp.
    Full pulse cfg and inactive tones remain in RunRecord.cfg. Cancellation and
    acquisition error status belong to the RunContext StopSignal, not these data.
    """

    tone_indices: NDArray[np.int64]
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    actual_detunes: NDArray[np.float64]
    source_frequencies: NDArray[np.float64]


def validate_coherence_result(result: MuxCoherenceResult) -> None:
    """Validate source shapes/dtypes/finite axes; permit NaN partial IQ only.

    Require K,P>0, unique nonnegative indices and nonnegative finite requested
    times. Source frequencies must be finite. Actual detunes may be NaN when the
    owner cannot estimate a rate, but not infinite. Raises ValueError, never fits or
    changes data; callers must not turn source-contract errors into fit failures.
    """
    indices, times = result.tone_indices, result.times
    if (
        indices.dtype != np.dtype(np.int64)
        or indices.ndim != 1
        or not indices.size
        or np.any(indices < 0)
        or np.unique(indices).size != indices.size
    ):
        raise ValueError("tone_indices must be unique nonnegative int64 (K,)")
    if (
        times.dtype != np.dtype(np.float64)
        or times.ndim != 1
        or not times.size
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
    ):
        raise ValueError("times must be finite nonnegative float64 (P,) us")
    if result.signals.dtype != np.dtype(np.complex128) or result.signals.shape != (
        indices.size,
        times.size,
    ):
        raise ValueError("signals must be complex128 (K,P)")
    if np.any(np.isinf(result.signals)):
        raise ValueError("signals may contain NaN partial data, not infinity")
    detunes, frequencies = result.actual_detunes, result.source_frequencies
    if (
        detunes.dtype != np.dtype(np.float64)
        or detunes.shape != indices.shape
        or np.any(np.isinf(detunes))
    ):
        raise ValueError("actual_detunes must be float64 (K,), finite or NaN")
    if (
        frequencies.dtype != np.dtype(np.float64)
        or frequencies.shape != indices.shape
        or not np.all(np.isfinite(frequencies))
    ):
        raise ValueError("source_frequencies must be finite float64 (K,)")


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            # collect_iq exposes (P,K,2); the Result/buffer is tone-major.
            return (iq[..., 0] + 1j * iq[..., 1]).T
    raise ValueError("Coherence program is missing its named MuxReadout")


def acquire_coherence(
    cfg: ProgramV2Cfg,
    modules: MuxDriveModuleCfg,
    indexed_tones: Sequence[tuple[int, MuxCoherenceToneCfg]],
    delay: SweepCfg,
    *,
    context: RunContext,
    pi_pulses: Sequence[PulseCfg | None] | None = None,
) -> MuxCoherenceResult:
    """Acquire admitted full tones using stage-wide Join and one hardware sweep.

    cfg supplies shared reps/rounds/initial/relax delays (us); modules includes
    full mux readout and optional reset. indexed_tones contains admitted selected
    (original index, tone) pairs in participant order; delay passes leaf admission.
    pi_pulses=None selects Ramsey; otherwise it is a full-list Echo pi-pulse
    sequence, same independent channels as x90 drives. Inputs remain unchanged.
    Every x90 stage starts together. Echo additionally joins the pi stage between
    two common half-delay waits; delays begin after the longest prior stage.
    Final x90 phase follows each tone's nominal detune and the shared total-delay
    parameter. Public program/sweep owners quantize time, phase and pulses.

    Return requested delay, actual MHz detunes/source frequencies and average IQ
    (K,P), retaining supplied original index order. Publish measurement_i and
    combined measurement with original index/source name legend.
    Completed-round averages survive cancellation, no-data slots stay NaN. Errors
    remain on context.cancel_signal; caller must raise_if_error before success.
    Build/conversion failures propagate. context owns hardware, fresh StopSignal
    and Plots; caller finishes/releases plots and cleans up devices. No I/O/save.
    """
    tones = dict(indexed_tones)
    indices = list(tones)
    drives: list[PulseCfg] = []
    for index in indices:
        drive = tones[index].drive
        drives.append(drive)
    scaler = 1.0 if pi_pulses is None else 0.5
    times = sweep2array(delay, "none")
    # Exercise the existing shared-delay grid owner before hardware admission.
    _ = sweep2array(delay, "time", {"soccfg": context.soccfg, "scaler": scaler})
    detunes = np.asarray(
        [
            actual_detune(
                delay,
                detune=tones[index].detune,
                gen_ch=drive.ch,
                soccfg=context.soccfg,
                time_scaler=scaler,
            )
            for index, drive in zip(indices, drives, strict=True)
        ],
        dtype=np.float64,
    )
    frequencies = np.asarray([float(drive.freq) for drive in drives], dtype=np.float64)
    readout = modules.readout.with_updates(
        tones=[modules.readout.tones[index] for index in indices]
    )
    viewers = [
        context.plots.liveplot_1d(
            f"measurement_{index}", "Requested delay (us)", "Magnitude (a.u.)"
        )
        for index in indices
    ]

    def update(data: NDArray[np.complex128]) -> None:
        for row, viewer in enumerate(viewers):
            viewer.update(times, np.abs(data[row]))
        draw_mux_magnitudes(
            context.plots,
            np.broadcast_to(times, data.shape),
            data,
            [f"{index}: {tones[index].name}" for index in indices],
            "Requested delay (us)",
        )

    signals = SignalBuffer((len(indices), times.size), on_update=update)
    with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
        length = sweep2param("length", delay)
        stages: list[Module] = [
            Join(
                *[
                    Pulse(f"pi2_first_{i}", drive)
                    for i, drive in zip(indices, drives, strict=True)
                ]
            ),
            Delay("delay_first", scaler * length),
        ]
        if pi_pulses is not None:
            echoes: list[Pulse] = []
            for index in indices:
                pi = pi_pulses[index]
                if pi is None:
                    raise ValueError(f"tones[{index}].pi_pulse is required")
                echoes.append(Pulse(f"pi_{index}", pi))
            stages.extend([Join(*echoes), Delay("delay_second", scaler * length)])
        stages.extend(
            [
                Join(
                    *[
                        Pulse(
                            f"pi2_final_{index}",
                            drive.with_updates(
                                phase=drive.phase + 360 * tones[index].detune * length
                            ),
                        )
                        for index, drive in zip(indices, drives, strict=True)
                    ]
                ),
                MuxReadout("readout", readout),
            ]
        )
        builder = (
            sched.prog_builder(context.soc, context.soccfg)
            .add_reset("reset", modules.reset)
            .add(*stages)
            .declare_sweep("length", delay)
        )
        prog = builder.build()
        _ = builder.run_program(prog, raw2signal_fn=partial(_averaged_signal, prog))
    signals.trigger_update(flush=True)
    return MuxCoherenceResult(
        np.asarray(indices, dtype=np.int64),
        times,
        signals.array.copy(),
        detunes,
        frequencies,
    )
