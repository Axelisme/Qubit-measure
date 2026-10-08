"""Concrete workflow callbacks using the current native cfg and Result types.

The Engine owns device setup, cancellation interpretation and publication.
Callbacks use Run.buffer/Run.schedule directly, never the old Exp.run host.
Acquisition evidence is captured at acquisition time, not at save time.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from zcu_tools.experiment.acquisition import (
    AcquisitionEvidence,
    capture_software_provenance,
    new_run_id,
)
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_as_signal
from zcu_tools.experiment.v2.utils.tracker.moment import MomentTracker
from zcu_tools.experiment.workflows import Run
from zcu_tools.program.v2 import (
    Branch,
    Delay,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    sweep2param,
)

from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiCfg, LenRabiResult
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainCfg, FreqGainResult
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Result
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg, T2EchoResult


@dataclass(frozen=True)
class FrequencyAcquisition(FreqResult):
    """Native frequency trace with its acquisition history.

    freqs is the one-dimensional MHz coordinate. signals is complex IQ of the
    same shape. evidence contains UTC acquisition boundaries and provenance;
    complete/stopped describes normal return with the Run signal unset/set.
    Engine separately interprets Schedule failures and user cancellation.
    """

    evidence: AcquisitionEvidence


@dataclass(frozen=True)
class RabiAcquisition(LenRabiResult):
    """Native length-Rabi trace with its acquisition history.

    lengths is the one-dimensional microsecond coordinate. signals is complex
    IQ of the same shape. evidence contains UTC boundaries and provenance,
    with complete/stopped determined by the Run signal at normal return.
    """

    evidence: AcquisitionEvidence


@dataclass(frozen=True)
class ReadoutAcquisition(FreqGainResult):
    """Native readout landscape with its acquisition history.

    freqs is the MHz coordinate and gains is the dimensionless gain coordinate.
    signals is real SNR magnitude, shaped (len(freqs), len(gains)). evidence
    contains UTC boundaries and provenance, with complete/stopped determined
    by the Run signal at normal return.
    """

    evidence: AcquisitionEvidence


@dataclass(frozen=True)
class T1Acquisition(T1Result):
    """Native T1 trace with its acquisition history.

    times is the one-dimensional microsecond coordinate. signals is complex IQ
    of the same shape. evidence contains UTC boundaries and provenance, with
    complete/stopped determined by the Run signal at normal return.
    """

    evidence: AcquisitionEvidence


@dataclass(frozen=True, kw_only=True)
class T2EchoAcquisition(T2EchoResult):
    """Native zero-detune echo trace with acquisition-time evidence.

    times is the one-dimensional total echo delay in microseconds, rounded
    through the native half-delay sweep. signals is complex IQ of that shape.
    true_activate_detune is 0.0 for this decay-only callback. evidence holds
    UTC acquisition boundaries and software provenance; complete/stopped is
    determined by the Run signal on return, not by an Engine commit.
    """

    evidence: AcquisitionEvidence


def run_qubit_freq(run: Run[FreqCfg]) -> FrequencyAcquisition:
    """Acquire a qubit-frequency sweep using a complete native FreqCfg.

    Run must supply soc/soccfg. Reset and optional init pulse precede the
    frequency-swept qubit pulse and readout. Return native MHz coordinates,
    complex IQ and acquisition evidence, including partial data on stop.
    Missing capabilities and metadata errors propagate. Schedule errors remain
    in Run.outcome for Engine to interpret; this callback does not rethrow them.
    No device setup, analysis, plotting or saving occurs here.
    """
    soc, soccfg = run.soc, run.soccfg
    cfg = run.cfg
    freqs = sweep2array(
        cfg.sweep.freq,
        "freq",
        {"soccfg": soccfg, "gen_ch": cfg.modules.qub_pulse.ch},
    )
    buffer = run.buffer((len(freqs),), axes=(freqs,))
    provenance = capture_software_provenance(context=None)
    started_at = datetime.now(UTC)
    run_id = new_run_id(started_at)

    with run.schedule(buffer) as sched:
        modules = sched.cfg.modules
        freq_sweep = sched.cfg.sweep.freq
        modules.qub_pulse.set_param("freq", sweep2param("freq", freq_sweep))
        (
            sched.prog_builder(soc, soccfg)
            .add_reset("reset", modules.reset)
            .add_pulse("init_pulse", modules.init_pulse)
            .add_pulse("qub_pulse", modules.qub_pulse)
            .add_readout("readout", modules.readout)
            .declare_sweep("freq", freq_sweep)
            .build_and_acquire()
        )

    evidence = AcquisitionEvidence(
        run_id=run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        completion="stopped" if run.cancel_signal.is_set() else "complete",
        provenance=provenance,
    )
    return FrequencyAcquisition(freqs=freqs, signals=buffer.array, evidence=evidence)


def run_len_rabi(run: Run[LenRabiCfg]) -> RabiAcquisition:
    """Acquire a length-Rabi sweep using a complete native LenRabiCfg.

    Run must supply soc/soccfg. The qubit waveform must be const or flat_top;
    other styles raise ValueError before acquisition. Reset and optional init
    pulse precede the length-swept qubit pulse and readout. Return native us
    coordinates, complex IQ and evidence, including partial data on stop.
    Capability/metadata errors propagate; Schedule errors stay in Run.outcome.
    No device setup, analysis, plotting or saving occurs here.
    """
    cfg = run.cfg
    if cfg.modules.qub_pulse.waveform.style not in ("const", "flat_top"):
        raise ValueError("Length-Rabi requires const or flat_top pulse style")
    soc, soccfg = run.soc, run.soccfg
    lengths = sweep2array(
        cfg.sweep.length,
        "time",
        {"soccfg": soccfg, "gen_ch": cfg.modules.qub_pulse.ch},
    )
    buffer = run.buffer((len(lengths),), axes=(lengths,))
    provenance = capture_software_provenance(context=None)
    started_at = datetime.now(UTC)
    run_id = new_run_id(started_at)

    with run.schedule(buffer) as sched:
        modules = sched.cfg.modules
        length_sweep = sched.cfg.sweep.length
        modules.qub_pulse.set_param("length", sweep2param("length", length_sweep))
        (
            sched.prog_builder(soc, soccfg)
            .add_reset("reset", modules.reset)
            .add_pulse("init_pulse", modules.init_pulse)
            .add_pulse("qubit_pulse", modules.qub_pulse)
            .add_readout("readout", modules.readout)
            .declare_sweep("length", length_sweep)
            .build_and_acquire()
        )

    evidence = AcquisitionEvidence(
        run_id=run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        completion="stopped" if run.cancel_signal.is_set() else "complete",
        provenance=provenance,
    )
    return RabiAcquisition(lengths=lengths, signals=buffer.array, evidence=evidence)


def run_ro_landscape(run: Run[FreqGainCfg]) -> ReadoutAcquisition:
    """Acquire a g/e readout frequency/gain sweep using native FreqGainCfg.

    Run must supply soc/soccfg. Use the existing MomentTracker/SNR scorer and
    cfg.skew_penalty. Return native MHz/gain coordinates, real SNR magnitude
    and evidence, including partial data on stop. Reset precedes the g/e pulse
    branch and swept readout. Capability/metadata errors propagate; Schedule
    errors stay in Run.outcome. No device setup, analysis, plotting or saving.
    """
    soc, soccfg = run.soc, run.soccfg
    cfg = run.cfg
    round_info = {"soccfg": soccfg, "gen_ch": cfg.modules.readout.pulse_cfg.ch}
    freqs = sweep2array(cfg.sweep.freq, "freq", round_info)
    gains = sweep2array(cfg.sweep.gain, "gain", round_info)
    buffer = run.buffer((len(freqs), len(gains)), axes=(freqs, gains))
    provenance = capture_software_provenance(context=None)
    started_at = datetime.now(UTC)
    run_id = new_run_id(started_at)

    with run.schedule(buffer) as sched:
        modules = sched.cfg.modules
        freq_sweep = sched.cfg.sweep.freq
        gain_sweep = sched.cfg.sweep.gain
        modules.readout.set_param("freq", sweep2param("freq", freq_sweep))
        modules.readout.set_param("gain", sweep2param("gain", gain_sweep))
        tracker = MomentTracker()
        (
            sched.prog_builder(soc, soccfg)
            .add(
                Reset("reset", modules.reset),
                Branch("ge", [], Pulse("qub_pulse", modules.qub_pulse)),
                Readout("readout", modules.readout),
            )
            .declare_sweep("ge", 2)
            .declare_sweep("freq", freq_sweep)
            .declare_sweep("gain", gain_sweep)
            .build_and_acquire(
                raw2signal_fn=lambda _raw: snr_as_signal(
                    [tracker], ge_axis=1, skew_penalty=sched.cfg.skew_penalty
                ),
                trackers=[tracker],
            )
        )

    evidence = AcquisitionEvidence(
        run_id=run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        completion="stopped" if run.cancel_signal.is_set() else "complete",
        provenance=provenance,
    )
    return ReadoutAcquisition(
        freqs=freqs, gains=gains, signals=np.abs(buffer.array), evidence=evidence
    )


def run_t1(run: Run[T1Cfg]) -> T1Acquisition:
    """Acquire uniform T1 using a complete native T1Cfg, without an Exp host.

    Run must supply soc/soccfg. cfg.uniform must be True and sweep.length must
    be SweepCfg; otherwise raise ValueError before acquisition. Reset, pi pulse,
    swept delay and readout produce native us coordinates and complex IQ.
    Return evidence captured around acquisition, with partial data on stop.
    Capability/metadata errors propagate; Schedule errors stay in Run.outcome.
    No device setup, analysis, plotting or saving occurs here.
    """
    cfg = run.cfg
    if not cfg.uniform or not isinstance(cfg.sweep.length, SweepCfg):
        raise ValueError("T1 requires uniform mode with a SweepCfg length sweep")
    soc, soccfg = run.soc, run.soccfg
    times = sweep2array(cfg.sweep.length, "time", {"soccfg": soccfg})
    buffer = run.buffer((len(times),), axes=(times,))
    provenance = capture_software_provenance(context=None)
    started_at = datetime.now(UTC)
    run_id = new_run_id(started_at)

    with run.schedule(buffer) as sched:
        modules = sched.cfg.modules
        # Use the validated Run sweep without mutating it; modules are schedule-local.
        length_sweep = cfg.sweep.length
        length_param = sweep2param("length", length_sweep)
        (
            sched.prog_builder(soc, soccfg)
            .add(
                Reset("reset", modules.reset),
                Pulse("pi_pulse", modules.pi_pulse),
                Delay("t1_delay", length_param),
                Readout("readout", modules.readout),
            )
            .declare_sweep("length", length_sweep)
            .build_and_acquire()
        )

    evidence = AcquisitionEvidence(
        run_id=run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        completion="stopped" if run.cancel_signal.is_set() else "complete",
        provenance=provenance,
    )
    return T1Acquisition(times=times, signals=buffer.array, evidence=evidence)


def run_t2echo(run: Run[T2EchoCfg]) -> T2EchoAcquisition:
    """Acquire a native uniform, zero-detune Hahn-echo decay using Run.

    Supply a complete T2EchoCfg with its SweepCfg total length and native
    pi/2, pi, reset and readout modules, and Run soc/soccfg capabilities.
    Nonzero detune raises ValueError before acquisition; fringe acquisition
    is outside this callback's contract. Native half-delay rounding preserves
    the canonical T2EchoResult time coordinate. Return complex IQ, zero
    activated detune and acquisition evidence, with partial data on stop.
    Capability/metadata errors propagate. Schedule failures remain in
    Run.outcome for Engine interpretation. No setup, fitting, plotting,
    saving, context capture or connection occurs, and cfg is not modified.
    """
    cfg = run.cfg
    if cfg.detune != 0.0:
        raise ValueError("T2echo decay acquisition requires zero detune")
    soc, soccfg = run.soc, run.soccfg
    times = sweep2array(cfg.sweep.length, "time", {"soccfg": soccfg, "scaler": 0.5})
    buffer = run.buffer((len(times),), axes=(times,))
    provenance = capture_software_provenance(context=None)
    started_at = datetime.now(UTC)
    run_id = new_run_id(started_at)

    with run.schedule(buffer) as sched:
        modules = sched.cfg.modules
        length_sweep = cfg.sweep.length
        length_param = sweep2param("length", length_sweep)
        (
            sched.prog_builder(soc, soccfg)
            .add(
                Reset("reset", modules.reset),
                Pulse("pi2_pulse1", modules.pi2_pulse),
                Delay("t2e_delay1", delay=0.5 * length_param),
                Pulse("pi_pulse", modules.pi_pulse),
                Delay("t2e_delay2", delay=0.5 * length_param),
                Pulse("pi2_pulse2", modules.pi2_pulse),
                Readout("readout", modules.readout),
            )
            .declare_sweep("length", length_sweep)
            .build_and_acquire()
        )

    evidence = AcquisitionEvidence(
        run_id=run_id,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        completion="stopped" if run.cancel_signal.is_set() else "complete",
        provenance=provenance,
    )
    return T2EchoAcquisition(
        times=times,
        signals=buffer.array,
        true_activate_detune=0.0,
        evidence=evidence,
    )
