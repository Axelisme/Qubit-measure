"""Concrete workflow callbacks using the current native cfg and Result types.

The Engine owns device setup, cancellation interpretation and publication.
Callbacks use Run.buffer/Run.schedule directly, never the old Exp.run host.
Acquisition evidence is captured at acquisition time, not at save time.
"""

from dataclasses import dataclass

from zcu_tools.experiment.acquisition import AcquisitionEvidence
from zcu_tools.experiment.workflows import Run

from zcu_lab.v2.twotone.freq.core import FreqCfg, FreqResult
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiCfg, LenRabiResult
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainCfg, FreqGainResult
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Result


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


def run_qubit_freq(run: Run[FreqCfg]) -> FrequencyAcquisition:
    """Acquire a qubit-frequency sweep using a complete native FreqCfg.

    Run must supply soc/soccfg. Reset and optional init pulse precede the
    frequency-swept qubit pulse and readout. Return native MHz coordinates,
    complex IQ and acquisition evidence, including partial data on stop.
    Missing capabilities and metadata errors propagate. Schedule errors remain
    in Run.outcome for Engine to interpret; this callback does not rethrow them.
    No device setup, analysis, plotting or saving occurs here.
    """
    raise NotImplementedError("Frequency acquisition body is not implemented")


def run_len_rabi(run: Run[LenRabiCfg]) -> RabiAcquisition:
    """Acquire a length-Rabi sweep using a complete native LenRabiCfg.

    Run must supply soc/soccfg. The qubit waveform must be const or flat_top;
    other styles raise ValueError before acquisition. Reset and optional init
    pulse precede the length-swept qubit pulse and readout. Return native us
    coordinates, complex IQ and evidence, including partial data on stop.
    Capability/metadata errors propagate; Schedule errors stay in Run.outcome.
    No device setup, analysis, plotting or saving occurs here.
    """
    raise NotImplementedError("Length-Rabi acquisition body is not implemented")


def run_ro_landscape(run: Run[FreqGainCfg]) -> ReadoutAcquisition:
    """Acquire a g/e readout frequency/gain sweep using native FreqGainCfg.

    Run must supply soc/soccfg. Use the existing MomentTracker/SNR scorer and
    cfg.skew_penalty. Return native MHz/gain coordinates, real SNR magnitude
    and evidence, including partial data on stop. Reset precedes the g/e pulse
    branch and swept readout. Capability/metadata errors propagate; Schedule
    errors stay in Run.outcome. No device setup, analysis, plotting or saving.
    """
    raise NotImplementedError("Readout acquisition body is not implemented")


def run_t1(run: Run[T1Cfg]) -> T1Acquisition:
    """Acquire uniform T1 using a complete native T1Cfg, without an Exp host.

    Run must supply soc/soccfg. cfg.uniform must be True and sweep.length must
    be SweepCfg; otherwise raise ValueError before acquisition. Reset, pi pulse,
    swept delay and readout produce native us coordinates and complex IQ.
    Return evidence captured around acquisition, with partial data on stop.
    Capability/metadata errors propagate; Schedule errors stay in Run.outcome.
    No device setup, analysis, plotting or saving occurs here.
    """
    raise NotImplementedError("T1 acquisition body is not implemented")
