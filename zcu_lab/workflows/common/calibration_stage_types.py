"""Scientific inputs and transient outcomes of the shared calibration stages."""

from dataclasses import dataclass

from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.workflows.common.calibration_analysis import FrequencyFit
from zcu_lab.workflows.common.calibration_plots import CalibrationTrace
from zcu_lab.workflows.common.calibration_results import (
    FrequencyResult,
    RabiResult,
)


@dataclass(frozen=True)
class CalibrationPulses:
    """Detached drive, independent spectroscopy probe and retained readout.

    qub is the plan's pi template; its length/gain are not a calibrated pi.
    readout already carries the latest accepted (or seed) readout frequency and
    gain. Stages copy these modules before changing them.
    """

    qub: PulseCfg
    probe: PulseCfg
    readout: PulseReadoutCfg


@dataclass(frozen=True)
class ReadoutPulses:
    """Calibrated pi preparation and the readout whose freq/gain are scanned around.

    pi holds this point's accepted frequency, length and gain. readout holds the
    retained readout centre. Stages copy these modules before changing them.
    """

    pi: PulseCfg
    readout: PulseReadoutCfg


@dataclass(frozen=True)
class CalibrationPoint:
    """Zero-based point index and commanded flux in the named device unit."""

    index: int
    flux: float
    flux_unit: str


@dataclass(frozen=True)
class PointGuess:
    """This point's identity and history-based T1 estimate for relax delays.

    relax_us is the positive, finite T1 estimate in microseconds.
    """

    point: CalibrationPoint
    relax_us: float


@dataclass(frozen=True)
class FrequencyGuess:
    """Point to predict and seed calibration flux; None disables matrix feedback."""

    point: CalibrationPoint
    seed_flux: float | None


@dataclass(frozen=True)
class RabiGuess:
    """Current frequency and history-derived Rabi timing/gain guesses.

    frequency_mhz is None without this point's accepted frequency. relax_us is
    the positive T1 estimate. prior_pi_us is the last accepted pi length (us)
    that centres the length sweep. target_pi_us is the resolved target, seed_product
    is seed pi length times template gain (us a.u.). m_ratio is the point/seed
    matrix-element ratio, or None. last_amp_gain centers amp-only tracking, or
    None for its initial full window. point labels the returned heatmap row.
    """

    point: CalibrationPoint
    frequency_mhz: float | None
    relax_us: float
    prior_pi_us: float
    target_pi_us: float
    seed_product: float
    m_ratio: float | None
    last_amp_gain: float | None


@dataclass(frozen=True)
class FrequencyStageResult:
    """Persisted spectroscopy decision plus transient fit and matrix estimates.

    fit is the last successfully fitted attempt, including rejected candidates;
    None means no new heatmap row. m_ratio is the point/seed matrix-element ratio
    captured before any physical recovery, or None. Only record is persisted.
    """

    record: FrequencyResult
    fit: FrequencyFit | None = None
    m_ratio: float | None = None


@dataclass(frozen=True)
class RabiStageResult:
    """Persisted Rabi decision and transient rows/first-stage acceptance.

    length_record preserves accepted Len calibration even when Amp rejects.
    len_trace/amp_trace are absent for failed/partial acquisitions. Only record
    goes into point JSON; calibrate_point applies history updates from these
    results, never lending mutable history to an experiment stage.
    """

    record: RabiResult
    length_record: RabiResult | None = None
    len_trace: CalibrationTrace | None = None
    amp_trace: CalibrationTrace | None = None
