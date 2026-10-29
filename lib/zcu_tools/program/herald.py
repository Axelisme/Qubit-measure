"""Explicit host-side herald declarations in normalized ADC coordinates."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import Centroid3Classifier, CircleClassifier, ReadoutClassifier
from zcu_tools.program._shot_processing import validated_payload
from zcu_tools.utils.shot_classification import classify_shots

if TYPE_CHECKING:
    from zcu_tools.program.acquisition import ShotProcessor


def herald_ground_mask(
    signals: NDArray[np.complex128], classifier: ReadoutClassifier
) -> NDArray[np.bool_]:
    """Return a boolean mask with the same shape as complex IQ signals.

    signals contains integrated ADC I+1j*Q divided by the integration length in
    ADC samples, without subtracting IQ offsets. No normalization or population
    correction is performed here. Circle accepts only points strictly within
    radius of g_center and strictly closer to g_center than e_center; ties and
    radius boundaries fail, exactly as classify_result's g region. Missing radius
    raises ValueError requesting GE Post calibration; invalid circle geometry
    also raises ValueError. Linear accepts real(signals*exp(1j*angle)) <= threshold
    (the g side), with angle in radians and threshold in these same ADC units.
    Centroid3 raises ValueError: three-state heralding is unsupported.
    Calibration and signals are not mutated.
    """
    if isinstance(classifier, Centroid3Classifier):
        raise ValueError("centroid3 classifier is unsupported for herald selection")
    if isinstance(classifier, CircleClassifier):
        if classifier.radius is None:
            raise ValueError("Herald circle requires radius; run GE Post calibration")
        ground, _, _ = classify_shots(
            signals, classifier.g_center, classifier.e_center, classifier.radius
        )
        return ground
    return np.real(signals * np.exp(1j * classifier.angle)) <= classifier.threshold


@dataclass(frozen=True)
class HeraldSpec:
    """One ADC read to classify before averaging the remaining reads.

    ro_ch is the logical ADC channel number, not its position in returned data.
    read_index is the zero-based trigger emission index on that ADC within each
    shot, independent of sweep/repetition loops. reads_per_shot is the total
    number of reads on that ADC, including herald and later signal reads; the
    acquisition consumer must compare it with the actual acquisition layout.
    classifier holds calibration in integrated ADC sums divided by integration
    length (ADC samples), without subtracting IQ offsets. Linear angle is in
    radians. Construction validates calibration but does not acquire data,
    classify shots or apply population correction. Pass this value explicitly
    to the acquisition caller.
    """

    ro_ch: int
    read_index: int
    classifier: ReadoutClassifier
    reads_per_shot: int

    def __post_init__(self) -> None:
        """Reject centroid3 or incomplete/invalid circle calibration before acquisition."""
        herald_ground_mask(np.empty(0, dtype=np.complex128), self.classifier)


def heralded(spec: HeraldSpec, inner: "ShotProcessor") -> "ShotProcessor":
    """Wrap a signal processor with one read's ground acceptance.

    The returned callable accepts normalized float64 (shots, reads, 2) IQ.
    It requires reads == spec.reads_per_shot, at least two reads and a valid
    read_index, otherwise raises ValueError. It removes that read before calling
    inner, which must return float64 (shots, out_reads, N) with positive output
    axes. Inner errors propagate. The result is (shots, out_reads, N+1): accepted
    payloads followed by acceptance 1; rejected payloads are zero even for NaN
    inner values, followed by acceptance 0. Neither input nor calibration is
    mutated. Use herald_ratio after acquisition averaging to recover the
    conditional signal and per-point pass fraction.
    """

    def process(iq: NDArray[np.float64]) -> NDArray[np.float64]:
        if iq.ndim != 3 or iq.shape[-1] != 2 or iq.dtype != np.float64:
            raise ValueError("heralded input must be float64 (shots, reads, 2) IQ")
        reads = iq.shape[1]
        if reads != spec.reads_per_shot or reads < 2:
            raise ValueError(
                "heralded reads must match reads_per_shot and be at least two"
            )
        if type(spec.read_index) is not int or not 0 <= spec.read_index < reads:
            raise ValueError("heralded read_index must identify an input read")
        herald_iq = iq[:, spec.read_index]
        accepted = herald_ground_mask(
            herald_iq[:, 0] + 1j * herald_iq[:, 1], spec.classifier
        )[:, None, None]
        signal = np.delete(iq, spec.read_index, axis=1)
        payload = validated_payload(inner(signal), shots=iq.shape[0])
        weighted = np.where(accepted, payload, 0.0)
        acceptance = np.broadcast_to(accepted, (*payload.shape[:2], 1)).astype(
            np.float64
        )
        return np.concatenate([weighted, acceptance], axis=-1)

    return process


def herald_ratio(
    result: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Recover conditional payloads and pass fraction from averaged heralded data.

    result is float64 (out_reads, *point_dims, N+1), with acceptance as its last
    component repeated for every read. Return conditional values of shape
    (out_reads, *point_dims, N) and fraction of shape point_dims from the first
    read. Zero acceptance yields NaN values without divide warnings. Input is
    not mutated. Non-float64/non-ndarray input, rank below two, empty read axis
    or fewer than two final components raises ValueError.
    """
    result = _validated_ratio_input(result)
    acceptance = result[..., -1:]
    values = np.divide(
        result[..., :-1],
        acceptance,
        out=np.full_like(result[..., :-1], np.nan),
        where=acceptance != 0,
    )
    return values, np.array(result[0, ..., -1], dtype=np.float64, copy=True)


def _validated_ratio_input(result: object) -> NDArray[np.float64]:
    if (
        not isinstance(result, np.ndarray)
        or result.dtype != np.float64
        or result.ndim < 2
        or result.shape[0] == 0
        or result.shape[-1] < 2
    ):
        raise ValueError("herald_ratio requires float64 (out_reads, *point_dims, N+1)")
    return result
