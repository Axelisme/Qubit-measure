"""Explicit host-side herald declarations in normalized ADC coordinates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import CircleClassifier, ReadoutClassifier
from zcu_tools.utils.shot_classification import classify_shots


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
    Calibration and signals are not mutated.
    """
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
        """Reject incomplete or invalid circle calibration before acquisition."""
        herald_ground_mask(np.empty(0, dtype=np.complex128), self.classifier)
