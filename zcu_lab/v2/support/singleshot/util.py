from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import CircleClassifier
from zcu_tools.experiment.v2.runtime import ShotResult
from zcu_tools.program.v2 import TimelineProgramV2
from zcu_tools.program.v2.modules.readout import (
    PulseReadoutCfg,
    ReadoutCfg,
    circle_classifier,
)


def readout_confusion_matrix(cfg: PulseReadoutCfg | None) -> NDArray[np.float64] | None:
    """Read saved single-qubit circle correction, never point calibration.

    Missing historical cfg/classifier/matrix skips correction. Linear has no
    other state and cannot correct a three-state population analysis.
    Pulse readout supplies the single ADC's classifier. Radius is only needed
    during acquisition.
    """
    if cfg is None:
        return None
    classifier = cfg.classifier
    if classifier is None:
        return None
    if not isinstance(classifier, CircleClassifier):
        raise ValueError(
            "Population correction requires other; linear has only g/e states"
        )
    matrix = classifier.confusion_matrix
    return None if matrix is None else np.asarray(matrix, dtype=np.float64)


def rabi_preview_classifier(cfg: ReadoutCfg) -> CircleClassifier | None:
    """Return the sole ADC's circle for raw-IQ Rabi preview, or None if absent.

    Raw Rabi supports Pulse or a one-tone Mux only. Missing calibration skips
    preview without blocking acquisition or joint fitting. A supplied linear
    classifier cannot show Other; a circle without radius requires GE Post.
    Values use raw integrated ADC divided by readout length, without offset.
    """
    if isinstance(cfg, PulseReadoutCfg):
        ro_ch, classifier = cfg.ro_ch, cfg.classifier
    else:
        if len(cfg.tones) != 1:
            raise ValueError("Raw Rabi requires exactly one readout ADC")
        ro_ch, classifier = cfg.tones[0].ro_ch, cfg.tones[0].classifier
    if classifier is None:
        return None
    circle = circle_classifier(cfg, ro_ch=ro_ch)
    if circle.radius is None:
        raise ValueError("Rabi preview requires radius; run singleshot GE Post")
    return circle


def ge_signals_by_state(
    signals: NDArray[np.complex128],
    initial_state: Literal["ground", "excited"],
) -> NDArray[np.complex128]:
    """Map acquisition rows (probe off/on) to predominantly ground/excited."""
    if initial_state == "ground":
        return signals
    if initial_state == "excited":
        return signals[::-1]
    raise ValueError(f"Unknown initial state: {initial_state!r}")


def raw_shots_to_signal(
    result: ShotResult, program: TimelineProgramV2
) -> NDArray[np.complex128]:
    """Convert every shot of the first visible read to (points, reps) complex IQ.

    Require one visible ADC in result. Divide by its integration length from
    the originating program, without offset subtraction or post-selection.
    """
    if len(result.shots) != 1:
        raise ValueError(
            "singleshot raw conversion requires exactly one readout channel"
        )
    (channel, raw) = next(iter(result.shots.items()))
    iq = raw[..., 0, :] / float(program.ro_chs[channel]["length"])
    return np.asarray(iq[..., 0] + 1j * iq[..., 1], dtype=np.complex128)
