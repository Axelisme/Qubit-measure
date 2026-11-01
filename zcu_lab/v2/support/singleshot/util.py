from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import CircleClassifier
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


class RawShotProgram(Protocol):
    @property
    def ro_chs(self) -> object: ...

    def get_raw(self) -> Sequence[NDArray[Any]] | None: ...


def raw_shots_to_signal(program: RawShotProgram) -> NDArray[np.complex128]:
    acc_buf = program.get_raw()
    if acc_buf is None:
        raise RuntimeError("program did not expose raw shot buffer")
    ro_chs = program.ro_chs
    if not isinstance(ro_chs, Mapping):
        raise TypeError("program readout channels must be a mapping")
    if len(ro_chs) != 1:
        raise ValueError(
            "singleshot raw conversion requires exactly one readout channel"
        )

    (ro_info,) = ro_chs.values()
    if not isinstance(ro_info, Mapping):
        raise TypeError("readout channel info must be a mapping")
    length = ro_info.get("length")
    if not isinstance(length, int | float):
        raise TypeError("readout channel length must be numeric")

    avgiq = acc_buf[0] / float(length)
    i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]
    return np.asarray(i0 + 1j * q0, dtype=np.complex128)
