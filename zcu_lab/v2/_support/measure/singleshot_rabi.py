from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from zcu_tools.gui.app.measure.adapter import MetaDictWriteback, WritebackItem

from zcu_lab.v2._support.singleshot.rabi_fit import RabiJointFitResult


def rabi_calibration_writeback(fit: RabiJointFitResult) -> Sequence[WritebackItem]:
    calibration_is_finite = (
        fit.backend.valid
        and np.isfinite([fit.g_center.real, fit.g_center.imag]).all()
        and np.isfinite([fit.e_center.real, fit.e_center.imag]).all()
        and np.isfinite(fit.radius)
        and np.isfinite(fit.confusion_matrix).all()
    )
    if not calibration_is_finite:
        return []

    return [
        MetaDictWriteback(
            target_name="g_center",
            description="Rabi fitted |g> IQ cluster centre (complex)",
            proposed_value=fit.g_center,
        ),
        MetaDictWriteback(
            target_name="e_center",
            description="Rabi fitted |e> IQ cluster centre (complex)",
            proposed_value=fit.e_center,
        ),
        MetaDictWriteback(
            target_name="ge_radius",
            description="Rabi fitted single-shot classification radius",
            proposed_value=fit.radius,
        ),
        MetaDictWriteback(
            target_name="confusion_matrix",
            description="Rabi fitted 3x3 confusion matrix",
            proposed_value=fit.confusion_matrix.tolist(),
        ),
    ]
