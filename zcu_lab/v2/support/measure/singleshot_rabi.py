from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from zcu_tools.analysis.fitting.singleshot_rabi import RabiJointFitSummary
from zcu_tools.cfg_model import CircleClassifier
from zcu_tools.gui.app.measure.adapter import WritebackItem
from zcu_tools.gui.app.measure.writeback import calibrated_module_writeback
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg, ReadoutCfg


def rabi_calibration_writeback(
    fit: RabiJointFitSummary,
    readout: ReadoutCfg | None,
    parameters: ParameterContext | None,
) -> Sequence[WritebackItem]:
    """Propose the complete fitted circle on the captured resonator readout slot.

    Fit centers/radius are raw ADC divided by readout length, without offset.
    Invalid/nonfinite fits yield no items. A valid fit requires saved cfg and
    capture; absent sources, multiple Mux tones or non-classifier cfg differences
    raise ValueError. Preserve library child keys in one complete replacement.
    """
    calibration_is_finite = (
        fit.backend.valid
        and np.isfinite([fit.g_center.real, fit.g_center.imag]).all()
        and np.isfinite([fit.e_center.real, fit.e_center.imag]).all()
        and np.isfinite(fit.radius)
        and np.isfinite(fit.confusion_matrix).all()
    )
    if not calibration_is_finite:
        return []
    if readout is None or parameters is None:
        raise ValueError("Rabi writeback requires captured parameters and readout cfg")
    if isinstance(readout, PulseReadoutCfg):
        path = "classifier"
    else:
        if len(readout.tones) != 1:
            raise ValueError("Rabi writeback requires exactly one readout ADC")
        path = "tones.0.classifier"
    return [
        calibrated_module_writeback(
            parameters,
            readout,
            target="$resonator.module.readout",
            description="Rabi fitted centers, radius and prepared-to-measured confusion",
            field_updates={
                path: CircleClassifier(
                    g_center=fit.g_center,
                    e_center=fit.e_center,
                    radius=fit.radius,
                    confusion_matrix=fit.confusion_matrix.tolist(),
                )
            },
        )
    ]
