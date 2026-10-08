"""User-facing native T1 plan input; hosts own historical capture and Start."""

from typing import Self

import numpy as np
from pydantic import Field, model_validator
from zcu_tools.datafile.native_models import RunSnapshot
from zcu_tools.gui.form import ExplicitList, FormModel, RangeInput, field_error

from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg
from zcu_lab.workflows.calibration import CalibrationPlan
from zcu_lab.workflows.t1 import T1Plan


class T1PlanForm(FormModel):
    """Editable setpoints and complete native calibration/T1 templates.

    flux_dev names the host device; fluxes are absolute values in its units.
    points controls the uniform delay grid. calibration and t1 retain their
    native models' validation and complete routing/timing inputs. Required
    templates stay absent until supplied, never replaced by dummy defaults.
    Templates are ordinary native cfg values, not library reference envelopes.
    """

    flux_dev: str = Field(min_length=1, description="Host flux device name")
    fluxes: RangeInput = Field(description="Ordered absolute device setpoints")
    points: int = Field(default=51, ge=5, description="Uniform T1 delay samples")
    calibration: CalibrationPlan = Field(description="Complete calibration templates")
    t1: T1Cfg = Field(description="Complete uniform T1 template")

    @model_validator(mode="after")
    def validate_uniform(self) -> Self:
        """Reject nonuniform acquisition at its consuming template field."""
        if not self.t1.uniform:
            raise field_error(
                ("t1", "uniform"), "T1 flux dependence requires uniform T1"
            )
        return self

    def to_cfg(self, snapshot: RunSnapshot) -> T1Plan:
        """Expand range/list to a native plan with host-captured history.

        snapshot is the fixed historical entry/point/roles, not a live context.
        Keep descending/repeated/list order and a single range point. Native
        validation failures propagate for invalid direct edits. No I/O, capture,
        submission or additional deep copy occurs; Engine isolates its plan.
        """
        if isinstance(self.fluxes, ExplicitList):
            fluxes = self.fluxes.values
        else:
            fluxes = tuple(
                float(value)
                for value in np.linspace(
                    self.fluxes.start,
                    self.fluxes.stop,
                    self.fluxes.n,
                )
            )
        return T1Plan(
            flux_dev=self.flux_dev,
            fluxes=fluxes,
            points=self.points,
            calibration=self.calibration,
            t1=self.t1,
            snapshot=snapshot,
        )
