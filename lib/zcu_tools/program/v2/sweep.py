from __future__ import annotations

import math
from typing import Annotated

from pydantic import Field, model_validator

from zcu_tools.cfg_model import FULL_SCALE_GAIN_BOUNDS, ConfigBase


class SweepCfg(ConfigBase):
    start: Annotated[float, Field(allow_inf_nan=False)]
    stop: Annotated[float, Field(allow_inf_nan=False)]
    expts: int
    step: Annotated[float, Field(allow_inf_nan=False)]

    @model_validator(mode="after")
    def _validate_sweep_consistency(self) -> SweepCfg:
        if self.expts <= 0:
            raise ValueError(f"expts must be greater than 0, got {self.expts}")

        if self.expts == 1:
            if not math.isclose(self.start, self.stop, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "for expts == 1, start and stop must be the same value "
                    f"(got start={self.start}, stop={self.stop})"
                )
            if not math.isclose(self.step, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"for expts == 1, step must be 0 (got step={self.step})"
                )
            return self

        if math.isclose(self.step, 0.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("step must not be zero when expts > 1")

        expected_stop = self.start + self.step * (self.expts - 1)
        if not math.isclose(self.stop, expected_stop, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "invalid sweep setting: stop must satisfy "
                "start + step * (expts - 1). "
                f"got start={self.start}, step={self.step}, expts={self.expts}, "
                f"stop={self.stop}, expected_stop={expected_stop}"
            )

        return self


GainSweepCfg = Annotated[SweepCfg, FULL_SCALE_GAIN_BOUNDS]
"""A native SweepCfg whose first and last gain lie within DAC full scale.

Use as a Pydantic field annotation, or validate with TypeAdapter(GainSweepCfg).
"""
