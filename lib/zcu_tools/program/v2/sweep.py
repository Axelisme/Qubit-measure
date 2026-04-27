from __future__ import annotations

import math

from pydantic import model_validator

from zcu_tools.cfg_model import ConfigBase


class SweepCfg(ConfigBase):
    start: float
    stop: float
    expts: int
    step: float

    @model_validator(mode="after")
    def _validate_sweep_consistency(self) -> "SweepCfg":
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
