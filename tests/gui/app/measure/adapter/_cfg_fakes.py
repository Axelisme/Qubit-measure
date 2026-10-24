"""Framework-only definitions and typed inputs for adapter seam tests."""

from typing import ClassVar, Literal

from pydantic import Field
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.base import Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.program.v2 import ModuleCfg


class CalibrationCfg(ConfigBase):
    """Typed fake input: frequency and width are positive; phase is finite."""

    frequency: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    width: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    phase: float = Field(default=0.0, allow_inf_nan=False)


class CalibrationInput(ConfigBase):
    """Fake cfg: mode is an editor choice, calibration is nested typed input.

    modules holds resolved program modules, keyed by their test-local names.
    Neither mode changes how the framework validates or lowers calibration.
    """

    mode: Literal["plain", "calibrated"] = "plain"
    calibration: CalibrationCfg = Field(default_factory=CalibrationCfg)
    modules: dict[str, ModuleCfg] = Field(default_factory=dict)


class _RejectRun(Experiment[float, CalibrationInput]):
    def run(self, cfg: CalibrationInput, *, context: RunContext) -> float:
        raise AssertionError("Invalid cfg reached core execution")


class CalibrationAdapter(BaseAdapter[CalibrationInput, float]):
    """Use real BaseAdapter cfg assembly with a core that rejects execution."""

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=False
    )
    exp_cls = _RejectRun
