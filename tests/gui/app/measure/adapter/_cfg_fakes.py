"""Framework-only definitions and typed inputs for adapter seam tests."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from pydantic import Field
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.program.v2 import ModuleCfg


def _empty_schema() -> CfgSchema:
    return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())


@dataclass(frozen=True)
class StaticCfgDefinition:
    """Supply independent copies of schema, the prepared spec/value test input."""

    schema: CfgSchema = field(default_factory=_empty_schema)

    @property
    def spec(self) -> CfgSectionSpec:
        """Return a detached declaration for framework shape comparisons."""
        return deepcopy(self.schema.spec)

    def instantiate(self, ctx: SessionEnv) -> CfgSchema:
        """Copy the prepared schema; this fake has no context-derived defaults."""
        return deepcopy(self.schema)


class CalibrationCfg(ConfigBase):
    """Typed fake input: frequency and width are positive; phase is finite."""

    frequency: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    width: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    phase: float = Field(default=0.0, allow_inf_nan=False)


class CalibrationInput(ExpCfgModel):
    """Fake cfg: mode is an editor choice, calibration is nested typed input.

    modules holds resolved program modules, keyed by their test-local names.
    Neither mode changes how the framework validates or lowers calibration.
    """

    mode: Literal["plain", "calibrated"] = "plain"
    calibration: CalibrationCfg = Field(default_factory=CalibrationCfg)
    modules: dict[str, ModuleCfg] = Field(default_factory=dict)


class _RejectRun:
    def run(self, cfg: CalibrationInput, *, context: RunContext) -> float:
        raise AssertionError("Invalid cfg reached core execution")


class CalibrationAdapter(BaseAdapter[CalibrationInput, float]):
    """Use real BaseAdapter cfg assembly with a core that rejects execution."""

    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=False
    )
    ExpCfg_cls = CalibrationInput
    exp_cls = _RejectRun

    @classmethod
    def cfg_definition(cls) -> StaticCfgDefinition:
        """Declare a mode selector and optional nested calibration inputs."""
        spec = CfgSectionSpec(
            fields={
                "mode": ScalarSpec("Mode", str, choices=["plain", "calibrated"]),
                "calibration": CfgSectionSpec(
                    fields={
                        name: ScalarSpec(name, float, optional=True)
                        for name in ("frequency", "width", "phase")
                    }
                ),
            }
        )
        value = CfgSectionValue(
            fields={
                "mode": DirectValue("plain"),
                "calibration": CfgSectionValue(
                    fields={
                        name: DirectValue(None)
                        for name in ("frequency", "width", "phase")
                    }
                ),
            }
        )
        return StaticCfgDefinition(CfgSchema(spec=spec, value=value))

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        """Return the fixed local name used by this non-running adapter fake."""
        return "calibration"
