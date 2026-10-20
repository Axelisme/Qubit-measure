"""Shared time-window inputs and captured pulse selections for CPMG."""

from typing import Annotated

from pydantic import Field, model_validator
from zcu_tools.cfg_model import Hidden, LogPoints, Slot, Template
from zcu_tools.gui.form import Display, FormModel
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time

from .core import CPMG_Cfg


class CPMGFormCfg(CPMG_Cfg):
    """Keep valid hidden time fields until GUI extras are lowered.

    These two defaults only supply the cfg validation before lower runs.
    Every successful conversion overwrites both from validated GUI extras.
    Notebook callers retain the required fields of CPMG_Cfg.
    """

    length_range: list[tuple[float, float]] | tuple[float, float] = (1.0, 100.0)
    length_expts: int = 101


class TimeWindowInput(FormModel):
    """Shared total-time bounds in us and sample count for every pulse count."""

    min_length_us: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False),
        Display(label="Minimum time", unit="us"),
    ] = 1.0
    max_length_us: Annotated[
        float,
        Field(gt=0, allow_inf_nan=False),
        Display(label="Maximum time", unit="us"),
    ] = 100.0
    length_points: Annotated[int, Field(ge=2), Display(label="Time points")] = 101

    @model_validator(mode="after")
    def validate_window(self) -> "TimeWindowInput":
        """Reject reversed or empty windows before converting to a run cfg."""
        if self.max_length_us <= self.min_length_us:
            raise ValueError("Maximum time must exceed minimum time")
        return self


def lower_time_window(extras: TimeWindowInput, cfg: CPMGFormCfg) -> CPMGFormCfg:
    """Replace hidden time fields from the validated shared GUI time window."""
    cfg.length_range = (extras.min_length_us, extras.max_length_us)
    cfg.length_expts = extras.length_points
    return cfg


def make_form(ctx: FormContext) -> Form[CPMGFormCfg]:
    """Build CPMG counts, shared time window and fixed-capture module defaults."""
    return Form(
        CPMGFormCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                "pi2_pulse": Slot("$qubit.module.x90")
                if ctx.has("$qubit.module.x90")
                else Template("pi2_pulse", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "times": LogPoints(max=16, points=5, include_zero=False),
                "length": Hidden(),
            },
            "length_range": Hidden(),
            "length_expts": Hidden(),
            "detune_ratio": 0.1,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "initial_delay": Hidden(),
            "reps": 1000,
            "rounds": 100,
        },
        extras=TimeWindowInput,
        lower=lower_time_window,
    )
