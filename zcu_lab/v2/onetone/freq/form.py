"""One-tone starting values; the Run sweep owns the shared generator/ADC frequency."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue
from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form import Display, FormEnv, FormModel
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.onetone._shared import initial_ro_length
from zcu_lab.v2.onetone.freq.core import FreqCfg, HomophasalSamplingCfg


class LinearInput(FormModel):
    """Linear frequency spacing, with no calibration parameters."""

    kind: Literal["Linear"] = "Linear"


class HomophasalInput(HomophasalSamplingCfg, FormModel):
    """Equal phase spacing using the core's required, positive fit parameters."""

    kind: Literal["Homophasal"] = "Homophasal"
    r_f: Annotated[float, Display(label="Resonance frequency", unit="MHz")]
    rf_w: Annotated[float, Display(label="Linewidth", unit="MHz")]
    theta0: Annotated[float, Display(label="Phase offset", unit="rad")]

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Seed fresh branches from the fixed capture; unresolved values stay missing."""
        del env
        return {
            "kind": "Homophasal",
            "r_f": {"expr": "$resonator.freq"},
            "rf_w": {"expr": "$resonator.kappa"},
            "theta0": {"expr": "$resonator.ext.theta0"},
        }


class SamplingInput(FormModel):
    """One sampling selector; only Homophasal exposes calibration parameters."""

    sampling: Annotated[
        LinearInput | HomophasalInput,
        Field(discriminator="kind"),
        Display(label="Sampling mode"),
    ] = Field(default_factory=LinearInput)


def lower_sampling(extras: SamplingInput, cfg: FreqCfg) -> FreqCfg:
    """Translate the selected branch to the core's unchanged paired sampling fields.

    extras contains validated calibration values for Homophasal, or a parameter-free
    Linear branch. cfg is the validated generated core configuration before lowering.
    Return a detached cfg; the Form validates the complete pair again before Run.
    """
    sampling = extras.sampling
    if isinstance(sampling, LinearInput):
        return cfg.model_copy(update={"sampling_mode": "linear", "homophasal": None})
    return cfg.model_copy(
        update={
            "sampling_mode": "homophasal",
            "homophasal": HomophasalSamplingCfg(
                r_f=sampling.r_f, rf_w=sampling.rf_w, theta0=sampling.theta0
            ),
        }
    )


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    del ctx
    return Form(
        FreqCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    set={
                        # Run supplies the linear sweep or homophasal frequency grid.
                        "freq": Hidden(0.0),
                        "gain": 0.05,
                        "ro_length": initial_ro_length(),
                    },
                ),
            },
            "sweep": {
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=301
                )
            },
            "sampling_mode": Hidden(),
            "homophasal": Hidden(),
            "relax_delay": 1.0,
            "reps": 100,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
        extras=SamplingInput,
        lower=lower_sampling,
    )
