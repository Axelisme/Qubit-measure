"""One-tone starting values; the Run sweep owns the shared generator/ADC frequency."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue
from zcu_tools.gui.form import (
    Display,
    FormCommandContext,
    FormCommandSnapshot,
    FormConversionError,
    FormConversionResult,
    FormEnv,
    FormModel,
    FormPath,
    InputConverter,
)
from zcu_tools.gui.form.commands import ReadonlyFormInput
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.onetone._form_shared import initial_spectrum_tree
from zcu_lab.v2.onetone._shared import LinearFreqCfg
from zcu_lab.v2.onetone.freq.core import HomophasalSamplingCfg
from zcu_lab.v2.support.measure.pulse_readout_metadata import (
    pulse_readout_seed_metadata,
)


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

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        """Install fresh branch input from the fixed capture when sampling changes.

        Linear discards calibration fields. Homophasal uses captured defaults;
        missing calibration stays required rather than falling back to Linear.
        """

        def convert(
            snapshot: FormCommandSnapshot,
            selection: ReadonlyFormInput,
            context: FormCommandContext,
        ) -> FormConversionResult:
            del snapshot, context
            if selection == {"kind": "Linear"}:
                return FormConversionResult(
                    LinearInput, LinearInput.default_input(env).data
                )
            if selection == {"kind": "Homophasal"}:
                return FormConversionResult(
                    HomophasalInput, HomophasalInput.default_input(env).data
                )
            raise FormConversionError("Unknown sampling mode")

        return {("sampling",): InputConverter(convert)}


class FreqInput(LinearFreqCfg, SamplingInput):
    """One-tone acquisition inputs with the selected sampling branch.

    Linear has no calibration fields. Homophasal requires resonance frequency
    and linewidth in MHz and phase offset in radians. Run records the resulting
    core cfg rather than this operator input.
    """


def make_form(ctx: FormContext) -> Form[FreqInput]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqInput,
        field_order=(
            "reset",
            "readout",
            "freq",
            "reps",
            "rounds",
            "relax_delay",
        ),
        tree=initial_spectrum_tree(ctx),
    ).with_seed_metadata(pulse_readout_seed_metadata(gain=0.05))
