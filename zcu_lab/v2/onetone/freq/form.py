"""One-tone starting values; the Run sweep owns the shared generator/ADC frequency."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue
from zcu_tools.cfg_model import ConfigBase, FieldMeta
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
from zcu_tools.program.v2 import PulseReadoutCfg, ResetCfg, SweepCfg

from zcu_lab.v2.onetone._form_shared import initial_spectrum_tree
from zcu_lab.v2.onetone.freq.core import HomophasalSamplingCfg
from zcu_lab.v2.support.measure.seed_metadata import (
    pulse_readout_seed_metadata,
)


class LinearInput(FormModel):
    """Linear frequency spacing, with no calibration parameters."""

    kind: Literal["Linear"] = "Linear"


class HomophasalInput(HomophasalSamplingCfg, FormModel):
    """Equal phase spacing using the core's required, positive fit parameters."""

    kind: Literal["Homophasal"] = "Homophasal"
    r_f: Annotated[
        float,
        Display(
            label="Resonance frequency",
            unit="MHz",
            description="Enter the center from a previous resonator fit.",
        ),
    ]
    rf_w: Annotated[
        float,
        Display(
            label="Linewidth (FWHM)",
            unit="MHz",
            description="Enter the full linewidth from that fit.",
        ),
    ]
    theta0: Annotated[
        float,
        Display(
            label="Circle phase offset",
            unit="rad",
            description="Enter the resonator-circle phase offset from that fit.",
        ),
    ]

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
        Display(
            label="Sampling mode",
            description="Choose equal frequency steps (Linear) or equal resonator-circle phase steps (Homophasal).",
        ),
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


class FreqInput(ConfigBase, SamplingInput):
    """One-tone acquisition inputs with the selected sampling branch.

    Linear has no calibration fields. Homophasal requires resonance frequency
    and linewidth in MHz and phase offset in radians. Run records the resulting
    core cfg rather than this operator input.
    """

    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        PulseReadoutCfg,
        FieldMeta(
            label="Readout",
            description="Set the pulse and ADC window used together at each frequency.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout RF frequency",
            unit="MHz",
            description="Enter the absolute readout RF sweep endpoints in MHz, not mixer offsets.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per frequency in each averaging round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter how many times to repeat and average the entire sweep.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the relaxation wait in µs after each shot.",
        ),
    ]


def make_form(ctx: FormContext) -> Form[FreqInput]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqInput,
        # The union selector replaces core sampling_mode/homophasal after the sweep.
        # Keep sampling after the frequency sweep instead of its inherited first position.
        field_order=(
            "reset",
            "readout",
            "freq",
            "sampling",
            "reps",
            "rounds",
            "relax_delay",
        ),
        tree=initial_spectrum_tree(ctx),
    ).with_seed_metadata(pulse_readout_seed_metadata(gain=0.05))
