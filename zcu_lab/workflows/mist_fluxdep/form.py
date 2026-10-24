"""Independent fixed gain-plan MIST forms and native module lowering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, JsonValue, model_validator
from zcu_tools.cfg_model import ModuleCfgFactory, ModuleSpec
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormErrors,
    FormModel,
    FormPath,
    InputConverter,
    Option,
    field_error,
)
from zcu_tools.gui.form_adapter import ModuleInput, ModuleInputField, SweepInput
from zcu_tools.program.v2.modules import ModuleCfgFactory as NativeModuleCfgFactory
from zcu_tools.program.v2.modules import PulseCfg
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)

from zcu_lab.workflows.common.calibration_form import CalibrationTunablesForm
from zcu_lab.workflows.common.calibration_modules import (
    DRIVE_MODULE_FIELD,
    READOUT_MODULE_FIELD,
    DriveModuleInput,
    ReadoutModuleInput,
    calibration_choices,
    calibration_defaults,
    expand_fluxes,
    resolve_calibration_modules,
)
from zcu_lab.workflows.mist_fluxdep.workflow import MISTPlan, MISTTunables

MIST_MODULE_FIELD = ModuleInputField(ModuleSpec)
RESET_MODULE_FIELD = ModuleInputField(ModuleSpec, nullable=True)

# Runtime schemas come from field.annotation; static aliases view nominal values.
if TYPE_CHECKING:
    type MistModuleInput = ModuleInput
    type ResetModuleInput = ModuleInput | None
else:
    MistModuleInput = MIST_MODULE_FIELD.annotation
    ResetModuleInput = RESET_MODULE_FIELD.annotation


class MISTPlanForm(FormModel):
    """Select flux and disturbance gain scans to measure MIST without a threshold fit.

    flux_dev/unit/fluxes are explicit absolute driver coordinates. qub_pulse
    supplies the Length-Rabi/calibrated pi waveform/gain; pi_length_us is its
    required initial range estimate, not a measured scalar. readout is shared
    pulse-backed calibration readout. reset is a ref/inline native reset or
    None to omit reset; mist_pulse is the required disturbance pulse selection.
    gain_start/stop/points are finite dimensionless sweep parameters, locked
    in plan throughout a run. No gain knobs, generic probe/init None or fitted
    threshold replaces these old native modules.
    """

    flux_dev: Annotated[str, Display(label="Flux device")] = Field(
        min_length=1, description="Connected flux driver name"
    )
    flux_unit: Annotated[Literal["A", "V", "native"], Display(label="Flux unit")] = (
        Field(description="Explicit setpoint unit")
    )
    fluxes: Annotated[SweepInput, Display(label="Fluxes")] = Field(
        description="Ordered absolute flux setpoints"
    )
    pi_length_us: Annotated[float, Display(label="Pi length", unit="µs")] = Field(
        gt=0, description="Initial pi-duration estimate (µs)"
    )
    qub_pulse: Annotated[DriveModuleInput, Display(label="Drive pulse")] = Field(
        description="Shared spectroscopy/Rabi/pi drive"
    )
    readout: Annotated[ReadoutModuleInput, Display(label="Readout")] = Field(
        description="Shared pulse-backed readout"
    )
    reset: Annotated[ResetModuleInput, Display(label="Reset")] = Field(
        default=None, description="Native reset selection or explicit omission"
    )
    mist_pulse: Annotated[MistModuleInput, Display(label="MIST pulse")] = Field(
        description="Required native MIST disturbance pulse"
    )
    gain_start: Annotated[float, Display(label="Gain start", unit="a.u.")] = Field(
        description="Fixed sweep start gain (a.u.)"
    )
    gain_stop: Annotated[float, Display(label="Gain stop", unit="a.u.")] = Field(
        description="Fixed sweep stop gain (a.u.)"
    )
    gain_points: Annotated[int, Display(label="Gain points")] = Field(
        ge=2, description="Run-locked uniform gain sample count"
    )

    @model_validator(mode="after")
    def _gain_range(self) -> MISTPlanForm:
        if self.gain_stop <= self.gain_start:
            raise field_error(("gain_stop",), "MIST gain stop must exceed start")
        return self

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Retain fixed flux alias/x180 pi estimate; leave gain/MIST fields required."""
        return calibration_defaults(super().defaults(env), env)

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer shared captured calibration roots without materialization/live I/O."""
        return calibration_choices(env)

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        """Bind each generated field's converter to the host's fixed capture."""
        return {
            ("qub_pulse",): DRIVE_MODULE_FIELD.converter(env),
            ("readout",): READOUT_MODULE_FIELD.converter(env),
            ("reset",): RESET_MODULE_FIELD.converter(env),
            ("mist_pulse",): MIST_MODULE_FIELD.converter(env),
        }

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        """Declare all four generated module selections, including nullable reset."""
        return {
            ("qub_pulse",): DRIVE_MODULE_FIELD.options(env),
            ("readout",): READOUT_MODULE_FIELD.options(env),
            ("reset",): RESET_MODULE_FIELD.options(env),
            ("mist_pulse",): MIST_MODULE_FIELD.options(env),
        }

    def to_cfg(self, env: FormEnv, snapshot: RunSnapshot) -> MISTPlan | FormErrors:
        """Resolve four fixed native modules, gain grid and historical snapshot.

        env supplies captured library/assets context for ref/inline selections.
        Invalid families/materialization ValueErrors return consuming-field
        FormErrors; asset I/O and native model errors propagate. reset may be
        omitted, but pi preparation and MIST disturbance never are. No live
        source/device read, acquisition, fit or threshold creation occurs.
        """
        shared = resolve_calibration_modules(
            qub_pulse=self.qub_pulse, readout=self.readout, env=env
        )
        if isinstance(shared, FormErrors):
            return shared
        context = env.library.validation_context() if env.library is not None else None
        errors: dict[FormPath, str] = {}
        mist = None
        reset = None
        try:
            core = ModuleCfgFactory.from_raw(
                MIST_MODULE_FIELD.to_spec(self.mist_pulse),
                context=context,
            )
            native = NativeModuleCfgFactory.from_core(core, context=context)
            if not isinstance(native, PulseCfg):
                raise ValueError("MIST disturbance selection must be a pulse")
            mist = native
        except ValueError as error:
            errors[("mist_pulse",)] = str(error)
        if self.reset is not None:
            try:
                core = ModuleCfgFactory.from_raw(
                    RESET_MODULE_FIELD.to_spec(self.reset),
                    context=context,
                )
                native_reset = NativeModuleCfgFactory.from_core(core, context=context)
                if not isinstance(
                    native_reset,
                    (PulseResetCfg, TwoPulseResetCfg, BathResetCfg),
                ):
                    raise ValueError("MIST reset selection must be a native reset")
                reset = native_reset
            except ValueError as error:
                errors[("reset",)] = str(error)
        if errors or mist is None:
            return FormErrors(errors)
        return MISTPlan(
            flux_dev=self.flux_dev,
            flux_unit=self.flux_unit,
            fluxes=expand_fluxes(self.fluxes),
            qub_pulse=shared.qub_pulse,
            readout=shared.readout,
            reset=reset,
            mist_pulse=mist,
            gain_start=self.gain_start,
            gain_stop=self.gain_stop,
            gain_points=self.gain_points,
            snapshot=snapshot,
        )


class MISTTunablesForm(FormModel):
    """Set calibration and averaging to resolve the raw MIST gain response.

    cal controls calibration policy; reps/rounds average the native MIST trace.
    guess_us is a positive microsecond T1 timing estimate (no extra acquisition);
    relax_factor scales it. The fixed gain grid is plan input, not a knob.
    """

    cal: Annotated[CalibrationTunablesForm, Display(label="Calibration")] = Field(
        default_factory=CalibrationTunablesForm
    )
    reps: Annotated[int, Display(label="Averages")] = Field(
        default=100, ge=1, description="MIST averages"
    )
    rounds: int = Field(default=10, ge=1, description="MIST averaging rounds")
    guess_us: Annotated[float, Display(label="T1 guess", unit="µs")] = Field(
        default=20.0, gt=0, description="Calibration T1 timing estimate (µs)"
    )
    relax_factor: Annotated[float, Display(label="Relax factor")] = Field(
        default=3.0, gt=0, description="MIST relax delay divided by guess"
    )

    def to_cfg(self) -> MISTTunables:
        """Copy validated ordinary knobs; no fit, owner update or source lookup."""
        return MISTTunables(
            cal=self.cal.to_cfg(),
            reps=self.reps,
            rounds=self.rounds,
            guess_us=self.guess_us,
            relax_factor=self.relax_factor,
        )
