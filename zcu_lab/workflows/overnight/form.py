"""Independent fixed-flux overnight plan/knobs forms; no calibration inputs."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue
from zcu_tools.cfg_model import ModuleCfgFactory, PulseReadoutSpec, PulseSpec
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormErrors,
    FormModel,
    FormPath,
    InputConverter,
    Option,
)
from zcu_tools.program.v2.modules import ModuleCfgFactory as NativeModuleCfgFactory
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.workflows.common.calibration_modules import (
    DRIVE_MODULE_FIELD,
    READOUT_MODULE_FIELD,
    DriveModuleInput,
    ReadoutModuleInput,
)
from zcu_lab.workflows.overnight.workflow import OvernightPlan, OvernightTunables


class OvernightPlanForm(FormModel):
    """Select fixed-flux T1/Echo pairs with calibrated pulses and fixed grids.

    flux_dev defaults to the captured $qubit.flux_source device name
    when input is built, or stays unfilled without one; flux_unit is required
    A/V/native with no conversion or membership precheck. flux is its absolute
    fixed setpoint. outer_rounds is finite pair count; start_interval_s is actual
    start spacing in seconds; points is delay samples per native acquisition.
    qub_pulse/readout are shared ref/inline modules, already calibrated; no pi
    estimate, frequency, Rabi or readout calibration is added. Native plan
    receives the historical host snapshot separately, not a live source.
    """

    flux_dev: Annotated[str, Display(label="Flux device")] = Field(
        min_length=1, description="Connected flux driver name"
    )
    flux_unit: Annotated[Literal["A", "V", "native"], Display(label="Flux unit")] = (
        Field(description="Explicit flux unit, without conversion")
    )
    flux: float = Field(description="Fixed absolute setpoint in flux_unit")
    outer_rounds: Annotated[int, Display(label="Outer rounds")] = Field(
        ge=1, description="Finite number of T1/Echo pairs"
    )
    start_interval_s: Annotated[float, Display(label="Start interval", unit="s")] = (
        Field(gt=0, description="Actual pair start spacing in seconds")
    )
    points: int = Field(
        default=51, ge=5, description="Uniform delay samples per member"
    )
    qub_pulse: Annotated[DriveModuleInput, Display(label="Drive pulse")] = Field(
        description="Fixed calibrated pi drive"
    )
    readout: Annotated[ReadoutModuleInput, Display(label="Readout")] = Field(
        description="Fixed pulse-backed readout"
    )

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Seed a direct device name from $qubit.flux_source; leave monitoring inputs empty."""
        data = super().defaults(env)
        # FormModel evaluates expression defaults once into direct input.
        data["flux_dev"] = {"expr": "$qubit.flux_source"}
        return data

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer captured pulse/readout module roots, without materialization or I/O."""
        if env.library is None:
            return {}
        return {
            ("qub_pulse", "name"): tuple(
                key
                for key in env.library.module_keys()
                if isinstance(env.library.module_spec(key), PulseSpec)
            ),
            ("readout", "name"): tuple(
                key
                for key in env.library.module_keys()
                if isinstance(env.library.module_spec(key), PulseReadoutSpec)
            ),
        }

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        """Bind each generated field's converter to the host's fixed capture."""
        return {
            ("qub_pulse",): DRIVE_MODULE_FIELD.converter(env),
            ("readout",): READOUT_MODULE_FIELD.converter(env),
        }

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        """Declare generated selections without narrowing inline module families."""
        return {
            ("qub_pulse",): DRIVE_MODULE_FIELD.options(env),
            ("readout",): READOUT_MODULE_FIELD.options(env),
        }

    def to_cfg(self, env: FormEnv, snapshot: RunSnapshot) -> OvernightPlan | FormErrors:
        """Resolve fixed modules through the captured native materialization owner.

        env supplies library/assets context for ref/inline modules. snapshot is
        historical entry/point/roles. Missing references, wrong families and
        materialization ValueErrors return consuming-field FormErrors; asset I/O
        and native plan validation errors propagate. No capture, live read,
        calibration or acquisition occurs. Fixed arb
        drive is supported because this workflow never changes pulse duration.
        """
        context = env.library.validation_context() if env.library is not None else None
        errors: dict[FormPath, str] = {}
        drive = None
        readout = None
        try:
            core = ModuleCfgFactory.from_raw(
                DRIVE_MODULE_FIELD.to_spec(self.qub_pulse),
                context=context,
            )
            native = NativeModuleCfgFactory.from_core(core, context=context)
            if not isinstance(native, PulseCfg):
                raise ValueError("Overnight drive selection must be a calibrated pulse")
            drive = native
        except ValueError as exc:
            errors[("qub_pulse",)] = str(exc)
        try:
            core = ModuleCfgFactory.from_raw(
                READOUT_MODULE_FIELD.to_spec(self.readout),
                context=context,
            )
            native_ro = NativeModuleCfgFactory.from_core(core, context=context)
            if not isinstance(native_ro, PulseReadoutCfg):
                raise ValueError("Overnight readout selection must be pulse-backed")
            readout = native_ro
        except ValueError as exc:
            errors[("readout",)] = str(exc)
        if drive is None or readout is None:
            return FormErrors(errors)
        return OvernightPlan(
            flux_dev=self.flux_dev,
            flux_unit=self.flux_unit,
            flux=self.flux,
            outer_rounds=self.outer_rounds,
            start_interval_s=self.start_interval_s,
            points=self.points,
            qub_pulse=drive,
            readout=readout,
            snapshot=snapshot,
        )


class OvernightTunablesForm(FormModel):
    """Set averaging and acceptance bounds to resolve fixed-flux lifetime drift.

    reps/rounds average each native member, not monitoring pairs. Guesses are in
    microseconds; sweep_factor fixes each grid. Both relax delays have a
    relax_min_us floor. T1 uses relax_factor*t1_guess_us. Echo uses this round's
    accepted T1, or t1_guess_us if none accepted.
    min_lifetime_points defaults to 5 acquired fitted delays in [0, lifetime].
    Invalid quality rejects too. Fits change only Echo relax, not grids or pulses.
    Expressions stay in FormInput; native conversion carries ordinary values.
    """

    uniform: Annotated[bool, Display(label="Uniform (linear) sweep")] = Field(
        default=True,
        description="T1 only; False concentrates delays within the same start/stop window",
    )
    reps: Annotated[int, Display(label="Averages")] = Field(
        default=1000, ge=1, description="Native acquisition averages"
    )
    rounds: int = Field(default=10, ge=1, description="Native acquisition rounds")
    t1_guess_us: Annotated[float, Display(label="T1 guess", unit="µs")] = Field(
        default=20.0, gt=0, description="Fixed T1 timing guess, microseconds"
    )
    t2_guess_us: Annotated[float, Display(label="Echo guess", unit="µs")] = Field(
        default=10.0, gt=0, description="Fixed Echo timing guess, microseconds"
    )
    t1_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    t1_earlystop_snr: float | None = Field(
        default=None,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    t2_acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    t2_earlystop_snr: float | None = Field(
        default=None,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    sweep_factor: Annotated[float, Display(label="Sweep factor")] = Field(
        default=4.0, gt=0, description="Delay endpoint / timing guess"
    )
    relax_factor: Annotated[float, Display(label="Relax factor")] = Field(
        default=5.0, gt=0, description="Relaxation delay / T1 estimate"
    )
    relax_min_us: Annotated[float, Display(label="Relax minimum", unit="µs")] = Field(
        default=1.0, ge=0, description="Relaxation lower bound in us"
    )
    max_model_window_factor: Annotated[float, Display(label="Model window factor")] = (
        Field(
            default=2.0,
            gt=0,
            description="Diagnostic lifetime / maximum fitted delay upper bound; no feedback change",
        )
    )
    min_lifetime_points: Annotated[int, Display(label="Min lifetime points")] = Field(
        default=5,
        ge=1,
        description="Minimum fitted delay points with 0 <= delay <= lifetime",
    )
    max_rel_err: Annotated[float, Display(label="Max relative error")] = Field(
        default=0.5, gt=0, description="Strict error/lifetime upper bound"
    )
    max_fit_residual: Annotated[float, Display(label="Max fit residual")] = Field(
        default=0.1, gt=0, description="Strict normalized residual RMS upper bound"
    )

    def to_cfg(self) -> OvernightTunables:
        """Convert validated ordinary knobs; no source, hardware or expression parsing."""
        return OvernightTunables(
            max_model_window_factor=self.max_model_window_factor,
            uniform=self.uniform,
            min_lifetime_points=self.min_lifetime_points,
            reps=self.reps,
            rounds=self.rounds,
            t1_acquire_retry=self.t1_acquire_retry,
            t1_earlystop_snr=self.t1_earlystop_snr,
            t2_acquire_retry=self.t2_acquire_retry,
            t2_earlystop_snr=self.t2_earlystop_snr,
            t1_guess_us=self.t1_guess_us,
            t2_guess_us=self.t2_guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
            relax_min_us=self.relax_min_us,
            max_rel_err=self.max_rel_err,
            max_fit_residual=self.max_fit_residual,
        )
