"""Captured mux plan forms using the workflow host's handwritten FormModel API."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated

from pydantic import Field, JsonValue
from zcu_tools.cfg_model import ModuleCfgFactory, MuxReadoutSpec, PulseSpec, ResetSpec
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
from zcu_tools.gui.form_adapter import ModuleInput, ModuleInputField
from zcu_tools.gui.session.parameters import ModuleSlotError, ParameterContext
from zcu_tools.program.v2.modules import ModuleCfgFactory as NativeModuleCfgFactory
from zcu_tools.resources.entry import RoleResolutionError

from zcu_lab.workflows.overnight_mux.workflow import (
    OvernightMuxPlan,
    OvernightMuxTunables,
)

DRIVE = ModuleInputField(PulseSpec, nullable=True)
READOUT = ModuleInputField(MuxReadoutSpec)
RESET = ModuleInputField(ResetSpec, nullable=True)

if TYPE_CHECKING:
    type DriveInput = ModuleInput | None
    type ReadoutInput = ModuleInput
    type ResetInput = ModuleInput | None
else:
    DriveInput = DRIVE.annotation
    ReadoutInput = READOUT.annotation
    ResetInput = RESET.annotation


def _slot(parameters: ParameterContext, path: str) -> JsonValue:
    """Return a captured module ref or blank when its optional slot is absent."""
    try:
        return {"kind": "ref", "name": parameters.library_key(path)}
    except ModuleSlotError:
        return None


class OvernightMuxToneForm(FormModel):
    """One original qubit: name, calibrated x180 drive and independent x90 pulse.

    Inactive drives may be None. Native core admission requires both pulses for
    selected participants. Ref/inline edits use the host's fixed library capture.
    """

    name: str = Field(min_length=1, description="Original qubit identity")
    drive: Annotated[DriveInput, Display(label="Pi drive (x180)")] = None
    pi2_pulse: Annotated[DriveInput, Display(label="Pi/2 drive (x90)")] = None

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        """Bind both pulse fields to the fixed capture."""
        return {(key,): DRIVE.converter(env) for key in ("drive", "pi2_pulse")}

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        """Offer nullable captured pulse refs and inline pulses."""
        return {(key,): DRIVE.options(env) for key in ("drive", "pi2_pulse")}


class OvernightMuxPlanForm(FormModel):
    """Select mux participants and fixed pulses without flux or calibration.

    Fresh defaults retain all captured mux qubits in readout order, select
    captured $participants (all qubits if absent/unbound), and reference existing
    x180/x90, readout and reset slots. Missing pulses/readout stay unfilled;
    reset may be None. No values are refreshed after the form is built.
    outer_rounds/interval are required; points counts shared delay samples.
    """

    participants: list[int] = Field(min_length=1, description="Original tone indices")
    tones: list[OvernightMuxToneForm] = Field(
        min_length=1, description="Full tone order"
    )
    reset: ResetInput = None
    readout: ReadoutInput
    outer_rounds: int = Field(ge=1, description="Finite number of T1/Echo pairs")
    start_interval_s: Annotated[float, Display(label="Start interval", unit="s")] = (
        Field(gt=0)
    )
    points: int = Field(default=51, ge=5, description="Common delay samples per member")

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Seed identities and existing module keys from one captured mux group.

        Missing capture or missing/unbound mux leaves required inputs blank.
        No module is invented. Unknown participant names stay invalid input;
        other alias/source faults propagate instead of silently selecting qubits.
        """
        data = super().defaults(env)
        parameters = env.parameters
        if parameters is None:
            return data
        try:
            names = parameters.lookup_point("$mux.qubits").value
        except RoleResolutionError as error:
            if error.reason not in ("Unknown alias", "Alias is not bound"):
                raise
            return data
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(n, str) for n in names)
        ):
            return data
        try:
            selection = parameters.lookup_point("$participants").value
        except RoleResolutionError as error:
            if error.reason not in ("Unknown alias", "Alias is not bound"):
                raise
            selection = names
        selected = [selection] if isinstance(selection, str) else selection
        if not isinstance(selected, list):
            raise ValueError("$participants must contain qubit names")
        indices: list[JsonValue] = [
            names.index(n) if n in names else n for n in selected
        ]
        data["participants"] = indices
        data["tones"] = [
            {
                "name": name,
                "drive": _slot(parameters, f"{name}.module.x180")
                if i in indices
                else None,
                "pi2_pulse": _slot(parameters, f"{name}.module.x90")
                if i in indices
                else None,
            }
            for i, name in enumerate(names)
        ]
        data["readout"] = _slot(parameters, "$mux.module.readout")
        data["reset"] = _slot(parameters, "$mux.module.reset")
        return data

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        """Bind mux readout and optional reset to the host's fixed capture."""
        return {("readout",): READOUT.converter(env), ("reset",): RESET.converter(env)}

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        """Offer only mux readout and native reset families."""
        return {("readout",): READOUT.options(env), ("reset",): RESET.options(env)}

    def to_cfg(
        self, env: FormEnv, snapshot: RunSnapshot
    ) -> OvernightMuxPlan | FormErrors:
        """Materialize fixed module inputs into a historical native plan.

        env owns captured library/assets; snapshot is the host capture. Module
        ValueErrors become consuming-field errors; asset I/O and native plan
        errors propagate. No source refresh, device access or calibration occurs.
        """
        context = env.library.validation_context() if env.library is not None else None
        errors: dict[FormPath, str] = {}
        for index in self.participants:
            if not 0 <= index < len(self.tones):
                errors[("participants",)] = (
                    "Select original tone indices within the tone list"
                )
                continue
            tone = self.tones[index]
            for key, pulse in (("drive", tone.drive), ("pi2_pulse", tone.pi2_pulse)):
                if pulse is None:
                    errors[("tones", index, key)] = (
                        "Selected participants require a calibrated pulse"
                    )
        if errors:
            return FormErrors(errors)

        def materialize(
            field: ModuleInputField, value: ModuleInput | None, path: FormPath
        ) -> object:
            if value is None:
                return None
            try:
                core = ModuleCfgFactory.from_raw(field.to_spec(value), context=context)
                return NativeModuleCfgFactory.from_core(core, context=context)
            except ValueError as error:
                errors[path] = str(error)
                return None

        tones = [
            {
                "name": tone.name,
                "drive": materialize(DRIVE, tone.drive, ("tones", i, "drive")),
                "pi2_pulse": materialize(
                    DRIVE, tone.pi2_pulse, ("tones", i, "pi2_pulse")
                ),
            }
            for i, tone in enumerate(self.tones)
        ]
        readout = materialize(READOUT, self.readout, ("readout",))
        reset = materialize(RESET, self.reset, ("reset",))
        if errors:
            return FormErrors(errors)
        return OvernightMuxPlan.model_validate(
            {
                "participants": self.participants,
                "tones": tones,
                "readout": readout,
                "reset": reset,
                "outer_rounds": self.outer_rounds,
                "start_interval_s": self.start_interval_s,
                "points": self.points,
                "snapshot": snapshot,
            }
        )


class OvernightMuxTunablesForm(FormModel):
    """Set averaging and acceptance bounds to resolve mux lifetime drift.

    Enter the longest qubit's T1/Echo guesses for the shared grids. uniform
    controls T1 sampling only; Echo retains its linear delay grid.
    reps/rounds average each native member, not monitoring pairs. Guesses are in
    microseconds; sweep_factor fixes each grid. Both relax delays have a
    relax_min_us floor. T1 uses relax_factor*t1_guess_us. Echo uses this round's
    largest accepted participant T1, or t1_guess_us if none accepted.
    min_lifetime_points defaults to 5 acquired fitted delays in [0, lifetime].
    Invalid quality rejects too. Fits change only Echo relax, not grids or pulses.
    Expressions stay in FormInput; native conversion carries ordinary values.
    """

    uniform: Annotated[bool, Display(label="Uniform (linear) sweep")] = Field(
        default=True,
        description="T1 only; False concentrates delays within the same start/stop window",
    )
    reps: Annotated[int, Display(label="Averages")] = Field(
        default=100, ge=1, description="Native acquisition averages"
    )
    rounds: int = Field(default=2, ge=1, description="Native acquisition rounds")
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

    def to_cfg(self) -> OvernightMuxTunables:
        """Convert validated ordinary knobs; no source, hardware or expression parsing."""
        return OvernightMuxTunables(
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
