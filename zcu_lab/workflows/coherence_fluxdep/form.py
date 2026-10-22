"""coherence_fluxdep form helpers for lab workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator
from zcu_tools.datafile.native_models import RunSnapshot
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
from zcu_tools.gui.form_adapter import SweepInput
from zcu_tools.resources.entry import RoleResolutionError

from zcu_lab.workflows.coherence_fluxdep.t1_form import T1TunablesForm
from zcu_lab.workflows.coherence_fluxdep.t2_form import (
    T2EchoTunablesForm,
    T2RamseyTunablesForm,
)
from zcu_lab.workflows.coherence_fluxdep.workflow import (
    CoherencePlan,
    CoherenceTunables,
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


class CoherenceTunablesForm(FormModel):
    """Ordinary captured policies; expressions remain in the host FormInput.

    cal controls shared calibration. t1, t2ramsey and t2echo each supply their
    own averaging/initial guess/timing/acceptance inputs. t1 has no static
    default because T1's guess_us has none; defaults supplies it, so direct
    construction must pass t1. Ramsey detune_ratio defaults to .05
    fringes/step and is validated even when Ramsey is disabled. Defaults use
    the existing nested form hooks and fixed source capture. This form does
    not capture sources, submit revisions or control hardware.
    """

    cal: Annotated[
        CalibrationTunablesForm, Display(label="Calibration", collapsed=True)
    ] = Field(default_factory=CalibrationTunablesForm)
    t1: Annotated[T1TunablesForm, Display(label="T1", collapsed=True)]
    t2ramsey: Annotated[
        T2RamseyTunablesForm, Display(label="Ramsey", collapsed=True)
    ] = Field(default_factory=T2RamseyTunablesForm)
    t2echo: Annotated[T2EchoTunablesForm, Display(label="Echo", collapsed=True)] = (
        Field(default_factory=T2EchoTunablesForm)
    )

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Fill t1 from T1TunablesForm.defaults, including its initial guess.

        t1 has no static default, so this is its only default source; the
        T1 initial guess follows T1TunablesForm.defaults (captured leaf
        evaluated into a value, or 20 us without capture/leaf). cal, t2ramsey
        and t2echo keep their static defaults. Existing input overlays remain
        the host's responsibility; no source is recaptured.
        """
        data = super().defaults(env)
        data["t1"] = T1TunablesForm.defaults(env)
        return data

    def to_cfg(self) -> CoherenceTunables:
        """Convert only validated ordinary values, without recapture or submission."""
        return CoherenceTunables(
            cal=self.cal.to_cfg(),
            t1=self.t1.to_cfg(),
            t2ramsey=self.t2ramsey.to_cfg(),
            t2echo=self.t2echo.to_cfg(),
        )


class CoherencePlanForm(FormModel):
    """Start-time setpoints and two shared module reference/inline selections.

    flux_dev is the resolved host device name; new input defaults to a literal
    resolved from $flux_dev in the fixed alias capture. Required flux_unit declares A, V or native.
    fluxes contain absolute values in that coordinate; no conversion occurs.
    points >= 5 controls runtime T1 delay resolution. qub_pulse selects the
    shared pulse drive; readout selects a pulse-backed readout because native
    readout landscape acquisition requires gain control. Both use ModuleInput
    wire envelopes and the fixed host library, never a live lookup. pi_length_us
    is the required initial Rabi-range estimate, not an accepted calibration;
    host seed composition must consume it separately from the native plan. Required
    selections stay unfilled until supplied. Runtime owns every cfg and grid.
    """

    flux_dev: Annotated[str, Display(label="Flux device")] = Field(
        min_length=1, description="Host flux device name"
    )
    flux_unit: Annotated[Literal["A", "V", "native"], Display(label="Flux unit")] = (
        Field(description="Explicit flux coordinate: YOKO A/V, Fake native")
    )
    fluxes: Annotated[SweepInput, Display(label="Fluxes")] = Field(
        description="Ordered absolute device setpoints"
    )
    pi_length_us: Annotated[float, Display(label="Pi length", unit="µs")] = Field(
        gt=0,
        description="Initial pi-duration estimate in µs at the selected drive gain",
    )
    points: int = Field(default=51, ge=5, description="Uniform T1 delay samples")
    qub_pulse: Annotated[DriveModuleInput, Display(label="Drive pulse")] = Field(
        description="Shared spectroscopy/Rabi/pi drive"
    )
    readout: Annotated[ReadoutModuleInput, Display(label="Readout")] = Field(
        description="Shared pulse-backed readout"
    )

    measure_t1: bool = Field(default=True, description="Measure T1 at each flux point")
    measure_t2ramsey: Annotated[bool, Display(label="Measure Ramsey")] = Field(
        default=True, description="Measure Ramsey at each flux point"
    )
    measure_t2echo: Annotated[bool, Display(label="Measure Echo")] = Field(
        default=True, description="Measure Echo at each flux point"
    )

    @model_validator(mode="after")
    def _enabled(self) -> CoherencePlanForm:
        if not (self.measure_t1 or self.measure_t2ramsey or self.measure_t2echo):
            raise field_error(("measure_t1",), "At least one lifetime must be enabled")
        return self

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Prefill the captured device name and x180 pi estimate.

        Resolve $flux_dev only for new defaults and keep the name literal.
        Missing captures or unbound aliases keep the $flux_dev expression
        default, which default_input cannot evaluate, so flux_dev stays
        unfilled and required, without a device fallback. Device membership is checked only when used.
        On initial input only, resolve $qubit.module.x180.waveform.length in
        the fixed alias capture and declare its concrete path as an expression
        default if readable and non-null; default_input stores its value. No
        capture, absent/null values or unreadable paths leave the field
        unfilled for explicit user input. Other modules and simulator constants
        are never consulted. Invalid readable values leave it unfilled too,
        not a fallback.
        No live selection, YAML refresh or device access occurs.
        """
        data = calibration_defaults(super().defaults(env), env)
        if env.parameters is None:
            return data
        try:
            data["flux_dev"] = env.parameters.resolve_path("$flux_dev")
        except RoleResolutionError as error:
            if error.reason not in {"Alias is not bound", "Unknown alias"}:
                raise
            # Keep the $flux_dev expression default; flux_dev then stays unfilled.
        return data

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer captured pulse/readout roots without materializing modules.

        Inline-only source-free drafts need no library. A used reference still
        requires the core ModuleRef source check; no live lookup or fallback.
        """
        return calibration_choices(env)

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        """Bind each generated field's converter to the host's fixed capture."""
        return {
            ("qub_pulse",): DRIVE_MODULE_FIELD.converter(env),
            ("readout",): READOUT_MODULE_FIELD.converter(env),
        }

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        """Declare generated module selections against the same fixed capture."""
        return {
            ("qub_pulse",): DRIVE_MODULE_FIELD.options(env),
            ("readout",): READOUT_MODULE_FIELD.options(env),
        }

    def to_cfg(self, env: FormEnv, snapshot: RunSnapshot) -> CoherencePlan | FormErrors:
        """Resolve selections and expand setpoints into a static native plan.

        env supplies the captured library and validation/assets context; pure
        inline modules can omit the library. snapshot is fixed historical
        entry/point/roles supplied by the host. Return consuming-field errors
        for missing refs, wrong families or materialization failure. Convert
        through the native factory with the same captured context, so arb paths
        (including flat_top raises) stay bound to its asset directory. Native
        asset I/O failures propagate; no global scope or recipe interpretation.
        Pure arb drive returns a qub_pulse field error because Length-Rabi must
        vary its duration; flat_top arb ramps and arb readout remain supported.
        Keep descending/repeated/list order and a single range point. No acquisition
        cfg, tunables, timing/grid generation, capture, submission or extra copy
        layer is involved; Engine owns plan isolation.
        """
        modules = resolve_calibration_modules(
            qub_pulse=self.qub_pulse, readout=self.readout, env=env
        )
        if isinstance(modules, FormErrors):
            return modules
        return CoherencePlan(
            measure_t1=self.measure_t1,
            measure_t2ramsey=self.measure_t2ramsey,
            measure_t2echo=self.measure_t2echo,
            flux_dev=self.flux_dev,
            flux_unit=self.flux_unit,
            fluxes=expand_fluxes(self.fluxes),
            points=self.points,
            qub_pulse=modules.qub_pulse,
            readout=modules.readout,
            snapshot=snapshot,
        )
