"""Independent fixed-flux overnight plan/knobs forms; no calibration inputs."""

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, JsonValue
from zcu_tools.cfg_model import ModuleCfgFactory, PulseReadoutSpec, PulseSpec
from zcu_tools.datafile import RunSnapshot
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormModel,
    FormPath,
    ModuleInput,
    ModuleRef,
)
from zcu_tools.program.v2.modules import ModuleCfgFactory as NativeModuleCfgFactory
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.workflows.overnight.workflow import OvernightPlan, OvernightTunables


class OvernightPlanForm(FormModel):
    """Fixed monitoring inputs, resolved only in the supplied host capture.

    flux_dev defaults to the captured $flux_dev expression; flux_unit is required
    A/V/native with no conversion or membership precheck. flux is its absolute
    fixed setpoint. outer_rounds is finite pair count; start_interval_s is actual
    start spacing in seconds; points is delay samples per native acquisition.
    qub_pulse/readout are shared ref/inline modules, already calibrated; no pi
    estimate, frequency, Rabi or readout calibration is added. Native plan
    receives the historical host snapshot separately, not a live source.
    """

    flux_dev: str = Field(min_length=1, description="Connected flux driver name")
    flux_unit: Literal["A", "V", "native"] = Field(
        description="Explicit flux unit, without conversion"
    )
    flux: float = Field(description="Fixed absolute setpoint in flux_unit")
    outer_rounds: int = Field(ge=1, description="Finite number of T1/Echo pairs")
    start_interval_s: float = Field(
        gt=0, description="Actual pair start spacing in seconds"
    )
    points: int = Field(
        default=51, ge=5, description="Uniform delay samples per member"
    )
    qub_pulse: ModuleInput = Field(description="Fixed calibrated pi drive")
    readout: ModuleInput = Field(description="Fixed pulse-backed readout")

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Keep the explicit flux alias expression; leave required monitoring inputs empty."""
        data = super().defaults(env, previous)
        data["flux_dev"] = {"expr": "$flux_dev"}
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

    def to_cfg(self, env: FormEnv, snapshot: RunSnapshot) -> OvernightPlan | FormErrors:
        """Resolve fixed modules through the captured native materialization owner.

        env supplies library/assets context for ref/inline modules. snapshot is
        historical entry/point/roles. Missing references, wrong families and
        materialization ValueErrors return consuming-field FormErrors; asset I/O
        and native plan validation errors propagate. No capture, live read,
        calibration, complete acquisition cfg or fit feedback occurs. Fixed arb
        drive is supported because this workflow never changes pulse duration.
        """
        context = env.library.validation_context() if env.library is not None else None
        errors: dict[FormPath, str] = {}
        drive = None
        readout = None
        try:
            core = ModuleCfgFactory.from_raw(
                self.qub_pulse.name
                if isinstance(self.qub_pulse, ModuleRef)
                else self.qub_pulse.module,
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
                self.readout.name
                if isinstance(self.readout, ModuleRef)
                else self.readout.module,
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
    """Independent overnight averaging, fixed timing and scalar fit thresholds.

    reps/rounds average each native member, not monitoring pairs. Guesses are in
    microseconds; sweep/relax factors multiply their fixed timing. Dimensionless
    fit thresholds only accept scalar summaries; no fit changes cfg. Expressions
    stay in FormInput, while this model and native conversion carry ordinary values.
    """

    reps: int = Field(default=100, ge=1, description="Native acquisition averages")
    rounds: int = Field(default=2, ge=1, description="Native acquisition rounds")
    t1_guess_us: float = Field(
        default=20.0, gt=0, description="Fixed T1 timing guess, microseconds"
    )
    t2_guess_us: float = Field(
        default=10.0, gt=0, description="Fixed Echo timing guess, microseconds"
    )
    sweep_factor: float = Field(
        default=4.0, gt=0, description="Delay endpoint / timing guess"
    )
    relax_factor: float = Field(
        default=5.0, gt=0, description="Relaxation delay / timing guess"
    )
    max_rel_err: float = Field(
        default=0.5, gt=0, description="Strict error/lifetime upper bound"
    )
    max_fit_residual: float = Field(
        default=0.1, gt=0, description="Strict normalized residual RMS upper bound"
    )

    def to_cfg(self) -> OvernightTunables:
        """Convert validated ordinary knobs; no source, hardware or expression parsing."""
        return OvernightTunables(
            reps=self.reps,
            rounds=self.rounds,
            t1_guess_us=self.t1_guess_us,
            t2_guess_us=self.t2_guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
            max_rel_err=self.max_rel_err,
            max_fit_residual=self.max_fit_residual,
        )
