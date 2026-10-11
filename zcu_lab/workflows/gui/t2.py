"""Independent T2 plan and Echo/Ramsey tunables attachments."""

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, JsonValue
from zcu_tools.datafile import RunSnapshot
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormModel,
    FormPath,
    ModuleInput,
    RangeInput,
)

from ..t2 import T2EchoTunables, T2Plan, T2RamseyTunables
from .calibration_modules import (
    calibration_choices,
    calibration_defaults,
    expand_fluxes,
    resolve_calibration_modules,
)
from .tunables import CalibrationTunablesForm


class T2PlanForm(FormModel):
    """Start-time fixed T2 flux grid and shared calibration module selections.

    flux_dev/unit/fluxes are explicit connected driver coordinates, not a live
    membership check or conversion. points is delay resolution (>=5).
    qub_pulse/readout are ref/inline modules resolved in the fixed capture;
    pure arb Length-Rabi drive is unsupported. pi_length_us is the required
    initial Rabi-range estimate, consumed by the four-scalar host seed only.
    Runtime calibration supplies fitted pi/2; no pi_length_us/2 is derived here.
    """

    flux_dev: str = Field(min_length=1, description="Connected flux driver name")
    flux_unit: Literal["A", "V", "native"] = Field(
        description="Explicit setpoint unit, no conversion"
    )
    fluxes: RangeInput = Field(description="Ordered absolute setpoints")
    pi_length_us: float = Field(
        gt=0, description="Initial Length-Rabi pi estimate (us)"
    )
    points: int = Field(default=51, ge=5, description="Uniform T2 delay samples")
    qub_pulse: ModuleInput = Field(description="Shared spectroscopy/Rabi/pi drive")
    readout: ModuleInput = Field(description="Shared pulse-backed readout")

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Use fixed flux alias/x180 pi estimate defaults without live I/O or guesses."""
        return calibration_defaults(super().defaults(env, previous), env, previous)

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer captured pulse/readout module roots without materialization."""
        return calibration_choices(env)

    def to_cfg(self, env: FormEnv, snapshot: RunSnapshot) -> T2Plan | FormErrors:
        """Resolve modules/setpoints in env and attach historical snapshot to T2Plan.

        Return consuming-field FormErrors for invalid calibration modules. Native
        asset I/O/model errors propagate. No fit, cfg grid or capture occurs.
        """
        modules = resolve_calibration_modules(
            qub_pulse=self.qub_pulse, readout=self.readout, env=env
        )
        if isinstance(modules, FormErrors):
            return modules
        return T2Plan(
            flux_dev=self.flux_dev,
            flux_unit=self.flux_unit,
            fluxes=expand_fluxes(self.fluxes),
            points=self.points,
            qub_pulse=modules.qub_pulse,
            readout=modules.readout,
            snapshot=snapshot,
        )


class T2EchoTunablesForm(FormModel):
    """Ordinary calibration/averaging/guess knobs for zero-detune Echo.

    cal controls calibration policy; reps/rounds average native Echo. guess_us
    seeds the microsecond delay EWMA and existing calibration/relax estimate;
    sweep_factor/relax_factor are positive timing multipliers. Old T2 rejection
    is fixed science policy, not extra T1 error/RMS fields or tunable thresholds.
    """

    cal: CalibrationTunablesForm = Field(default_factory=CalibrationTunablesForm)
    reps: int = Field(default=100, ge=1, description="Echo averages")
    rounds: int = Field(default=10, ge=1, description="Echo averaging rounds")
    guess_us: float = Field(
        default=20.0, gt=0, description="Initial lifetime estimate (us)"
    )
    sweep_factor: float = Field(
        default=5.0, gt=0, description="Maximum delay divided by guess"
    )
    relax_factor: float = Field(
        default=3.0, gt=0, description="Relax delay divided by initial guess"
    )

    def to_cfg(self) -> T2EchoTunables:
        """Copy validated ordinary form values; no owner update or source lookup."""
        return T2EchoTunables(
            cal=self.cal.to_cfg(),
            reps=self.reps,
            rounds=self.rounds,
            guess_us=self.guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
        )


class T2RamseyTunablesForm(FormModel):
    """Ordinary Ramsey knobs with a required detune ratio in cycles per step.

    cal/reps/rounds/guess_us/sweep_factor/relax_factor have the same calibrated
    timing meanings as Echo. detune_ratio is finite, signed or zero, and must
    be supplied; native MHz detune is ratio divided by the delay step in us.
    No guessed detune or extra T1 fit-quality threshold is supplied.
    """

    cal: CalibrationTunablesForm = Field(default_factory=CalibrationTunablesForm)
    reps: int = Field(default=100, ge=1, description="Ramsey averages")
    rounds: int = Field(default=10, ge=1, description="Ramsey averaging rounds")
    guess_us: float = Field(
        default=20.0, gt=0, description="Initial lifetime estimate (us)"
    )
    sweep_factor: float = Field(
        default=5.0, gt=0, description="Maximum delay divided by guess"
    )
    relax_factor: float = Field(
        default=3.0, gt=0, description="Relax delay divided by initial guess"
    )
    detune_ratio: float = Field(description="Caller-required cycles per delay step")

    def to_cfg(self) -> T2RamseyTunables:
        """Copy ordinary knobs to native Ramsey, retaining required detune ratio."""
        return T2RamseyTunables(
            cal=self.cal.to_cfg(),
            reps=self.reps,
            rounds=self.rounds,
            guess_us=self.guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
            detune_ratio=self.detune_ratio,
        )
