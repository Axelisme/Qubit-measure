"""Independent Echo/Ramsey lifetime knob forms for Coherence."""

from typing import Annotated

from pydantic import Field
from zcu_tools.gui.form import Display, FormModel

from zcu_lab.workflows.coherence_fluxdep.t2 import T2EchoTunables, T2RamseyTunables


class T2EchoTunablesForm(FormModel):
    """Ordinary averaging/guess knobs for zero-detune Echo.

    reps/rounds average native Echo. guess_us
    seeds the microsecond delay EWMA;
    sweep_factor/relax_factor are positive timing multipliers. Old T2 rejection
    is fixed science policy, not extra T1 error/RMS fields or tunable thresholds.
    """

    reps: Annotated[int, Display(label="Averages")] = Field(
        default=100, ge=1, description="Echo averages"
    )
    rounds: int = Field(default=10, ge=1, description="Echo averaging rounds")
    guess_us: Annotated[float, Display(label="Initial guess", unit="µs")] = Field(
        default=20.0, gt=0, description="Initial lifetime estimate (µs)"
    )
    sweep_factor: Annotated[float, Display(label="Sweep factor")] = Field(
        default=5.0, gt=0, description="Maximum delay divided by guess"
    )
    relax_factor: Annotated[float, Display(label="Relax factor")] = Field(
        default=3.0, gt=0, description="Relax delay divided by initial guess"
    )

    def to_cfg(self) -> T2EchoTunables:
        """Copy validated ordinary form values; no owner update or source lookup."""
        return T2EchoTunables(
            reps=self.reps,
            rounds=self.rounds,
            guess_us=self.guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
        )


class T2RamseyTunablesForm(FormModel):
    """Ordinary Ramsey knobs with a required detune ratio in cycles per step.

    reps/rounds/guess_us/sweep_factor/relax_factor have the same native
    timing meanings as Echo. detune_ratio is finite, signed or zero, with native default .05; native MHz detune is ratio divided by the delay step in us.
    No guessed detune or extra T1 fit-quality threshold is supplied.
    """

    reps: Annotated[int, Display(label="Averages")] = Field(
        default=100, ge=1, description="Ramsey averages"
    )
    rounds: int = Field(default=10, ge=1, description="Ramsey averaging rounds")
    guess_us: Annotated[float, Display(label="Initial guess", unit="µs")] = Field(
        default=20.0, gt=0, description="Initial lifetime estimate (µs)"
    )
    sweep_factor: Annotated[float, Display(label="Sweep factor")] = Field(
        default=5.0, gt=0, description="Maximum delay divided by guess"
    )
    relax_factor: Annotated[float, Display(label="Relax factor")] = Field(
        default=3.0, gt=0, description="Relax delay divided by initial guess"
    )
    detune_ratio: Annotated[
        float, Display(label="Detune ratio", unit="fringes/step")
    ] = Field(default=0.05, description="Fringes per delay step")

    def to_cfg(self) -> T2RamseyTunables:
        """Copy ordinary knobs to native Ramsey, retaining required detune ratio."""
        return T2RamseyTunables(
            reps=self.reps,
            rounds=self.rounds,
            guess_us=self.guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
            detune_ratio=self.detune_ratio,
        )
