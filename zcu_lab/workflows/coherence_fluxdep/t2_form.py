"""Independent Echo/Ramsey lifetime knob forms for Coherence."""

from typing import Annotated, Literal

from pydantic import Field
from zcu_tools.gui.form import Display, FormModel

from zcu_lab.workflows.coherence_fluxdep.t2 import T2EchoTunables, T2RamseyTunables


class T2EchoTunablesForm(FormModel):
    """T2 averaging, independent delay history and T1-based relaxation.

    Auto sweep caps sweep_factor*delay_history at max_length_us; auto relax
    floors relax_factor*T1 at relax_min_us. Fixed modes use the explicit
    sweep endpoints or relax_delay_us without clamps. Echo detune_ratio=0
    selects pure decay; a nonzero ratio selects a fringe fit.
    min_lifetime_points defaults to 5 fitted delays in [0, lifetime], inclusive.
    """

    reps: Annotated[int, Display(label="Averages")] = Field(
        default=1000, ge=1, description="T2 averages"
    )
    rounds: Annotated[int, Display(label="Rounds")] = Field(
        default=10, ge=1, description="T2 averaging rounds"
    )
    acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    earlystop_snr: float | None = Field(
        default=20.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    guess_us: Annotated[float, Display(label="Initial guess", unit="µs")] = Field(
        default=20.0, gt=0, description="Delay-history seed"
    )
    sweep_factor: Annotated[float, Display(label="Sweep factor")] = Field(
        default=5.0, gt=0, description="Auto stop divided by delay-history estimate"
    )
    relax_factor: Annotated[float, Display(label="Relax factor")] = Field(
        default=3.0, gt=0, description="Auto relax delay divided by T1 estimate"
    )
    sweep_mode: Annotated[Literal["auto", "fixed"], Display(label="Sweep mode")] = (
        "auto"
    )
    sweep_start_us: Annotated[float, Display(label="Sweep start", unit="µs")] = Field(
        default=0.0, ge=0
    )
    sweep_stop_us: Annotated[float, Display(label="Fixed sweep stop", unit="µs")] = (
        Field(default=12.5, gt=0)
    )
    max_length_us: Annotated[float, Display(label="Auto stop maximum", unit="µs")] = (
        Field(default=12.5, gt=0)
    )
    relax_mode: Annotated[Literal["auto", "fixed"], Display(label="Relax mode")] = (
        "auto"
    )
    relax_delay_us: Annotated[float, Display(label="Fixed relax delay", unit="µs")] = (
        Field(default=30.0, ge=0)
    )
    relax_min_us: Annotated[float, Display(label="Auto relax minimum", unit="µs")] = (
        Field(default=1.0, ge=0)
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
    detune_ratio: Annotated[
        float, Display(label="Detune ratio", unit="fringes/step")
    ] = Field(
        default=0.1,
        description="Artificial fringes per delay step; zero selects pure Echo decay",
    )

    def to_cfg(self) -> T2EchoTunables:
        """Copy ordinary values; native validation rejects invalid timing windows."""
        return T2EchoTunables(
            max_model_window_factor=self.max_model_window_factor,
            min_lifetime_points=self.min_lifetime_points,
            reps=self.reps,
            rounds=self.rounds,
            acquire_retry=self.acquire_retry,
            earlystop_snr=self.earlystop_snr,
            guess_us=self.guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
            sweep_mode=self.sweep_mode,
            sweep_start_us=self.sweep_start_us,
            sweep_stop_us=self.sweep_stop_us,
            max_length_us=self.max_length_us,
            relax_mode=self.relax_mode,
            relax_delay_us=self.relax_delay_us,
            relax_min_us=self.relax_min_us,
            detune_ratio=self.detune_ratio,
        )


class T2RamseyTunablesForm(T2EchoTunablesForm):
    """Ramsey uses the same timing controls, with .05 artificial fringes per step."""

    detune_ratio: Annotated[
        float, Display(label="Detune ratio", unit="fringes/step")
    ] = Field(default=0.05, description="Artificial fringes per delay step")

    def to_cfg(self) -> T2RamseyTunables:
        """Copy ordinary values, retaining the Ramsey detune ratio."""
        return T2RamseyTunables(
            max_model_window_factor=self.max_model_window_factor,
            min_lifetime_points=self.min_lifetime_points,
            reps=self.reps,
            rounds=self.rounds,
            acquire_retry=self.acquire_retry,
            earlystop_snr=self.earlystop_snr,
            guess_us=self.guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
            sweep_mode=self.sweep_mode,
            sweep_start_us=self.sweep_start_us,
            sweep_stop_us=self.sweep_stop_us,
            max_length_us=self.max_length_us,
            relax_mode=self.relax_mode,
            relax_delay_us=self.relax_delay_us,
            relax_min_us=self.relax_min_us,
            detune_ratio=self.detune_ratio,
        )
