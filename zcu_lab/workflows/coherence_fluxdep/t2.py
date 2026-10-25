"""Independent Ramsey and Echo averaging, timing and fit tunables."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class T2EchoTunables(BaseModel):
    """Lifetime generation knobs captured per T2 invocation.

    reps/rounds control T2 averaging; shared calibration is owned by Coherence. guess_us
    seeds the delay-history EWMA in microseconds; sweep_factor multiplies it
    into maximum delay, capped at max_length_us. Auto relax uses relax_factor
    times the caller's T1 estimate, floored at relax_min_us. Fixed modes use
    sweep_start_us/sweep_stop_us and relax_delay_us without auto clamps.
    detune_ratio adds fringes per delay step; zero selects pure Echo decay.
    min_lifetime_points defaults to 5 fitted delays in [0, lifetime], including
    endpoints. Invalid quality diagnostics and undersampling reject before the
    twice-window and mean-error rules. Ramsey inherits the same gates.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    reps: int = Field(default=1000, ge=1)
    rounds: int = Field(default=10, ge=1)
    acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    earlystop_snr: float | None = Field(
        default=20.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    guess_us: float = Field(default=20.0, gt=0)
    sweep_factor: float = Field(default=5.0, gt=0)
    relax_factor: float = Field(default=3.0, gt=0)
    sweep_mode: Literal["auto", "fixed"] = Field(
        default="auto", description="History-based or fixed delay window"
    )
    sweep_start_us: float = Field(
        default=0.0, ge=0, description="First delay in us, in either mode"
    )
    sweep_stop_us: float = Field(
        default=12.5, gt=0, description="Fixed last delay in us"
    )
    max_length_us: float = Field(
        default=12.5, gt=0, description="Auto sweep stop upper bound in us"
    )
    relax_mode: Literal["auto", "fixed"] = Field(
        default="auto", description="T1-based or fixed relaxation"
    )
    relax_delay_us: float = Field(
        default=30.0, ge=0, description="Fixed relaxation delay in us"
    )
    relax_min_us: float = Field(
        default=1.0, ge=0, description="Auto relaxation lower bound in us"
    )
    max_model_window_factor: float = Field(
        default=2.0,
        gt=0,
        description="Diagnostic lifetime / maximum fitted delay upper bound; no feedback change",
    )
    min_lifetime_points: int = Field(
        default=5,
        ge=1,
        description="Minimum fitted delay points with 0 <= delay <= lifetime",
    )
    detune_ratio: float = Field(
        default=0.1,
        description="Artificial fringes per delay step; zero selects pure decay",
    )

    @model_validator(mode="after")
    def validate_window(self) -> T2EchoTunables:
        """Reject fixed windows or auto caps that cannot exceed the start."""
        stop = self.sweep_stop_us if self.sweep_mode == "fixed" else self.max_length_us
        if stop <= self.sweep_start_us:
            raise ValueError("T2 stop/cap must exceed start")
        return self


class T2RamseyTunables(T2EchoTunables):
    """Ramsey generation adds finite detune_ratio, in fringes per step.

    Native detune in MHz equals detune_ratio divided by the requested delay
    step in microseconds. The finite ratio may be zero or signed; there is no
    guessed detune. The native default is .05; nonfinite inputs are rejected.
    All inherited fields retain their Echo timing meanings.
    """

    detune_ratio: float = 0.05
