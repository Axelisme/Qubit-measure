"""coherence_fluxdep t1_form helpers for lab workflows."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue
from zcu_tools.gui.form import Display, FormEnv, FormModel

from zcu_lab.workflows.coherence_fluxdep.t1 import T1Tunables


class T1TunablesForm(FormModel):
    """User-facing T1 lifetime knobs; instances hold ordinary values only.

    Fields control T1 generation/acceptance as described in their Fields.
    guess_us has no static default: construction requires it, and its
    initial value comes from defaults. defaults read only captured
    parameters; these knobs need neither library nor device infos.
    min_lifetime_points defaults to 5 fitted delays in [0, lifetime], inclusive.
    Submission belongs to the host.
    """

    reps: Annotated[int, Display(label="Averages")] = Field(
        default=1000, ge=1, description="T1 averages"
    )
    rounds: Annotated[int, Display(label="Rounds")] = Field(
        default=10, ge=1, description="T1 averaging rounds"
    )
    acquire_retry: int = Field(
        default=3, ge=0, description="Retries after a transient acquisition failure"
    )
    earlystop_snr: float | None = Field(
        default=15.0,
        gt=0,
        description="Completed-round trace SNR threshold; None disables early stop",
    )
    guess_us: Annotated[
        float,
        Display(label="Initial guess", unit="µs", placeholder="general.ext.guess_us"),
    ] = Field(gt=0, description="Initial lifetime in µs")
    sweep_factor: Annotated[float, Display(label="Sweep factor")] = Field(
        default=5.0, gt=0, description="Sweep end divided by lifetime guess"
    )
    relax_factor: Annotated[float, Display(label="Relax factor")] = Field(
        default=3.0, gt=0, description="Relax delay divided by lifetime guess"
    )
    sweep_mode: Annotated[Literal["auto", "fixed"], Display(label="Sweep mode")] = (
        "auto"
    )
    sweep_start_us: Annotated[float, Display(label="Sweep start", unit="µs")] = Field(
        default=0.5, ge=0
    )
    sweep_stop_us: Annotated[float, Display(label="Fixed sweep stop", unit="µs")] = (
        Field(default=50.0, gt=0)
    )
    sweep_stop_min_us: Annotated[
        float, Display(label="Auto stop minimum", unit="µs")
    ] = Field(default=1.0, gt=0)
    max_length_us: Annotated[float, Display(label="Auto stop maximum", unit="µs")] = (
        Field(default=50.0, gt=0)
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
    uniform: Annotated[bool, Display(label="Uniform (linear) sweep")] = True
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
        default=0.3, gt=0, le=1, description="Strict T1 error/lifetime upper bound"
    )
    max_fit_residual: Annotated[float, Display(label="Max fit residual")] = Field(
        default=0.2, gt=0, le=1, description="Strict normalized residual RMS bound"
    )

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        """Build defaults, deriving the initial guess from the captured point.

        When parameters contain general.ext.guess_us
        (even explicit null), declare that path as an expression default;
        default_input evaluates it into a direct value. A null, non-numeric
        or non-positive leaf leaves guess_us unfilled, so result reports
        only its "Field required" instead of silently using 20 us. Otherwise
        (no capture or an absent leaf) set an explicit
        ordinary 20 us. Reads use the fixed point only, never
        materialization, devices or I/O. Unavailable captured sources
        propagate their source error.
        """
        data = super().defaults(env)
        if (
            env.parameters is not None
            and env.parameters.lookup_point("general.ext.guess_us").present
        ):
            data["guess_us"] = {"expr": "general.ext.guess_us"}
        else:
            data["guess_us"] = 20.0
        return data

    def to_cfg(self) -> T1Tunables:
        """Copy validated ordinary form values to native T1Tunables.

        Expression input stays in FormInput, not in the native values or Engine
        journal. No owner update or source access occurs. Native validation
        failures propagate after invalid direct edits.
        """
        return T1Tunables(
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
            sweep_stop_min_us=self.sweep_stop_min_us,
            max_length_us=self.max_length_us,
            relax_mode=self.relax_mode,
            relax_delay_us=self.relax_delay_us,
            relax_min_us=self.relax_min_us,
            uniform=self.uniform,
            max_rel_err=self.max_rel_err,
            max_fit_residual=self.max_fit_residual,
        )
