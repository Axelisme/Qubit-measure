"""coherence_fluxdep t1_form helpers for lab workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import Field, JsonValue
from zcu_tools.gui.form import Display, FormEnv, FormModel

from zcu_lab.workflows.coherence_fluxdep.t1 import T1Tunables


class T1TunablesForm(FormModel):
    """User-facing T1 lifetime knobs; instances hold ordinary values only.

    Fields control T1 generation/acceptance as
    described in their Fields. defaults read only captured parameters; these
    knobs need neither library nor device infos. Submission belongs to the host.
    """

    reps: Annotated[int, Display(label="Averages")] = Field(
        default=100, ge=1, description="T1 averages"
    )
    rounds: Annotated[int, Display(label="Rounds")] = Field(
        default=10, ge=1, description="T1 averaging rounds"
    )
    guess_us: Annotated[
        float,
        Display(label="Initial guess", unit="µs", placeholder="general.ext.guess_us"),
    ] = Field(default=20.0, gt=0, description="Initial lifetime in µs")
    sweep_factor: Annotated[float, Display(label="Sweep factor")] = Field(
        default=5.0, gt=0, description="Sweep end divided by lifetime guess"
    )
    relax_factor: Annotated[float, Display(label="Relax factor")] = Field(
        default=3.0, gt=0, description="Relax delay divided by lifetime guess"
    )
    max_rel_err: Annotated[float, Display(label="Max relative error")] = Field(
        default=0.3, gt=0, le=1, description="Strict T1 error/lifetime upper bound"
    )
    max_fit_residual: Annotated[float, Display(label="Max fit residual")] = Field(
        default=0.2, gt=0, le=1, description="Strict normalized residual RMS bound"
    )

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Build defaults, retaining a captured guess expression for initial input.

        When previous is empty and parameters contain general.ext.guess_us
        (even explicit null),
        retain that path as an expression rather than copying its value. With
        no capture or an absent leaf, keep the provisional ordinary 20 us.
        Invalid referenced values remain result errors, not a fallback. Reads
        use the fixed point only, never materialization, devices or I/O.
        Unavailable captured sources propagate their source error.
        """
        data = super().defaults(env, previous)
        if (
            not previous
            and env.parameters is not None
            and env.parameters.lookup_point("general.ext.guess_us").present
        ):
            data["guess_us"] = {"expr": "general.ext.guess_us"}
        return data

    def to_cfg(self) -> T1Tunables:
        """Copy validated ordinary form values to native T1Tunables.

        Expression input stays in FormInput, not in the native values or Engine
        journal. No owner update or source access occurs. Native validation
        failures propagate after invalid direct edits.
        """
        return T1Tunables(
            reps=self.reps,
            rounds=self.rounds,
            guess_us=self.guess_us,
            sweep_factor=self.sweep_factor,
            relax_factor=self.relax_factor,
            max_rel_err=self.max_rel_err,
            max_fit_residual=self.max_fit_residual,
        )
