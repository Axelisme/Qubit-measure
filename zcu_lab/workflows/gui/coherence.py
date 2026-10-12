"""Coherence's shared calibration and independent lifetime knob attachment."""

from collections.abc import Mapping
from typing import Annotated

from pydantic import Field, JsonValue
from zcu_tools.gui.form import Display, FormEnv, FormModel

from ..coherence import CoherenceTunables
from .t2 import T2EchoTunablesForm, T2RamseyTunablesForm
from .tunables import CalibrationTunablesForm, T1TunablesForm


class CoherenceTunablesForm(FormModel):
    """Ordinary captured policies; expressions remain in the host FormInput.

    cal controls shared calibration. t1, t2ramsey and t2echo each supply their
    own averaging/initial guess/timing/acceptance inputs. Ramsey detune_ratio
    defaults to .05 fringes/step and is validated even when Ramsey is disabled.
    Defaults use the existing nested form hooks and fixed source capture.
    This form does not capture sources, submit revisions or control hardware.
    """

    cal: Annotated[
        CalibrationTunablesForm, Display(label="Calibration", collapsed=True)
    ] = Field(default_factory=CalibrationTunablesForm)
    t1: Annotated[T1TunablesForm, Display(label="T1", collapsed=True)] = Field(
        default_factory=T1TunablesForm
    )
    t2ramsey: Annotated[
        T2RamseyTunablesForm, Display(label="Ramsey", collapsed=True)
    ] = Field(default_factory=T2RamseyTunablesForm)
    t2echo: Annotated[T2EchoTunablesForm, Display(label="Echo", collapsed=True)] = (
        Field(default_factory=T2EchoTunablesForm)
    )

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Retain T1's captured initial-guess expression at its new nested path.

        Other lifetime defaults remain native ordinary values. Existing input
        overlays remain the host's responsibility; no source is recaptured.
        """
        data = super().defaults(env, previous)
        prior_t1 = previous.get("t1", {})
        data["t1"] = T1TunablesForm.defaults(
            env, prior_t1 if isinstance(prior_t1, dict) else {}
        )
        return data

    def to_cfg(self) -> CoherenceTunables:
        """Convert only validated ordinary values, without recapture or submission."""
        return CoherenceTunables(
            cal=self.cal.to_cfg(),
            t1=self.t1.to_cfg(),
            t2ramsey=self.t2ramsey.to_cfg(),
            t2echo=self.t2echo.to_cfg(),
        )
