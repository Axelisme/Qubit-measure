"""Live eligibility, spacing and shot controls for raw GE acquisition."""

from typing import Annotated

from pydantic import Field
from zcu_tools.gui.form import Display, FormModel

from zcu_lab.workflows.coherence_fluxdep.singleshot import SingleshotTunables


class SingleshotTunablesForm(FormModel):
    """Edit GE policy using provisional simulation defaults, without analysis.

    Spacing uses the plan flux unit. shots counts each probe-off/on preparation
    separately. No extra Rabi contrast or residual gate selects GE points.
    """

    min_spacing: Annotated[float, Display(label="Minimum flux spacing")] = Field(
        default=0.0, ge=0, description="Minimum GE spacing in plan flux units"
    )
    shots: Annotated[int, Display(label="Shots per preparation")] = Field(
        default=100000, ge=1, description="Shots per probe-off/on cloud"
    )

    def to_cfg(self) -> SingleshotTunables:
        """Copy validated ordinary controls to native policy without I/O."""
        return SingleshotTunables(
            min_spacing=self.min_spacing,
            shots=self.shots,
        )
