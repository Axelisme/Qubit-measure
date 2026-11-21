"""Shared two-qubit target binding, selected writes and report conventions.

Analyze into the owner's named type. Forward its fit quality and reasons
unchanged, and pass each scalar through recipe_api.report_number. Leave base
fit_quality null when there is no primary fit; forward per-fit diagnostics
beside the corresponding estimates. Do not infer a fit or add a threshold.
"""

from pydantic import Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeTab, WritebackAnswer


class TwoQubitTarget(FormModel):
    """Explicit role bindings for this Reset only; null leaves a role unchanged."""

    f1: str | None = Field(default=None, description="Fluxonium in the F1 role.")
    f2: str | None = Field(default=None, description="Fluxonium in the F2 role.")
    coupler: str | None = Field(default=None, description="Coupler transmon.")
    mux: str | None = Field(default=None, description="Mux group containing F1 and F2.")

    def aliases(self) -> dict[str, str]:
        """Return non-null role names for open_tab, without inferring any roles."""
        return {
            role: name
            for role, name in (
                ("f1", self.f1),
                ("f2", self.f2),
                ("coupler", self.coupler),
                ("mux", self.mux),
            )
            if name is not None
        }


class TwoQubitPositionTarget(FormModel):
    """Bind positions on this Open/Reset only; null leaves a binding unchanged."""

    qubit1: str | None = Field(default=None, description="Component at Qubit1.")
    qubit2: str | None = Field(default=None, description="Component at Qubit2.")
    coupler: str | None = Field(
        default=None, description="Component holding the pair's CZ calibration."
    )
    mux: str | None = Field(
        default=None, description="Mux group containing Qubit1 and Qubit2."
    )

    def aliases(self) -> dict[str, str]:
        """Return only supplied alias bindings for open_tab, without inference."""
        return {
            alias: name
            for alias, name in (
                ("qubit1", self.qubit1),
                ("qubit2", self.qubit2),
                ("coupler", self.coupler),
                ("mux", self.mux),
            )
            if name is not None
        }


def apply_selected_writeback(tab: RecipeTab, decision: WritebackAnswer) -> bool:
    """Apply only the answered selection from this tab's latest writeback ask.

    Skip leaves calibration unchanged. Raise on a failed receipt without
    retrying. Return true only if the completed receipt confirms a write.
    """
    if not decision.accepted:
        return False
    receipt = tab.accept_writeback(decision.items)
    receipt.raise_for_error()
    return bool(receipt.items)
