"""Captured mux onetone and local DC form defaults."""

from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_flux_forms import make_flux_form

from .core import OnetoneFluxLocalCfg


def make_form(ctx: FormContext) -> Form[OnetoneFluxLocalCfg]:
    """Use Open/Reset capture for identities, DC ranges and readout candidates."""
    return make_flux_form(ctx, OnetoneFluxLocalCfg, local=True, twotone=False)
