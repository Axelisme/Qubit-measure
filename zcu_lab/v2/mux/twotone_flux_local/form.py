"""Captured mux twotone and local DC form defaults."""

from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_flux_forms import make_flux_form

from .core import TwotoneFluxLocalCfg


def make_form(ctx: FormContext) -> Form[TwotoneFluxLocalCfg]:
    """Use Open/Reset capture for identities, DC ranges and readout candidates."""
    return make_flux_form(ctx, TwotoneFluxLocalCfg, local=True, twotone=True)
