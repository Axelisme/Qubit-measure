"""Captured mux onetone and global DC form defaults."""

from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_flux_forms import make_flux_form

from .core import OnetoneFluxGlobalCfg


def make_form(ctx: FormContext) -> Form[OnetoneFluxGlobalCfg]:
    """Use Open/Reset capture for identities, DC ranges and readout candidates."""
    return make_flux_form(ctx, OnetoneFluxGlobalCfg, local=False, twotone=False)
