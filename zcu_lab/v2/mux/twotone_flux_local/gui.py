"""Measure registration for mux twotone versus local DC bias."""

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from zcu_tools.gui.app.measure.adapter import AdapterCapabilities
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_flux_adapter import MuxFluxAdapter

from .core import TwotoneFluxLocalCfg, TwotoneFluxLocalExp
from .form import make_form


class TwotoneFluxLocalAdapter(MuxFluxAdapter[TwotoneFluxLocalCfg]):
    """Acquire mux maps and calibrate one original participant per Analyze."""

    capabilities: ClassVar[AdapterCapabilities] = replace(
        MuxFluxAdapter.capabilities, load_data=True
    )
    exp_cls = TwotoneFluxLocalExp
    force_magnitude: ClassVar[bool] = False
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[TwotoneFluxLocalCfg]:
        """Prepare full mux identities, spectrum and local DC range candidates."""
        return make_form(ctx)
