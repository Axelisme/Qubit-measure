"""Measure registration for mux onetone versus global DC bias."""

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from zcu_tools.gui.app.measure.adapter import AdapterCapabilities
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_flux_adapter import MuxFluxAdapter

from .core import OnetoneFluxGlobalCfg, OnetoneFluxGlobalExp
from .form import make_form


class OnetoneFluxGlobalAdapter(MuxFluxAdapter[OnetoneFluxGlobalCfg]):
    """Acquire mux maps and calibrate one original participant per Analyze."""

    capabilities: ClassVar[AdapterCapabilities] = replace(
        MuxFluxAdapter.capabilities, load_data=True
    )
    experiment_tag: ClassVar[str | None] = OnetoneFluxGlobalExp.persistence_tag()
    exp_cls = OnetoneFluxGlobalExp
    force_magnitude: ClassVar[bool] = True
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[OnetoneFluxGlobalCfg]:
        """Prepare full mux identities, spectrum and global DC range candidates."""
        return make_form(ctx)
