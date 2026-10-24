"""Independent local line DC scan with mux twotone spectra."""

from typing import ClassVar

from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext

from zcu_lab.v2.support.mux_flux import (
    FluxFrequencyCfg,
    LocalFluxCfg,
    MuxFluxResult,
    mux_flux_spec,
    run_flux_scan,
)


class TwotoneFluxLocalCfg(LocalFluxCfg[FluxFrequencyCfg]):
    """Full mux twotone settings and local native-unit DC scan.

    Inherited tones retain original readout identity; participants determines
    acquisition order. See LocalFluxCfg for range, unit and settle contracts.
    """


class TwotoneFluxLocalExp(Experiment[MuxFluxResult, TwotoneFluxLocalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, TwotoneFluxLocalCfg]] = (
        mux_flux_spec(TwotoneFluxLocalCfg, tag="mux/twotone_flux_local")
    )

    def run(self, config: TwotoneFluxLocalCfg, *, context: RunContext) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Stop discards only the unfinished flux point; driver errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        return run_flux_scan(config, context=context)
