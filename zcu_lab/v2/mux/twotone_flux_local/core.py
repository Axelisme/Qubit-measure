"""Independent local line DC scan with mux twotone spectra."""

from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.context import RunContext

from zcu_lab.v2._support.mux_flux import (
    FluxFrequencyCfg,
    LocalFluxCfg,
    MuxFluxResult,
    mux_flux_spec,
    run_flux_scan,
)
from zcu_lab.v2.mux.twotone_spectrum.core import MuxFreqCfg


class TwotoneFluxLocalCfg(LocalFluxCfg[MuxFreqCfg]):
    """Full mux twotone settings and local native-unit DC scan.

    Inherited tones retain original readout identity; participants determines
    acquisition order. See LocalFluxCfg for range, unit and settle contracts.
    """

    spectrum: FluxFrequencyCfg


class TwotoneFluxLocalExp(PersistableExperiment[MuxFluxResult, TwotoneFluxLocalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC = mux_flux_spec(TwotoneFluxLocalCfg, tag="mux/twotone_flux_local")

    def run(self, config: TwotoneFluxLocalCfg, *, context: RunContext) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Stop discards only the unfinished flux point; driver errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        return run_flux_scan(config, context=context)
