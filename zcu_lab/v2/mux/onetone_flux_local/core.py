"""Independent local line DC scan with mux onetone spectra."""

from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.context import RunContext

from zcu_lab.v2._support.mux_flux import (
    LocalFluxCfg,
    MuxFluxResult,
    mux_flux_spec,
    run_flux_scan,
)
from zcu_lab.v2.mux.onetone.core import MuxOnetoneSweepCfg


class OnetoneFluxLocalCfg(LocalFluxCfg[MuxOnetoneSweepCfg]):
    """Full mux onetone settings and local native-unit DC scan.

    Inherited tones retain original readout identity; participants determines
    acquisition order. See LocalFluxCfg for range, unit and settle contracts.
    """


class OnetoneFluxLocalExp(PersistableExperiment[MuxFluxResult, OnetoneFluxLocalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC = mux_flux_spec(OnetoneFluxLocalCfg, tag="mux/onetone_flux_local")

    def run(self, config: OnetoneFluxLocalCfg, *, context: RunContext) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Stop discards only the unfinished flux point; driver errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        return run_flux_scan(config, context=context)
