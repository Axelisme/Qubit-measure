"""Global coil DC scan with mux onetone spectra."""

from zcu_tools.experiment import PersistableExperiment
from zcu_tools.experiment.context import RunContext

from zcu_lab.v2._support.mux_flux import (
    GlobalFluxCfg,
    MuxFluxResult,
    mux_flux_spec,
    run_flux_scan,
)
from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg


class OnetoneFluxGlobalCfg(GlobalFluxCfg[MuxOnetoneCfg]):
    """Full mux onetone settings and global native-unit DC scan.

    Inherited tones retain original readout identity; participants determines
    acquisition order. See GlobalFluxCfg for range, unit and settle contracts.
    """


class OnetoneFluxGlobalExp(PersistableExperiment[MuxFluxResult, OnetoneFluxGlobalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC = mux_flux_spec(OnetoneFluxGlobalCfg, tag="mux/onetone_flux_global")

    def run(
        self, config: OnetoneFluxGlobalCfg, *, context: RunContext
    ) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Stop discards only the unfinished flux point; driver errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        return run_flux_scan(config, context=context)
