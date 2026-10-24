"""Independent local line DC scan with mux onetone spectra."""

from typing import ClassVar

from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext

from zcu_lab.v2.mux.onetone.core import MuxFrequencyOffsetCfg
from zcu_lab.v2.support.mux_flux import (
    LocalFluxCfg,
    MuxFluxResult,
    mux_flux_spec,
    run_flux_scan,
)


class OnetoneFluxLocalCfg(LocalFluxCfg[MuxFrequencyOffsetCfg]):
    """Full mux onetone settings and local native-unit DC scan.

    Inherited tones retain original readout identity; participants determines
    acquisition order. See LocalFluxCfg for range, unit and settle contracts.
    """


class OnetoneFluxLocalExp(Experiment[MuxFluxResult, OnetoneFluxLocalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, OnetoneFluxLocalCfg]] = (
        mux_flux_spec(OnetoneFluxLocalCfg, tag="mux/onetone_flux_local")
    )

    def run(self, config: OnetoneFluxLocalCfg, *, context: RunContext) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Stop discards only the unfinished flux point; driver errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        return run_flux_scan(config, context=context)
