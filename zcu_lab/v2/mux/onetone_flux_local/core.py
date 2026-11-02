"""Independent local line DC scan with mux onetone spectra."""

from typing import ClassVar

import numpy as np
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
        mux_flux_spec(
            OnetoneFluxLocalCfg,
            tag="mux/onetone_flux_local",
            cfg_schema_version="4.0",
        )
    )

    def run(self, config: OnetoneFluxLocalCfg, *, context: RunContext) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Cancelling an inner acquire leaves that point NaN and scans the next.
        Cancelling the Run scope stops the scan and retains completed points.
        Driver and acquisition errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        flux_values = np.asarray(
            [
                np.linspace(
                    config.tones[i].require_flux().start,
                    config.tones[i].require_flux().stop,
                    config.flux_expts,
                )
                for i in config.participants
            ],
            dtype=np.float64,
        )
        devices = [config.tones[i].flux_dev for i in config.participants]
        return run_flux_scan(
            config.inner_cfg(),
            flux_values=flux_values,
            devices=devices,
            flux_unit=config.flux_unit,
            settle=config.settle,
            context=context,
        )
