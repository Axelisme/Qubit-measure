"""Global coil DC scan with mux onetone spectra."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import numpy as np
from pydantic import Field, StrictInt, model_validator
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.program.v2 import MuxReadout, Reset, SweepCfg

from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg
from zcu_lab.v2.support.mux_flux import MuxFluxResult, mux_flux_spec, run_flux_scan
from zcu_lab.v2.support.mux_sweep_cfg import validate_participants


class OnetoneFluxToneCfg(ConfigBase):
    """One original readout position's unique, nonempty qubit name."""

    name: Annotated[str, FieldMeta(readonly=True)] = Field(min_length=1)


class OnetoneFluxGlobalCfg(ConfigBase):
    """Acquire common-offset readout spectra along one shared DC sweep.

    reset optionally precedes the full mux readout. tones names every original
    readout position, including inactive tones. flux_dev names the connected
    coil; flux is its finite native-unit SweepCfg. df is a finite nominal MHz
    offset SweepCfg shared by all readout tones, not an absolute frequency.
    participants is a nonempty unique list of original tone indices in result
    row order. flux_unit is A, V or native, without conversion. settle is a
    finite nonnegative wait in seconds after each DC write. reps and rounds
    control inner averaging; initial_delay and relax_delay are in microseconds.
    Invalid identities, selections or nonfinite scans raise validation errors.
    """

    reset: Reset.Cfg | None = None
    readout: MuxReadout.Cfg
    tones: Annotated[list[OnetoneFluxToneCfg], FieldMeta(element_label="name")]

    flux_dev: DeviceName = Field(min_length=1)
    flux: SweepCfg
    df: Annotated[SweepCfg, FieldMeta(label="Frequency offset", unit="MHz")]
    participants: list[StrictInt]
    flux_unit: Literal["A", "V", "native"]
    settle: Annotated[float, FieldMeta(unit="s")] = Field(
        default=0.1, ge=0, allow_inf_nan=False
    )

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0

    @model_validator(mode="after")
    def validate_scan(self) -> OnetoneFluxGlobalCfg:
        """Reject mismatched identities, invalid indices and nonfinite scans."""
        names = [tone.name for tone in self.tones]
        if len(names) != len(self.readout.tones):
            raise ValueError(
                "tones must name every original readout position exactly once"
            )
        validate_participants(names, self.participants)
        for name, sweep in (("flux", self.flux), ("df", self.df)):
            if not all(np.isfinite(v) for v in (sweep.start, sweep.stop, sweep.step)):
                raise ValueError(f"{name} coordinates must be finite")
        return self


class OnetoneFluxGlobalExp(Experiment[MuxFluxResult, OnetoneFluxGlobalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, OnetoneFluxGlobalCfg]] = (
        mux_flux_spec(
            OnetoneFluxGlobalCfg,
            tag="mux/onetone_flux_global",
            cfg_schema_version="5.0",
        )
    )

    def run(
        self, config: OnetoneFluxGlobalCfg, *, context: RunContext
    ) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return original tone indices, float64 DC/frequency coordinates and
        complex128 IQ (participants, flux points, df points). Frequencies are
        each original readout frequency plus nominal df, in absolute MHz.
        Cancelling an inner acquire leaves that point NaN and scans the next.
        Cancelling the Run scope stops the scan and retains completed points.
        Missing devices raise KeyError; unit mismatch raises ValueError and
        unsupported drivers raise TypeError. Driver/acquisition errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        spectrum = MuxOnetoneCfg(
            reset=config.reset,
            readout=config.readout,
            df=config.df,
            participants=list(config.participants),
            reps=config.reps,
            rounds=config.rounds,
            initial_delay=config.initial_delay,
            relax_delay=config.relax_delay,
        )
        values = np.linspace(config.flux.start, config.flux.stop, config.flux.expts)
        flux_values = np.tile(values, (len(config.participants), 1))
        return run_flux_scan(
            spectrum,
            flux_values=flux_values,
            devices=[config.flux_dev],
            flux_unit=config.flux_unit,
            settle=config.settle,
            context=context,
        )
