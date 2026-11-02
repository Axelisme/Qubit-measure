"""Global coil DC scan with mux twotone spectra."""

from typing import Annotated, ClassVar, Literal

import numpy as np
from pydantic import Field, StrictInt, model_validator
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.program.v2 import ResetCfg, SweepCfg
from zcu_tools.program.v2.modules import MuxReadoutCfg

from zcu_lab.v2.mux.twotone_spectrum.core import MuxFreqCfg, MuxFreqToneCfg
from zcu_lab.v2.support.mux_flux import (
    MuxFluxResult,
    mux_flux_spec,
    run_flux_scan,
)
from zcu_lab.v2.support.mux_sweep_cfg import require_tone_value, validate_mux_scan


class TwotoneFluxGlobalCfg(ConfigBase):
    """Simultaneous qubit spectra at each shared coil coordinate.

    reset is optional; readout and tones retain every original group position.
    Each tone supplies its fixed name, drive and absolute MHz frequency bounds.
    flux_dev names one connected DC driver; flux is its finite native-unit sweep.
    expts is the positive hardware frequency point count shared by all tones.
    participants selects unique original indices in acquisition/result order.
    flux_unit is A, V or native without conversion; settle is a finite
    nonnegative wait in seconds after each coil write. reps/rounds are positive
    averaging counts; initial_delay/relax_delay are finite nonnegative us.
    Selected tones need drive and frequency settings, with independent drive
    generators separate from readout. Invalid settings raise before any DC write.
    """

    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[MuxFreqToneCfg], FieldMeta(element_label="name")]
    flux_dev: DeviceName = Field(min_length=1)
    flux: SweepCfg
    expts: int = Field(
        strict=True,
        gt=0,
        title="Expts",
        description="Hardware frequency point count shared by all tones.",
    )
    participants: list[StrictInt]
    flux_unit: Literal["A", "V", "native"] = "A"
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
    def validate_scan(self) -> "TwotoneFluxGlobalCfg":
        """Reject invalid participants, active drives, timing or DC coordinates."""
        validate_mux_scan(self, self.tones, self.participants, self.expts)
        for index in self.participants:
            require_tone_value(self.tones[index].freq, ("tones", index, "freq"))
        if not all(
            np.isfinite(v) for v in (self.flux.start, self.flux.stop, self.flux.step)
        ):
            raise ValueError("flux coordinates must be finite")
        return self


class TwotoneFluxGlobalExp(Experiment[MuxFluxResult, TwotoneFluxGlobalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, TwotoneFluxGlobalCfg]] = (
        mux_flux_spec(
            TwotoneFluxGlobalCfg,
            tag="mux/twotone_flux_global",
            cfg_schema_version="5.0",
        )
    )

    def run(
        self, config: TwotoneFluxGlobalCfg, *, context: RunContext
    ) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        At each coil point, apply optional reset, simultaneous drive sweeps,
        then common mux readout. Return tone-major maps in absolute MHz and
        native DC coordinates. Cancelling an inner acquire leaves that point
        NaN and scans the next. Cancelling the Run scope stops the scan and
        retains completed points. Driver and acquisition errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        spectrum = MuxFreqCfg(
            reset=config.reset,
            readout=config.readout,
            tones=config.tones,
            expts=config.expts,
            participants=config.participants,
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
