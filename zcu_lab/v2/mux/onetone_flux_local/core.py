"""Independent local line DC scan with mux onetone spectra."""

from typing import Annotated, ClassVar, Literal, Self

import numpy as np
from pydantic import Field, StrictInt, model_validator
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.program.v2 import MuxReadout, Reset, SweepCfg

from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg
from zcu_lab.v2.support.mux_flux import MuxFluxResult, mux_flux_spec, run_flux_scan
from zcu_lab.v2.support.mux_sweep_cfg import (
    ToneRangeCfg,
    make_sweep,
    require_tone_value,
    validate_participants,
)


class OnetoneFluxLocalToneCfg(ConfigBase):
    """One original qubit's identity and native-unit local DC range.

    name identifies its full readout position. flux_dev names a connected source.
    flux may be None for inactive tones; selected tones require bounds and use
    the shared flux_expts count. Lines advance together, not as a Cartesian grid.
    """

    name: Annotated[str, FieldMeta(readonly=True)] = Field(min_length=1)
    flux_dev: DeviceName = Field(min_length=1)
    flux: ToneRangeCfg | None = None

    def require_flux(self) -> ToneRangeCfg:
        """Return this selected line's DC bounds; raise ValueError if absent."""
        if self.flux is None:
            raise ValueError(f"{self.name}: active tone requires DC sweep")
        return self.flux


class OnetoneFluxLocalCfg(ConfigBase):
    """Full mux readout and zipped independent local DC scans.

    reset optionally precedes readout. tones names every original readout
    position, including inactive tones. Selected flux_dev names must be distinct;
    flux_expts supplies their shared positive point count. df is the common
    nominal MHz offset added to each selected readout tone's base frequency.
    participants selects unique original indices in acquisition/result row order.
    flux_unit is the driver's native A/V unit or native, with no conversion.
    settle waits in seconds after all DC writes. reps and rounds count averaging;
    initial_delay and relax_delay are program delays in us.
    """

    reset: Reset.Cfg | None = None
    readout: MuxReadout.Cfg
    tones: Annotated[list[OnetoneFluxLocalToneCfg], FieldMeta(element_label="name")]

    flux_expts: Annotated[int, FieldMeta(label="Flux points")] = Field(
        default=101, strict=True, gt=0
    )
    df: Annotated[SweepCfg, FieldMeta(label="Frequency offset", unit="MHz")]
    participants: Annotated[
        list[StrictInt],
        FieldMeta(description="Original tone indices in acquisition/result row order."),
    ]
    flux_unit: Literal["A", "V", "native"] = "A"
    settle: Annotated[float, FieldMeta(unit="s")] = Field(
        default=0.1, ge=0, allow_inf_nan=False
    )

    reps: Annotated[int, FieldMeta(description="Number of repetitions per round.")] = 1
    rounds: Annotated[int, FieldMeta(description="Number of software rounds.")] = 1
    initial_delay: Annotated[
        float, FieldMeta(unit="us", description="Delay before the first repetition.")
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(unit="us", decimals=3, description="Delay after each repetition."),
    ] = 1.0

    @model_validator(mode="after")
    def validate_lines(self) -> Self:
        """Require full unique identities and distinct valid selected DC ranges.

        Missing active bounds, duplicate devices, invalid indices or ranges
        raise validation errors before acquisition. Inactive ranges stay optional.
        """
        names = [tone.name for tone in self.tones]
        if len(names) != len(self.readout.tones) or len(set(names)) != len(names):
            raise ValueError(
                "tones must name every original readout position exactly once"
            )
        validate_participants(names, self.participants)
        devices = [self.tones[index].flux_dev for index in self.participants]
        if len(set(devices)) != len(devices):
            raise ValueError("local flux lines must use distinct devices")
        for index in self.participants:
            tone = self.tones[index]
            require_tone_value(tone.flux, ("tones", index, "flux"))
            bounds = tone.require_flux()
            make_sweep(bounds.start, bounds.stop, self.flux_expts)
        return self


class OnetoneFluxLocalExp(Experiment[MuxFluxResult, OnetoneFluxLocalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, OnetoneFluxLocalCfg]] = (
        mux_flux_spec(
            OnetoneFluxLocalCfg,
            tag="mux/onetone_flux_local",
            cfg_schema_version="5.0",
        )
    )

    def run(self, config: OnetoneFluxLocalCfg, *, context: RunContext) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Cancelling an inner acquire leaves that point NaN and scans the next.
        Cancelling the Run scope stops the scan and retains completed points.
        Driver and acquisition errors propagate. Invalid inner spectrum settings
        raise before DC writes. Sources retain their last value; no output is enabled.
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
            spectrum,
            flux_values=flux_values,
            devices=devices,
            flux_unit=config.flux_unit,
            settle=config.settle,
            context=context,
        )
