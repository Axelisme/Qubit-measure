"""Independent local line DC scan with mux twotone spectra."""

from typing import Annotated, ClassVar, Literal

import numpy as np
from pydantic import Field, StrictInt, model_validator
from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.experiment import Experiment
from zcu_tools.experiment.axes_spec import GroupedAxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.utils.sweep import make_sweep
from zcu_tools.program.v2 import MuxReadoutCfg, ResetCfg

from zcu_lab.v2.mux.twotone_spectrum.core import MuxFreqCfg, MuxFreqToneCfg
from zcu_lab.v2.support.mux_flux import MuxFluxResult, mux_flux_spec, run_flux_scan
from zcu_lab.v2.support.mux_sweep_cfg import (
    ToneRangeCfg,
    require_tone_value,
    validate_mux_scan,
)


class TwotoneFluxLocalToneCfg(MuxFreqToneCfg):
    """One original qubit's drive, MHz range and local DC source/range.

    name, drive and freq retain the spectrum tone's meaning. flux_dev names a
    connected DC source even for inactive tones. flux gives finite native-unit
    bounds, required only when selected. All selected lines advance together
    using the parent flux_expts, not as a Cartesian product.
    """

    flux_dev: DeviceName = Field(min_length=1)
    flux: ToneRangeCfg | None = None

    def require_flux(self) -> ToneRangeCfg:
        """Return selected DC bounds, or raise ValueError naming the qubit."""
        if self.flux is None:
            raise ValueError(f"{self.name}: active tone requires DC sweep")
        return self.flux


class TwotoneFluxLocalCfg(ConfigBase):
    """Full original mux tones with zipped local DC and inner MHz scans.

    reset is optional; readout contains every original readout position. tones
    pairs each position with its named drive, absolute MHz bounds, source and
    native DC bounds. flux_expts is the positive shared outer count; expts is
    the positive shared hardware frequency count. participants selects unique
    original indices in acquisition/result order, without slicing either list.
    Selected tones require drive, freq and flux, with distinct DC sources.
    flux_unit is A, V or native without conversion. settle is a nonnegative
    finite wait in seconds after each point's DC writes. reps/rounds are
    positive averaging counts; initial_delay/relax_delay are finite nonnegative
    microseconds. Invalid identities, drives, timing or scan bounds raise.
    """

    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[TwotoneFluxLocalToneCfg], FieldMeta(element_label="name")]
    flux_expts: int = Field(default=101, strict=True, gt=0)
    expts: int = Field(
        strict=True, gt=0, description="Hardware frequency point count shared by all tones."
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
        FieldMeta(label="Initial delay", unit="us", description="Delay before the first repetition."),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay", unit="us", decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0

    @model_validator(mode="after")
    def validate_scan(self) -> TwotoneFluxLocalCfg:
        """Reject invalid selected drives, duplicate sources and inconsistent ranges."""
        validate_mux_scan(self, self.tones, self.participants, self.expts)
        devices = [self.tones[index].flux_dev for index in self.participants]
        if len(set(devices)) != len(devices):
            raise ValueError("local flux lines must use distinct devices")
        for index in self.participants:
            tone = self.tones[index]
            require_tone_value(tone.freq, ("tones", index, "freq"))
            require_tone_value(tone.flux, ("tones", index, "flux"))
            make_sweep(tone.require_flux().start, tone.require_flux().stop, self.flux_expts)
        return self


class TwotoneFluxLocalExp(Experiment[MuxFluxResult, TwotoneFluxLocalCfg]):
    """Set DC, settle, acquire spectra and preserve completed flux points."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[MuxFluxResult, TwotoneFluxLocalCfg]] = (
        mux_flux_spec(
            TwotoneFluxLocalCfg,
            tag="mux/twotone_flux_local",
            cfg_schema_version="5.0",
        )
    )

    def run(self, config: TwotoneFluxLocalCfg, *, context: RunContext) -> MuxFluxResult:
        """Run config without mutation using context's drivers, board and Stop.

        Return tone-major maps in absolute MHz and native DC coordinates.
        Cancelling an inner acquire leaves that point NaN and scans the next.
        Cancelling the Run scope stops the scan and retains completed points.
        Driver and acquisition errors propagate.
        Sources remain at the final written value. No output is enabled here.
        """
        spectrum = MuxFreqCfg(
            reset=config.reset,
            readout=config.readout,
            tones=list[MuxFreqToneCfg](config.tones),
            expts=config.expts,
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
