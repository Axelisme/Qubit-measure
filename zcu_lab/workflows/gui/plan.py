"""User-facing T1 choices; the workflow constructs each acquisition cfg."""

from collections.abc import Mapping

import numpy as np
from pydantic import Field, JsonValue
from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    FlatTopWaveformCfg,
    ModuleCfgFactory,
    PulseReadoutSpec,
    PulseSpec,
    WaveformCfg,
)
from zcu_tools.cfg_model import (
    PulseCfg as MaterializedPulseCfg,
)
from zcu_tools.cfg_model import (
    PulseReadoutCfg as MaterializedPulseReadoutCfg,
)
from zcu_tools.datafile.native_models import RunSnapshot
from zcu_tools.gui.form import (
    ExplicitList,
    FormEnv,
    FormErrors,
    FormModel,
    FormPath,
    ModuleInput,
    ModuleRef,
    RangeInput,
)
from zcu_tools.gui.session.context_types import ContextSourceUnavailableError
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg
from zcu_tools.resources.entry import RoleResolutionError

from zcu_lab.workflows.t1 import T1Plan


def _reject_arb_assets(waveform: WaveformCfg) -> None:
    """Reject assets before native conversion loses the captured binding."""
    if isinstance(waveform, ArbWaveformCfg) or (
        isinstance(waveform, FlatTopWaveformCfg)
        and isinstance(waveform.raise_waveform, ArbWaveformCfg)
    ):
        raise ValueError("Arb assets are currently unsupported by T1 calibration")


class T1PlanForm(FormModel):
    """Start-time setpoints and two shared module reference/inline selections.

    flux_dev names the host device; fluxes contain absolute values in its units.
    points >= 5 controls runtime T1 delay resolution. qub_pulse selects the
    shared pulse drive; readout selects a pulse-backed readout because native
    readout landscape acquisition requires gain control. Both use ModuleInput
    wire envelopes and the fixed host library, never a live lookup. pi_length_us
    is the required initial Rabi-range estimate, not an accepted calibration;
    host seed composition must consume it separately from the native plan. Required
    selections stay unfilled until supplied. Runtime owns every cfg and grid.
    """

    flux_dev: str = Field(min_length=1, description="Host flux device name")
    fluxes: RangeInput = Field(description="Ordered absolute device setpoints")
    pi_length_us: float = Field(
        gt=0,
        description="Initial pi-duration estimate in us at the selected drive gain",
    )
    points: int = Field(default=51, ge=5, description="Uniform T1 delay samples")
    qub_pulse: ModuleInput = Field(description="Shared spectroscopy/Rabi/pi drive")
    readout: ModuleInput = Field(description="Shared pulse-backed readout")

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Prefill the required pi estimate from the fixed x180 module capture.

        On initial input only, retain qubit.module.x180.waveform.length as an
        expression if readable and non-null. No capture, absent/null values or
        unreadable paths leave the field unfilled for explicit user input. Other
        modules and simulator constants are never consulted. Invalid readable
        values produce ordinary field validation errors, not a fallback.
        No live selection, YAML refresh or device access occurs.
        """
        data = super().defaults(env, previous)
        if previous or env.parameters is None:
            return data
        path = "qubit.module.x180.waveform.length"
        try:
            value = env.parameters.lookup(path)
        except (
            AttributeError,
            ValueError,
            TypeError,
            RoleResolutionError,
            ContextSourceUnavailableError,
        ):
            # An unreadable optional default stays required, per the Start contract.
            return data
        if value.present and value.value is not None:
            data["pi_length_us"] = {"expr": path}
        return data

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer captured pulse/readout roots without materializing modules.

        Inline-only source-free drafts need no library. A used reference still
        requires the core ModuleRef source check; no live lookup or fallback.
        """
        if env.library is None:
            return {}
        return {
            ("qub_pulse", "name"): tuple(
                key
                for key in env.library.module_keys()
                if isinstance(env.library.module_spec(key), PulseSpec)
            ),
            ("readout", "name"): tuple(
                key
                for key in env.library.module_keys()
                if isinstance(env.library.module_spec(key), PulseReadoutSpec)
            ),
        }

    def to_cfg(self, env: FormEnv, snapshot: RunSnapshot) -> T1Plan | FormErrors:
        """Resolve selections and expand setpoints into a static native plan.

        env supplies the captured library and validation/assets context; pure
        inline modules can omit the library. snapshot is fixed historical
        entry/point/roles supplied by the host. Return consuming-field errors
        for missing refs, wrong families or materialization failure. Arb assets
        are currently unsupported, including flat_top arb ramps; return the
        consuming module field error rather than using global asset scope. Keep
        descending/repeated/list order and a single range point. No acquisition
        cfg, tunables, timing/grid generation, capture, submission or extra copy
        layer is involved; Engine owns plan isolation.
        """
        context = env.library.validation_context() if env.library is not None else None
        errors: dict[FormPath, str] = {}
        qub_pulse = None
        readout = None
        try:
            drive = ModuleCfgFactory.from_raw(
                self.qub_pulse.name
                if isinstance(self.qub_pulse, ModuleRef)
                else self.qub_pulse.module,
                context=context,
            )
            if isinstance(drive, MaterializedPulseCfg):
                _reject_arb_assets(drive.waveform)
            qub_pulse = PulseCfg.model_validate(drive.model_dump(mode="python"))
        except ValueError as exc:
            errors[("qub_pulse",)] = str(exc)
        try:
            ro = ModuleCfgFactory.from_raw(
                self.readout.name
                if isinstance(self.readout, ModuleRef)
                else self.readout.module,
                context=context,
            )
            if isinstance(ro, MaterializedPulseReadoutCfg):
                _reject_arb_assets(ro.pulse_cfg.waveform)
            readout = PulseReadoutCfg.model_validate(ro.model_dump(mode="python"))
        except ValueError as exc:
            errors[("readout",)] = str(exc)
        if qub_pulse is None or readout is None:
            return FormErrors(errors)
        fluxes = (
            self.fluxes.values
            if isinstance(self.fluxes, ExplicitList)
            else tuple(
                float(value)
                for value in np.linspace(
                    self.fluxes.start, self.fluxes.stop, self.fluxes.n
                )
            )
        )
        return T1Plan(
            flux_dev=self.flux_dev,
            fluxes=fluxes,
            points=self.points,
            qub_pulse=qub_pulse,
            readout=readout,
            snapshot=snapshot,
        )
