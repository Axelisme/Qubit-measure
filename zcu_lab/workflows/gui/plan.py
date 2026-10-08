"""User-facing T1 choices; the workflow constructs each acquisition cfg."""

from collections.abc import Mapping

import numpy as np
from pydantic import Field, JsonValue
from zcu_tools.cfg_model import ModuleCfgFactory, PulseReadoutSpec, PulseSpec
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
from zcu_tools.program.v2.modules import PulseCfg, PulseReadoutCfg

from zcu_lab.workflows.t1 import T1Plan


class T1PlanForm(FormModel):
    """Start-time setpoints and two shared module reference/inline selections.

    flux_dev names the host device; fluxes contain absolute values in its units.
    points >= 5 controls runtime T1 delay resolution. qub_pulse selects the
    shared pulse drive; readout selects a pulse-backed readout because native
    readout landscape acquisition requires gain control. Both use ModuleInput
    wire envelopes and the fixed host library, never a live lookup. Required
    selections stay unfilled until supplied. Runtime owns every cfg and grid.
    """

    flux_dev: str = Field(min_length=1, description="Host flux device name")
    fluxes: RangeInput = Field(description="Ordered absolute device setpoints")
    points: int = Field(default=51, ge=5, description="Uniform T1 delay samples")
    qub_pulse: ModuleInput = Field(description="Shared spectroscopy/Rabi/pi drive")
    readout: ModuleInput = Field(description="Shared pulse-backed readout")

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
        for missing refs, wrong families or materialization failure. Keep
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
