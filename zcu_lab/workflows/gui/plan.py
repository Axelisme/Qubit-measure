"""User-facing T1 choices; the workflow constructs each acquisition cfg."""

from collections.abc import Mapping

from pydantic import Field, JsonValue
from zcu_tools.cfg_model import PulseReadoutSpec, PulseSpec
from zcu_tools.datafile.native_models import RunSnapshot
from zcu_tools.gui.form import (
    ExplicitList,
    FormEnv,
    FormErrors,
    FormModel,
    FormPath,
    ModuleInput,
    RangeInput,
)

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
        raise NotImplementedError("Implement the captured module conversion seam")
