"""User-facing T1 choices; the workflow constructs each acquisition cfg."""

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, JsonValue
from zcu_tools.datafile.native_models import RunSnapshot
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormModel,
    FormPath,
    ModuleInput,
    RangeInput,
)

from zcu_lab.workflows.t1 import T1Plan

from .calibration_modules import (
    calibration_choices,
    calibration_defaults,
    expand_fluxes,
    resolve_calibration_modules,
)


class T1PlanForm(FormModel):
    """Start-time setpoints and two shared module reference/inline selections.

    flux_dev is the resolved host device name; its input defaults to $flux_dev
    in the fixed alias capture. Required flux_unit declares A, V or native.
    fluxes contain absolute values in that coordinate; no conversion occurs.
    points >= 5 controls runtime T1 delay resolution. qub_pulse selects the
    shared pulse drive; readout selects a pulse-backed readout because native
    readout landscape acquisition requires gain control. Both use ModuleInput
    wire envelopes and the fixed host library, never a live lookup. pi_length_us
    is the required initial Rabi-range estimate, not an accepted calibration;
    host seed composition must consume it separately from the native plan. Required
    selections stay unfilled until supplied. Runtime owns every cfg and grid.
    """

    flux_dev: str = Field(min_length=1, description="Host flux device name")
    flux_unit: Literal["A", "V", "native"] = Field(
        description="Explicit flux coordinate: YOKO A/V, Fake native"
    )
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
        """Prefill the device alias expression and captured x180 pi estimate.

        Keep $flux_dev as an expression for shared preview/Start evaluation.
        Missing captures or bindings produce an alias-naming field error, not
        a device fallback. Device membership is checked only when used.
        On initial input only, resolve $qubit.module.x180.waveform.length in
        the fixed alias capture and retain its concrete path as an expression
        if readable and non-null. No capture, absent/null values or
        unreadable paths leave the field unfilled for explicit user input. Other
        modules and simulator constants are never consulted. Invalid readable
        values produce ordinary field validation errors, not a fallback.
        No live selection, YAML refresh or device access occurs.
        """
        return calibration_defaults(super().defaults(env, previous), env, previous)

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer captured pulse/readout roots without materializing modules.

        Inline-only source-free drafts need no library. A used reference still
        requires the core ModuleRef source check; no live lookup or fallback.
        """
        return calibration_choices(env)

    def to_cfg(self, env: FormEnv, snapshot: RunSnapshot) -> T1Plan | FormErrors:
        """Resolve selections and expand setpoints into a static native plan.

        env supplies the captured library and validation/assets context; pure
        inline modules can omit the library. snapshot is fixed historical
        entry/point/roles supplied by the host. Return consuming-field errors
        for missing refs, wrong families or materialization failure. Convert
        through the native factory with the same captured context, so arb paths
        (including flat_top raises) stay bound to its asset directory. Native
        asset I/O failures propagate; no global scope or recipe interpretation.
        Pure arb drive returns a qub_pulse field error because Length-Rabi must
        vary its duration; flat_top arb ramps and arb readout remain supported.
        Keep descending/repeated/list order and a single range point. No acquisition
        cfg, tunables, timing/grid generation, capture, submission or extra copy
        layer is involved; Engine owns plan isolation.
        """
        modules = resolve_calibration_modules(
            qub_pulse=self.qub_pulse, readout=self.readout, env=env
        )
        if isinstance(modules, FormErrors):
            return modules
        return T1Plan(
            flux_dev=self.flux_dev,
            flux_unit=self.flux_unit,
            fluxes=expand_fluxes(self.fluxes),
            points=self.points,
            qub_pulse=modules.qub_pulse,
            readout=modules.readout,
            snapshot=snapshot,
        )
