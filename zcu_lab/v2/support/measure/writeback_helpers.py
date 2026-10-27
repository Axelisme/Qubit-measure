"""Shared pulse-readout proposal mechanics and existing DPM calibration policy.

Fluxdep proposals pair signed period and offset calibration with raw-fit Run records.
Reset finishers live in their adapters and materialize the Run-tested cfg
without reading calibration values from the capture.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Protocol

from zcu_tools.analysis.flux import DevicePeriod, offset_for_flux
from zcu_tools.cfg_model import ConfigBase, module_spec_from_raw
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    ModuleWriteback,
    ParameterWriteback,
    RecordWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
from zcu_tools.program.v2.modules import AbsModuleCfg, PulseReadoutCfg

from zcu_lab.components.qubit import QubitBaseSchema, QubitFluxCalibration
from zcu_lab.v2.support.flux import (
    record_flux_frame,
    snapshot_numeric_values,
)
from zcu_lab.v2.support.measure.ctx_helpers import optional_parameter_float

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv


class _HasReadout(Protocol):
    @property
    def readout(self) -> object: ...


_READOUT_DPM_KEYS = ("best_ro_freq", "best_ro_gain", "best_ro_length")
READOUT_DPM_PULSE_TAIL_US = 0.1


def _resolve_readout_dpm_values(
    ctx: SessionEnv, proposed: Mapping[str, float]
) -> tuple[float, float, float] | None:
    values: list[float] = []
    for key in _READOUT_DPM_KEYS:
        if key in proposed:
            values.append(float(proposed[key]))
            continue
        if ctx.parameters is None:
            return None
        calibrated = optional_parameter_float(ctx.parameters, f"$qubit.ext.{key}")
        if calibrated is None:
            return None
        values.append(calibrated)

    best_ro_freq, best_ro_gain, best_ro_length = values
    if not all(math.isfinite(value) for value in values):
        return None
    return best_ro_freq, best_ro_gain, best_ro_length


def module_writeback(
    module: AbsModuleCfg,
    *,
    target: str,
    description: str,
    field_updates: Sequence[tuple[str, float]] = (),
    role_id: str | None = None,
) -> ModuleWriteback:
    """Propose the persistent Spec of a Run-time module with fitted fields replaced.

    module is a materialized module from the Run cfg snapshot and is not
    modified; its child pulses and waveforms become inline Specs. Each
    (dotted path, value) in field_updates replaces that field, for example
    ("pulse_cfg.waveform.length", 1.2), before validation, so a swept source
    value can be replaced by its fitted scalar. A path through a missing
    section raises KeyError; a result that is not a valid persistent module
    Spec raises ValueError. target, description and role_id fill the item.
    """
    raw: dict[str, Any] = module.to_dict()
    for path, value in field_updates:
        *sections, leaf = path.split(".")
        node = raw
        for section in sections:
            node = node[section]
        node[leaf] = value
    return ModuleWriteback(
        target_name=target,
        description=description,
        proposal=module_spec_from_raw(raw),
        role_id=role_id,
    )


def pulse_readout_module_writeback_items(
    cfg_snapshot: _HasReadout | None,
    *,
    target: str,
    desc: str,
    field_updates: Sequence[tuple[str, float]],
    role_id: str | None = None,
) -> list[ModuleWriteback]:
    if cfg_snapshot is None:
        return []

    readout_cfg = cfg_snapshot.readout
    if not isinstance(readout_cfg, PulseReadoutCfg):
        return []

    return [
        module_writeback(
            readout_cfg,
            target=target,
            description=desc,
            field_updates=field_updates,
            role_id=role_id,
        )
    ]


def readout_dpm_writeback_items(
    ctx: SessionEnv,
    cfg_snapshot: _HasReadout | None,
    *,
    proposed: Mapping[str, float],
) -> list[WritebackItem]:
    """Propose the resonator's optimized pulse readout and select it for later forms.

    cfg_snapshot supplies the Run's readout settings. proposed supplies any of
    best_ro_freq (MHz), best_ro_gain (normalized), and best_ro_length (us);
    missing coordinates come from ctx's captured $qubit.ext calibration.
    Missing cfg, non-pulse readout, or missing/non-finite coordinates yields no
    items. A captured parameter source adds a slot proposal after the module,
    pointing $resonator.module.readout to the resolved optimized module root.
    Without a capture only the module is proposed. Source/alias errors propagate.
    """
    if cfg_snapshot is None:
        return []

    resolved = _resolve_readout_dpm_values(ctx, proposed)
    if resolved is None:
        return []

    best_ro_freq, best_ro_gain, best_ro_length = resolved
    items: list[WritebackItem] = []
    module_items = pulse_readout_module_writeback_items(
        cfg_snapshot,
        target="$resonator.readout.readout_dpm",
        desc="Optimized readout (DPM)",
        field_updates=(
            ("freq", best_ro_freq),
            ("gain", best_ro_gain),
            (
                "waveform.length",
                best_ro_length + READOUT_DPM_PULSE_TAIL_US,
            ),
            ("ro_length", best_ro_length),
        ),
        role_id="readout_dpm",
    )
    items.extend(module_items)
    if module_items and ctx.parameters is not None:
        items.append(
            ParameterWriteback(
                target_name="$resonator.module.readout",
                description="Select optimized readout for downstream experiments",
                proposed_value=ctx.parameters.resolve_path(
                    "$resonator.readout.readout_dpm"
                ),
            )
        )
    return items


def fluxdep_writeback_items[C: ConfigBase, R, A: FluxPickResult](
    req: WritebackRequest[RunRecord[C, R], A],
    *,
    device_name: str,
    qubit: str = "$qubit",
    element: ElementIdentity | None = None,
) -> list[WritebackItem]:
    """Propose a signed device period, qubit offset and raw fitted record.

    req carries the Analyze-time Run parameter capture and RunMetadata. qubit
    is a concrete name or captured alias; device_name is the scanned cfg device.
    element optionally labels the original mux tone/Q.
    Return no items without a parameter capture. Missing Run device snapshot,
    a non-qubit target, invalid calibration or present invalid contributor bias
    raises ValueError with target/device context. Missing device values contribute
    zero. Simultaneous scans do not prevent independent period calibration.
    Alias and source failures propagate.

    flux.period.<device_name> uses the fitted magnitude and stored direction,
    initially positive. flux.offset is selected by default and places the fitted
    half at flux 0.5, subtracting other calibrated devices' Run-start biases.
    Its note lists each present other bias, native unit and flux contribution,
    including zero; without other biases the note is None. The scanned device's
    Run-start value is replaced by the fitted half, never counted twice.
    The ordinary record stores raw native
    half/integer/positive period, device, the complete connected-device snapshot,
    and the original captured flux_frame, never the new calibration. Neither
    inputs nor live parameters/devices are read or changed at acceptance.
    """
    parameters = req.ctx.parameters
    if parameters is None:
        return []
    target = parameters.resolve_path(qubit)
    component = parameters.document().components.get(target)
    if not isinstance(component, QubitBaseSchema):
        raise ValueError(f"{target!r}: flux writeback requires a qubit component")
    metadata = req.run_result.metadata
    if metadata is None or metadata.device_snapshot is None:
        raise ValueError(f"{target!r}: flux writeback requires a Run device snapshot")
    devices = deepcopy(dict(metadata.device_snapshot))
    picked = req.analyze_result
    try:
        fitted = QubitFluxCalibration(
            half=picked.flx_half,
            integer=picked.flx_int,
            period=abs(picked.flx_period),
        )
        frame = record_flux_frame(component.flux, devices)
        previous_period = component.flux.period.get(device_name, 1.0)
        period = math.copysign(fitted.period, previous_period)
        periods = {
            name: DevicePeriod(value)
            for name, value in {**component.flux.period, device_name: period}.items()
        }
        values = snapshot_numeric_values(devices, periods)
        offset = offset_for_flux(
            periods, {**values, device_name: fitted.half}, flux=0.5
        )
        contributions = [
            f"{name} Run-start value = {value:.6g} "
            f"{devices[name].get('value_unit', 'native')} → "
            f"{value / periods[name].period:.6g} Φ0"
            for name, value in values.items()
            if name != device_name
        ]
    except ValueError as error:
        raise ValueError(f"{target!r}: {error}") from error
    return [
        ParameterWriteback(
            target_name=f"{target}.flux.period.{device_name}",
            description="Signed period (native DC units per flux quantum)",
            proposed_value=period,
            element=element,
        ),
        ParameterWriteback(
            target_name=f"{target}.flux.offset",
            description="Zero-output flux offset (flux quanta)",
            proposed_value=offset,
            selected=True,
            note="; ".join(contributions) or None,
            element=element,
        ),
        RecordWriteback(
            target_name=target,
            description="Raw flux calibration with Run frame and device snapshot",
            record={
                "device": device_name,
                "half": fitted.half,
                "integer": fitted.integer,
                "period": fitted.period,
                "devices": {name: deepcopy(info) for name, info in devices.items()},
                "flux_frame": frame,
                "run": {"run_id": metadata.run_id, "experiment": metadata.experiment},
            },
            element=element,
        ),
    ]
