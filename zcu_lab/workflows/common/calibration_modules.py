"""Shared captured module and seed policy for native calibrated workflows."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from pydantic import JsonValue
from zcu_tools.cfg_model import ModuleCfgFactory, PulseReadoutSpec, PulseSpec
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormPath,
    LinRange,
    ModuleInput,
    ModuleRef,
)
from zcu_tools.gui.session.context_types import ContextSourceUnavailableError
from zcu_tools.program.v2.modules import ArbWaveformCfg, PulseCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules import ModuleCfgFactory as NativeModuleCfgFactory
from zcu_tools.resources.entry import RoleResolutionError

from zcu_lab.workflows.common.calibration import Calibrated


@dataclass(frozen=True)
class CalibrationModules:
    """Resolved shared native modules for T1, T2 and MIST calibration.

    qub_pulse supplies Length-Rabi's variable-duration drive and waveform/gain;
    pure arb drive is excluded. readout supplies the pulse-backed landscape
    routing and gain control. Both are detached products of the fixed capture.
    """

    qub_pulse: PulseCfg
    readout: PulseReadoutCfg


def resolve_calibration_modules(
    *, qub_pulse: ModuleInput, readout: ModuleInput, env: FormEnv
) -> CalibrationModules | FormErrors:
    """Resolve shared ref/inline drive and readout in one captured FormEnv.

    Library references and arb assets use env's fixed validation context. The
    drive must be a PulseCfg whose length is variable for Length-Rabi (not pure
    arb); flat_top arb ramps remain valid. Readout must be pulse-backed for gain
    scanning; arb readout remains valid. Resolution/materialization ValueErrors
    return qub_pulse/readout consuming-field FormErrors. Asset I/O errors
    propagate. No live source/device read, acquisition or default module occurs.
    """
    context = env.library.validation_context() if env.library is not None else None
    errors: dict[FormPath, str] = {}
    drive = None
    native_readout = None
    try:
        core = ModuleCfgFactory.from_raw(
            qub_pulse.name if isinstance(qub_pulse, ModuleRef) else qub_pulse.module,
            context=context,
        )
        native = NativeModuleCfgFactory.from_core(core, context=context)
        if not isinstance(native, PulseCfg):
            raise ValueError("Calibration drive selection must be a pulse")
        if isinstance(native.waveform, ArbWaveformCfg):
            raise ValueError(
                "Length-Rabi calibration requires variable duration; pure arb drive cannot change length and is unsupported"
            )
        drive = native
    except ValueError as exc:
        errors[("qub_pulse",)] = str(exc)
    try:
        core = ModuleCfgFactory.from_raw(
            readout.name if isinstance(readout, ModuleRef) else readout.module,
            context=context,
        )
        native_ro = NativeModuleCfgFactory.from_core(core, context=context)
        if not isinstance(native_ro, PulseReadoutCfg):
            raise ValueError(
                "Calibration readout selection must support pulse gain control"
            )
        native_readout = native_ro
    except ValueError as exc:
        errors[("readout",)] = str(exc)
    if drive is None or native_readout is None:
        return FormErrors(errors)
    return CalibrationModules(drive, native_readout)


def calibration_defaults(
    data: dict[str, JsonValue], env: FormEnv, previous: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """Extend a fresh form-default dict with fixed flux alias and pi estimate.

    data is caller-owned and changed in place. Always declare flux_dev as the
    $flux_dev expression default; default_input and branch defaults evaluate it
    into the captured device name, or leave the required field unfilled when
    the alias is missing or unbound. With no previous input, resolve captured
    $qubit x180 length and declare its concrete path as the pi_length_us
    expression default if present/non-null, evaluated the same way.
    Missing/unreadable optional defaults stay required, without simulator
    literals or live lookups.
    """
    data["flux_dev"] = {"expr": "$flux_dev"}
    if previous or env.parameters is None:
        return data
    try:
        path = env.parameters.resolve_path("$qubit.module.x180.waveform.length")
        value = env.parameters.lookup(path)
    except (
        AttributeError,
        ValueError,
        TypeError,
        RoleResolutionError,
        ContextSourceUnavailableError,
    ):
        return data
    if value.present and value.value is not None:
        data["pi_length_us"] = {"expr": path}
    return data


def calibration_choices(env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
    """Offer captured pulse/readout roots; no materialization or live I/O occurs.

    Missing library gives no suggestions; used refs still fail at resolution.
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


def expand_fluxes(fluxes: LinRange) -> tuple[float, ...]:
    """Expand a validated linear range using its exact start/stop/n count.

    Descending and repeated setpoints keep their order. No unit conversion,
    hardware lookup or sorting occurs.
    """
    return tuple(
        float(value) for value in np.linspace(fluxes.start, fluxes.stop, fluxes.n)
    )


def calibration_seed(
    qub_pulse: PulseCfg, readout: PulseReadoutCfg, pi_length_us: float
) -> Calibrated:
    """Build the original four-scalar seed from fixed modules and pi estimate.

    qub_pulse/readout supply positive MHz frequencies and readout native gain.
    pi_length_us is the validated initial Length-Rabi range estimate, not half
    a pi/2 fit. Non-scalar module frequency/gain or invalid calibration scalars
    raise ValueError. No fit, capture, live source or device read occurs.
    """
    frequency = qub_pulse.freq
    pulse = readout.pulse_cfg
    if (
        not isinstance(frequency, float)
        or not isinstance(pulse.freq, float)
        or not isinstance(pulse.gain, float)
    ):
        raise ValueError("Initial calibration requires scalar module frequency/gain")
    return Calibrated(frequency, pi_length_us, pulse.freq, pulse.gain)
