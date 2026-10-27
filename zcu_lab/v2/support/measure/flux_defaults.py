"""Flux starting coordinates from fixed Form captures, without device I/O."""

from dataclasses import dataclass

from pydantic import JsonValue
from zcu_tools.analysis.flux import DeviceFluxPoint
from zcu_tools.cfg_model import validate_device_name
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.form import FormEnv

from zcu_lab.components.qubit import QubitBaseSchema
from zcu_lab.v2.support.flux import device_flux_points, qubit_flux_frame


@dataclass(frozen=True)
class FluxSeed:
    """Captured DC identity, native unit and actual coordinates for one qubit."""

    device: str | None
    """Selected device name, or None when the component has no source."""
    unit: str | None
    """Native device-info label, or None without a selected source."""
    points: DeviceFluxPoint | None
    """Actual half/integer, or None when this device has no calibration."""


def captured_flux_seed(env: FormEnv, qubit: str, device_name: str | None) -> FluxSeed:
    """Read a qubit frame and chosen device from an Open/Reset capture.

    qubit is a concrete component name or captured alias. device_name is its
    chosen local or global DC source, never a reserved coil alias. Missing
    source or selected calibration leaves required coordinates blank. Every
    frame device and selected source must have a valid shared name and belong
    to env.devices. Values and units come only from detached DC device infos.
    Non-qubits, unavailable DC infos, missing contributors or inconsistent
    frames raise ValueError naming the component/device. No catalog or live
    driver is consulted; input objects are not changed.
    """
    component = captured_qubit(env, qubit)
    qubit_flux_frame(component.flux)
    for name in component.flux:
        if name not in env.devices:
            raise ValueError(f"Frame device {name!r} is not in the Form device options")
    if device_name is None:
        return FluxSeed(None, None, None)
    validate_device_name(device_name)
    if device_name not in env.devices:
        raise ValueError(f"Device {device_name!r} is not in the Form device options")
    info = env.devices[device_name]
    if not isinstance(info, (FakeDeviceInfo, YOKOGS200Info)):
        raise ValueError(f"Device {device_name!r} has no captured DC value/unit")
    if device_name not in component.flux:
        return FluxSeed(device_name, info.value_unit, None)
    values: dict[str, float] = {}
    for name in component.flux:
        if name == device_name:
            continue
        contributor = env.devices[name]
        if not isinstance(contributor, (FakeDeviceInfo, YOKOGS200Info)):
            raise ValueError(f"Device {name!r} has no captured DC value")
        values[name] = contributor.value
    points = device_flux_points(component.flux, device_name, values)
    return FluxSeed(device_name, info.value_unit, points)


def captured_qubit(env: FormEnv, qubit: str) -> QubitBaseSchema:
    """Return one qubit from the fixed capture; absent/wrong kind raises ValueError.

    qubit is a concrete component name or captured scalar alias. Source and
    alias errors propagate. This performs no file, device or library I/O.
    """
    if env.parameters is None:
        raise ValueError("Flux forms require a parameter capture")
    name = env.parameters.resolve_path(qubit)
    component = env.parameters.document().components.get(name)
    if not isinstance(component, QubitBaseSchema):
        raise ValueError(f"{name!r}: flux defaults require a qubit component")
    return component


def flux_scan_seed(points: DeviceFluxPoint | None, margin: float) -> JsonValue:
    """Return a 101-point range input from integer to half, expanded by margin.

    margin is a fraction of their separation, normally 0 or 0.1. Missing points
    leave both endpoints blank. The scan retains the points' direction and uses
    native device units. No frame conversion or source access occurs here.
    """
    start, stop = flux_bounds(points, margin)
    return {"kind": "linrange", "start": start, "stop": stop, "n": 101}


def flux_bounds(
    points: DeviceFluxPoint | None, margin: float
) -> tuple[float | None, float | None]:
    """Expand actual integer-to-half coordinates by a separation fraction.

    margin is normally 0 or 0.1. Return native endpoints in the original
    direction, or two blanks without points. No source or frame is read.
    """
    if points is None:
        return None, None
    delta = margin * (points.integer - points.half)
    return points.integer + delta, points.half - delta


def qubit_flux_initial_values(
    env: FormEnv, *, margin: float = 0.1
) -> dict[str, JsonValue]:
    """Seed the selected $qubit DC device/unit and 101-point actual flux range.

    The default device remains $qubit.flux_dev. margin expands the range by
    the given fraction of half/integer separation. Missing calibration leaves
    required endpoints blank. Invalid names, nonmember devices, source faults
    and invalid frames propagate from captured_flux_seed. No live reads occur.
    """
    qubit = captured_qubit(env, "$qubit")
    seed = captured_flux_seed(env, "$qubit", qubit.flux_dev)
    return {
        "flux_dev": seed.device,
        "flux_unit": seed.unit,
        "flux": flux_scan_seed(seed.points, margin),
    }
