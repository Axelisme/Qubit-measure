"""Private JPA-family device lowering/preflight mechanics.

Only mechanics actually shared by JPA-family adapters live here; per-experiment
policy stays in the authoritative adapter file. No generic GUI support is
expanded from this package.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from zcu_tools.device import BaseDeviceInfo, DeviceInfo, GlobalDeviceManager

# Role-oriented GUI device keys and the labeled device patches they lower to.
# The core JPA experiments look the device up by these labels
# (``set_freq_in_dev_cfg(..., label="jpa_rf_dev")``, ``set_flux_in_dev_cfg(...,
# label="jpa_flux_dev")``), and the experiment assembler expects
# ``dev = {selected_device_name: {label: "jpa_<role>_dev"}}``.
JPA_RF_ROLE_KEY = "jpa_rf_dev"
JPA_RF_LABEL = "jpa_rf_dev"
JPA_FLUX_ROLE_KEY = "jpa_flux_dev"
JPA_FLUX_LABEL = "jpa_flux_dev"


def supports_freq_knob(info: DeviceInfo) -> bool:
    """True when ``info`` implements the frequency knob (``set_freq``).

    A class-level fact: drivers that only expose another frequency API (e.g.
    ``set_frequency`` on the Anritsu) or no frequency knob at all (DC sources)
    keep the base raising ``set_freq``.
    """
    return type(info).set_freq is not BaseDeviceInfo.set_freq


def supports_flux_knob(info: DeviceInfo) -> bool:
    """True when ``info`` implements the flux knob (``set_flux``)."""
    return type(info).set_flux is not BaseDeviceInfo.set_flux


def supports_power_knob(info: DeviceInfo) -> bool:
    """True when ``info`` implements the power knob (``set_power``)."""
    return type(info).set_power is not BaseDeviceInfo.set_power


def supports_output_knob(info: DeviceInfo) -> bool:
    """True when ``info`` implements the output knob (``set_output``)."""
    return type(info).set_output is not BaseDeviceInfo.set_output


def cached_device_snapshot() -> dict[str, DeviceInfo]:
    """Registry-cached device snapshot; never queries or commands hardware.

    Builds each entry's info from its static ``info_model`` (address only), so
    preflight stays pure: knob support is a class-level fact, and labels are
    run-time patch artifacts that never persist in the registry. The run-time
    snapshot (``GlobalDeviceManager.get_all_info``) remains the assembler's job.
    """
    devices = GlobalDeviceManager.get_all_devices()
    snapshot: dict[str, DeviceInfo] = {}
    for name, device in devices.items():
        # info_model is a concrete BaseDeviceInfo subclass; the base constructor
        # only knows its abstract shape, so construct through the concrete class.
        info_cls = cast(Any, device.info_model)
        snapshot[name] = cast(DeviceInfo, info_cls(address=device.address))
    return snapshot


def _lower_jpa_dev(
    raw_cfg: Mapping[str, object],
    device_snapshot: Mapping[str, DeviceInfo],
    *,
    role_key: str,
    role_label: str,
    label: str,
    knob_name: str,
    supports_knob: Callable[[DeviceInfo], bool],
) -> dict[str, dict[str, str]]:
    """Lower one role-oriented JPA selector to the assembler's labeled patch.

    The GUI cfg carries the selection as ``dev.<role_key> = <device name>``; the
    experiment assembler expects ``dev = {<device name>: {label: <label>}}``.
    Fast-fails before any hardware work on:

    - a missing selection (no ``dev`` section, or an empty ``<role_key>`` value);
    - a selected device that is not in the (cached) device snapshot;
    - a selected device without the required knob.

    The lowering yields exactly one labeled device: labels are run-time cfg
    patches, not registry state, so no other device's metadata is inspected or
    inferred. Returns the assembler patch.
    """
    dev_section = raw_cfg.get("dev")
    if not isinstance(dev_section, Mapping):
        raise ValueError(
            f"missing JPA {role_label} selection: cfg has no 'dev' section"
        )
    selected = dev_section.get(role_key)
    if not isinstance(selected, str) or not selected:
        raise ValueError(
            f"missing JPA {role_label} selection: 'dev.{role_key}' is empty"
        )
    info = device_snapshot.get(selected)
    if info is None:
        raise ValueError(
            f"JPA {role_label} {selected!r} not found in the device snapshot"
        )
    if not supports_knob(info):
        raise ValueError(
            f"JPA {role_label} {selected!r} ({type(info).__name__}) does not "
            f"support the {knob_name} knob"
        )
    return {selected: {"label": label}}


def lower_jpa_rf_dev(
    raw_cfg: Mapping[str, object],
    device_snapshot: Mapping[str, DeviceInfo],
) -> dict[str, dict[str, str]]:
    """Lower the RF selector for the frequency knob (``jpa/freq``)."""
    return _lower_jpa_dev(
        raw_cfg,
        device_snapshot,
        role_key=JPA_RF_ROLE_KEY,
        role_label="RF device",
        label=JPA_RF_LABEL,
        knob_name="frequency",
        supports_knob=supports_freq_knob,
    )


def lower_jpa_rf_power_dev(
    raw_cfg: Mapping[str, object],
    device_snapshot: Mapping[str, DeviceInfo],
) -> dict[str, dict[str, str]]:
    """Lower the RF selector for the power knob (``jpa/power``)."""
    return _lower_jpa_dev(
        raw_cfg,
        device_snapshot,
        role_key=JPA_RF_ROLE_KEY,
        role_label="RF device",
        label=JPA_RF_LABEL,
        knob_name="power",
        supports_knob=supports_power_knob,
    )


def lower_jpa_rf_output_dev(
    raw_cfg: Mapping[str, object],
    device_snapshot: Mapping[str, DeviceInfo],
) -> dict[str, dict[str, str]]:
    """Lower the RF selector for the output knob (``jpa/check``)."""
    return _lower_jpa_dev(
        raw_cfg,
        device_snapshot,
        role_key=JPA_RF_ROLE_KEY,
        role_label="RF device",
        label=JPA_RF_LABEL,
        knob_name="output",
        supports_knob=supports_output_knob,
    )


def lower_jpa_flux_dev(
    raw_cfg: Mapping[str, object],
    device_snapshot: Mapping[str, DeviceInfo],
) -> dict[str, dict[str, str]]:
    """Lower the flux selector (``jpa/flux``)."""
    return _lower_jpa_dev(
        raw_cfg,
        device_snapshot,
        role_key=JPA_FLUX_ROLE_KEY,
        role_label="flux device",
        label=JPA_FLUX_LABEL,
        knob_name="flux",
        supports_knob=supports_flux_knob,
    )
