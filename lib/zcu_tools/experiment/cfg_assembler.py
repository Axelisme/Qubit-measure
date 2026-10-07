from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickParam

from zcu_tools.cfg_model import (
    ArbWaveformCfg,
    BathResetCfg,
    BathResetSpec,
    CfgInputValue,
    CfgValidationContext,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DirectReadoutCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    FlatTopWaveformSpec,
    GaussWaveformCfg,
    ModuleCfg,
    ModuleCfgFactory,
    ModuleSpec,
    NoneResetCfg,
    PulseCfg,
    PulseReadoutCfg,
    PulseReadoutSpec,
    PulseResetCfg,
    PulseResetSpec,
    PulseSpec,
    TwoPulseResetCfg,
    TwoPulseResetSpec,
    WaveformCfg,
    WaveformSpec,
)
from zcu_tools.device import DeviceInfo, DeviceManager
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import format_sweep1D, get_single_sweep_name
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.utils import deepupdate

if TYPE_CHECKING:
    from zcu_tools.resources.context import MetaDict
    from zcu_tools.resources.context.library import LibrarySnapshot

T_ExpCfg = TypeVar("T_ExpCfg", bound=ExpCfgModel)

type ExperimentScalar = bool | int | float | complex | str
type ExperimentCfgValue = (
    None
    | ExperimentScalar
    | QickParam
    | SweepCfg
    | ModuleSpec
    | WaveformSpec
    | ModuleCfg
    | WaveformCfg
    | NDArray[np.float64]
    | Sequence[ExperimentCfgValue]
    | Mapping[str, ExperimentCfgValue]
)
# Assembly adds device models and sweep arrays whose dtype the formatter infers.
type _AssemblyValue = (
    ExperimentCfgValue | DeviceInfo | NDArray[np.generic] | Mapping[str, _AssemblyValue]
)


class ExperimentCfgInput:
    """Isolated target experiment fields, before validation or materialization.

    Keys are cfg field names; module references are concrete library keys.
    Nested mappings become dicts, tuples remain tuples, and other sequences
    become lists. Runtime cfgs, float64 arrays and QickParam retain their types.
    None is an explicit null value. Construction performs no source I/O,
    expression evaluation or cfg validation.
    """

    def __init__(self, values: Mapping[str, ExperimentCfgValue]) -> None:
        """Copy fields recursively, retaining core-owned asset capabilities."""
        self._values = {key: _copy_value(value) for key, value in values.items()}

    def to_raw(self) -> dict[str, ExperimentCfgValue]:
        """Return a mutable, independent copy of the captured field tree."""
        return {key: _copy_value(value) for key, value in self._values.items()}


def _copy_value(value: ExperimentCfgValue) -> ExperimentCfgValue:
    if isinstance(value, Mapping):
        return {key: _copy_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_value(child) for child in value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_copy_value(child) for child in value]
    return deepcopy(value)


@dataclass(frozen=True)
class CfgEnv:
    """Caller-selected resources for Notebook cfg assembly.

    md is the borrowed parameter mapping; assembly never evaluates it.
    ml is a fixed library snapshot with its captured asset capability.
    device_manager supplies fresh device info on each make_cfg call.
    This container owns no resource lifetime and never refreshes ml.
    """

    md: MetaDict
    ml: LibrarySnapshot
    device_manager: DeviceManager


def assemble_experiment_cfg(
    raw_cfg: ExperimentCfgInput,
    cfg_model: type[T_ExpCfg],
    *,
    context: CfgValidationContext | None,
    device_snapshot: Mapping[str, DeviceInfo],
    overrides: ExperimentCfgInput | None = None,
) -> T_ExpCfg:
    """Materialize captured experiment fields and validate the target cfg.

    overrides force-merge target fields without changing either input.
    context supplies fixed module references and assets; None permits only
    dependency-free inline modules. device_snapshot maps registered names to
    observed settings, and dev field patches cannot add devices.
    Single-sweep normalization follows experiment.utils.

    No input is mutated, library refreshed or hardware configured.
    Malformed device/module tokens raise TypeError with their input path;
    unknown devices raise KeyError. Core reference/asset/schema errors
    propagate. Target validation raises ValueError retaining its cause.
    """
    values = raw_cfg.to_raw()
    if overrides is not None:
        deepupdate(values, overrides.to_raw(), behavior="force")

    devices = deepcopy(dict(device_snapshot))
    for name, patch in _device_patches(values).items():
        if name not in devices:
            raise KeyError(f"Device {name} not found in device snapshot.")
        if not isinstance(patch, Mapping):
            raise TypeError(f"Device patch for {name!r} must be a mapping.")
        devices[name] = devices[name].with_updates(**dict(patch))

    assembled: dict[str, _AssemblyValue] = {**values, "dev": devices}
    modules = values.get("modules")
    if modules is not None:
        if not isinstance(modules, Mapping):
            raise TypeError("Experiment config field 'modules' must be a mapping.")
        assembled["modules"] = {
            name: ModuleCfgFactory.from_raw(
                _module_input(value, f"modules.{name}"), context=context
            )
            for name, value in modules.items()
        }

    sweep = values.get("sweep")
    if sweep is not None and (name := get_single_sweep_name(cfg_model)) is not None:
        assembled["sweep"] = format_sweep1D(sweep, name)

    try:
        return cfg_model.model_validate(assembled)
    except ValueError as exc:
        raise ValueError(
            f"Error validating experiment config with {cfg_model.__name__}:\n"
            f"exp_cfg: {assembled}\n"
            f"error: {exc}"
        ) from exc


def make_cfg(
    raw_cfg: Mapping[str, ExperimentCfgValue],
    cfg_model: type[T_ExpCfg],
    env: CfgEnv,
    *,
    overrides: Mapping[str, ExperimentCfgValue] | None = None,
) -> T_ExpCfg:
    """Assemble mapping fields with env's fixed library and current device info.

    raw_cfg and overrides follow ExperimentCfgInput; cfg_model is the target
    experiment model. env supplies borrowed resources. Device reads and
    assembly errors propagate. Inputs and existing cfgs remain unchanged.
    This call does not refresh sources, evaluate md expressions or setup devices.
    """
    return assemble_experiment_cfg(
        ExperimentCfgInput(raw_cfg),
        cfg_model,
        context=env.ml.validation_context(),
        device_snapshot=env.device_manager.get_all_info(),
        overrides=ExperimentCfgInput(overrides) if overrides is not None else None,
    )


def _device_patches(
    values: Mapping[str, ExperimentCfgValue],
) -> Mapping[str, ExperimentCfgValue]:
    patches = values.get("dev")
    if patches is None:
        return {}
    if not isinstance(patches, Mapping):
        raise TypeError("Experiment config field 'dev' must be a mapping.")
    return patches


def _module_input(
    value: ExperimentCfgValue, path: str
) -> str | Mapping[str, CfgInputValue] | ModuleSpec | ModuleCfg:
    # The public aliases are Annotated unions, not runtime isinstance targets.
    if isinstance(
        value,
        (
            str,
            PulseSpec,
            PulseCfg,
            DirectReadoutCfg,
            PulseReadoutSpec,
            PulseReadoutCfg,
            NoneResetCfg,
            PulseResetSpec,
            PulseResetCfg,
            TwoPulseResetSpec,
            TwoPulseResetCfg,
            BathResetSpec,
            BathResetCfg,
        ),
    ):
        return value
    if isinstance(value, Mapping):
        return {
            key: _module_leaf(child, f"{path}.{key}") for key, child in value.items()
        }
    raise TypeError(f"Invalid module input at {path}: {type(value).__name__}")


def _module_leaf(value: ExperimentCfgValue, path: str) -> CfgInputValue:
    if isinstance(value, Mapping):
        return {
            key: _module_leaf(child, f"{path}.{key}") for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _module_leaf(child, f"{path}[{index}]") for index, child in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _module_leaf(child, f"{path}[{index}]") for index, child in enumerate(value)
        ]
    if value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            complex,
            str,
            QickParam,
            PulseSpec,
            PulseCfg,
            DirectReadoutCfg,
            PulseReadoutSpec,
            PulseReadoutCfg,
            NoneResetCfg,
            PulseResetSpec,
            PulseResetCfg,
            TwoPulseResetSpec,
            TwoPulseResetCfg,
            BathResetSpec,
            BathResetCfg,
            ConstWaveformCfg,
            CosineWaveformCfg,
            GaussWaveformCfg,
            DragWaveformCfg,
            ArbWaveformCfg,
            FlatTopWaveformSpec,
            FlatTopWaveformCfg,
        ),
    ):
        return value
    raise TypeError(f"Invalid module input at {path}: {type(value).__name__}")
