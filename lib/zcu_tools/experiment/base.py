"""Experiment persistence and interfaces during the explicit-result migration.

``PersistableExperiment`` maps an explicit Result through ``AXES_SPEC`` without
keeping operation state. ``AbsExperiment``, ``ExperimentProtocol`` and the result
bookkeeping decorators still serve experiments whose callers have not migrated.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Concatenate,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np

from zcu_tools.experiment.axes_spec import AxesSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord

__all__ = [
    "AbsExperiment",
    "PersistableExperiment",
    "ExperimentProtocol",
    "record_result",
    "retrieve_result",
]

T_Result = TypeVar("T_Result")
T_Config = TypeVar("T_Config", bound=ExpCfgModel)
T_Config_contra = TypeVar("T_Config_contra", bound=ExpCfgModel, contravariant=True)
P = ParamSpec("P")
R = TypeVar("R")


def record_result(
    fn: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    """Cache the method's returned Result on ``self.last_result`` (run/load).

    Preserves the wrapped method's exact signature via ``ParamSpec``.
    """

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> R:
        result = fn(self, *args, **kwargs)
        self.last_result = result
        return result

    return wrapper


def retrieve_result(
    fn: Callable[Concatenate[Any, P], R],
) -> Callable[Concatenate[Any, P], R]:
    """Fall the ``result`` argument back to ``self.last_result`` when omitted/None
    (analyze/save). The wrapped method still asserts non-None.

    ``result`` is located by name via the bound signature, so it works wherever
    it sits in the parameter list (1st in analyze, 2nd in save).
    """
    sig = signature(fn)

    @wraps(fn)
    def wrapper(self: Any, *args: P.args, **kwargs: P.kwargs) -> R:
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("result") is None:
            bound.arguments["result"] = self.last_result
        return fn(*bound.args, **bound.kwargs)

    return wrapper


@runtime_checkable
class ExperimentProtocol(Protocol[T_Result, T_Config_contra]):
    """Pre-migration structural contract for stateful experiment callers.

    Experiments may add methods (e.g. ``calc_confusion_matrix``).
    ``run``/``analyze`` keyword surfaces are per-experiment. Migrated cores use
    explicit Results instead of this protocol's cached-result contract.

    ``T_Config_contra`` is contravariant: it appears only in input (``cfg``)
    position, so an experiment over a wider cfg satisfies a protocol over a
    narrower one.
    """

    last_result: T_Result | None

    def run(
        self,
        soc: Any,
        soccfg: Any,
        cfg: T_Config_contra,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> T_Result: ...

    def analyze(
        self, result: T_Result | None = ..., /, *args: Any, **kwargs: Any
    ) -> Any: ...

    def save(
        self,
        filepath: str,
        result: T_Result | None = ...,
        comment: str | None = ...,
        tag: str | None = ...,
        **kwargs: Any,
    ) -> None: ...

    def load(self, filepath: str, **kwargs: Any) -> T_Result: ...


class AbsExperiment(Generic[T_Result, T_Config]):
    """Minimal base: just the ``last_result`` cache.

    Native persistence (``save``/``load`` via ``AXES_SPEC``) is OPT-IN — inherit
    ``PersistableExperiment`` instead to gain it. Un-migrated experiments keep
    their own incompatible ``save``/``load`` signatures off this minimal base.
    """

    def __init__(self) -> None:
        self.last_result: T_Result | None = None


class PersistableExperiment(Generic[T_Result, T_Config]):
    """Stateless canonical save/load via ``AXES_SPEC``.

    Callers choose the RunRecord and destination explicitly. Loading returns a
    new RunRecord without changing this instance. Callers also own path reservation;
    an existing destination is rejected rather than overwritten.
    """

    #: per-experiment persistence declaration; required for save()/load().
    AXES_SPEC: ClassVar[AxesSpec[Any, Any] | None] = None

    def _spec(self) -> AxesSpec[T_Result, T_Config]:
        spec = type(self).AXES_SPEC
        if spec is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no AXES_SPEC; "
                "not migrated to native persistence"
            )
        return spec

    def _validate_canonical_labber_data(
        self, data: Any, spec: AxesSpec[T_Result, T_Config]
    ) -> None:
        if len(data.axes) != len(spec.axes):
            raise ValueError(
                f"{type(self).__name__} canonical data has {len(data.axes)} axes; "
                f"expected {len(spec.axes)}"
            )

        axis_lengths: list[int] = []
        for index, (loaded_axis, expected_axis) in enumerate(
            zip(data.axes, spec.axes, strict=True)
        ):
            if loaded_axis.name != expected_axis.label:
                raise ValueError(
                    f"{type(self).__name__} canonical axis {index} label is "
                    f"{loaded_axis.name!r}; expected {expected_axis.label!r}"
                )
            if loaded_axis.unit != expected_axis.unit:
                raise ValueError(
                    f"{type(self).__name__} canonical axis {index} unit is "
                    f"{loaded_axis.unit!r}; expected {expected_axis.unit!r}"
                )
            axis_values = np.asarray(loaded_axis.values)
            if axis_values.ndim != 1:
                raise ValueError(
                    f"{type(self).__name__} canonical axis {index} is "
                    f"{axis_values.ndim}D; expected 1D"
                )
            axis_lengths.append(axis_values.shape[0])

        if data.data.name != spec.z.label:
            raise ValueError(
                f"{type(self).__name__} canonical z channel label is "
                f"{data.data.name!r}; expected {spec.z.label!r}"
            )
        if data.data.unit != spec.z.unit:
            raise ValueError(
                f"{type(self).__name__} canonical z channel unit is "
                f"{data.data.unit!r}; expected {spec.z.unit!r}"
            )

        z_shape = np.asarray(data.z).shape
        expected_shape = tuple(reversed(axis_lengths))
        if z_shape != expected_shape:
            raise ValueError(
                f"{type(self).__name__} canonical z shape {z_shape} != "
                f"expected {expected_shape}"
            )

    def save(
        self,
        source: RunRecord[T_Config, T_Result],
        destination: Path,
        *,
        comment: str | None = None,
        tag: str | None = None,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> None:
        from zcu_tools.datafile import (
            save_labber_data,
            upload_to_server,
        )
        from zcu_tools.experiment.utils import make_comment

        spec = self._spec()
        result = source.result

        cfg = source.cfg
        if cfg is None:
            raise ValueError("RunRecord.cfg is None; cannot save without configuration")
        comment = make_comment(cfg, comment)

        axes = [
            (ax.label, ax.unit, np.asarray(getattr(result, ax.field_name)) * ax.scale)
            for ax in spec.axes
        ]
        z = (spec.z.label, spec.z.unit, np.asarray(getattr(result, spec.z.field_name)))

        saved_path = save_labber_data(
            str(destination), z=z, axes=axes, comment=comment, tags=tag or spec.tag
        )
        if server_ip is not None:
            upload_to_server(saved_path, server_ip, port)
            os.remove(saved_path)

    def load(
        self,
        source: Path,
        *,
        server_ip: str | None = None,
        port: int = 4999,
    ) -> RunRecord[T_Config, T_Result]:
        from zcu_tools.datafile import download_from_server, load_labber_data
        from zcu_tools.experiment.utils import parse_comment

        spec = self._spec()

        filepath = str(source)
        if server_ip is not None and not source.exists():
            download_from_server(filepath, server_ip, port)

        ld = load_labber_data(filepath)
        self._validate_canonical_labber_data(ld, spec)

        cfg_snapshot = None
        if ld.comment:
            cfg_dict, _, _ = parse_comment(ld.comment)
            if cfg_dict is not None:
                cfg_snapshot = spec.cfg_type.validate_or_warn(cfg_dict, source=filepath)

        kwargs: dict[str, Any] = {
            ax.field_name: (
                np.asarray(ld.axes[i].values, dtype=np.float64) / ax.scale
            ).astype(ax.dtype)
            for i, ax in enumerate(spec.axes)
        }
        kwargs[spec.z.field_name] = self._cast_loaded_z(ld.z, spec)
        return RunRecord(cfg=cfg_snapshot, result=spec.result_type(**kwargs))

    def _cast_loaded_z(
        self,
        loaded_z: Any,
        spec: AxesSpec[T_Result, T_Config],
    ) -> np.ndarray:
        target_dtype = np.dtype(spec.z.dtype)
        z_values = np.asarray(loaded_z)
        if target_dtype.kind != "c" and np.iscomplexobj(z_values):
            if np.any(np.imag(z_values) != 0.0):
                raise ValueError(
                    f"{type(self).__name__} canonical z channel "
                    f"{spec.z.label!r} contains non-zero imaginary component; "
                    f"cannot load as {target_dtype}"
                )
            z_values = np.real(z_values)
        return z_values.astype(target_dtype)
