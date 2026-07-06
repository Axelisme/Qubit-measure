"""Run-time mutation guards for shared session-control facets."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zcu_tools.device.base import BaseDeviceInfo
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.result_scope import ResultScope
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.events import (
        DeviceChangedPayload,
        DeviceSetupFinishedPayload,
        DeviceSetupStartedPayload,
        PredictorChangedPayload,
    )
    from zcu_tools.gui.session.pbar_host import ProgressBarModel
    from zcu_tools.gui.session.predictor_control import PredictorControlPort
    from zcu_tools.gui.session.services.connection import ConnectRequest
    from zcu_tools.gui.session.services.device import (
        ActiveDeviceOperation,
        ConnectDeviceRequest,
        DeviceEntry,
        DeviceSnapshot,
        DisconnectDeviceRequest,
        SetupDeviceRequest,
    )
    from zcu_tools.gui.session.services.predictor import (
        LoadPredictorRequest,
        PredictCurveRequest,
        PredictCurveResult,
        PredictFreqRequest,
        PredictMatrixCurveRequest,
        PredictMatrixCurveResult,
        SetModelParamsRequest,
    )
    from zcu_tools.gui.session.services.startup import (
        PersistedStartup,
        StartupConnectionRequest,
        StartupProjectRequest,
    )
    from zcu_tools.gui.session.setup_control import SetupControlPort
    from zcu_tools.gui.session.types import SocCfgHandle
    from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


GuardFn = Callable[[str], None]


class GuardedSetupControl:
    """SetupControlPort wrapper that rejects setup mutations during a run."""

    def __init__(self, inner: SetupControlPort, guard: GuardFn) -> None:
        self._inner = inner
        self._guard = guard

    def get_bus(self) -> BaseEventBus:
        return self._inner.get_bus()

    def get_persisted_startup(self) -> PersistedStartup:
        return self._inner.get_persisted_startup()

    def list_result_scopes(self) -> tuple[ResultScope, ...]:
        return self._inner.list_result_scopes()

    def apply_startup_project(self, req: StartupProjectRequest) -> bool:
        self._guard("setup")
        return self._inner.apply_startup_project(req)

    def use_context(self, label: str) -> None:
        self._guard("setup")
        self._inner.use_context(label)

    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None:
        self._guard("setup")
        self._inner.new_context(bind_device=bind_device, clone_from=clone_from)

    def get_context_labels(self) -> list[str]:
        return self._inner.get_context_labels()

    def get_active_context_label(self) -> str | None:
        return self._inner.get_active_context_label()

    def start_connect(self, req: ConnectRequest) -> int:
        self._guard("setup")
        return self._inner.start_connect(req)

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        self._inner.bind_connection_outcome(on_finished, on_failed)

    def remember_startup_connection(self, req: StartupConnectionRequest) -> None:
        self._guard("setup")
        self._inner.remember_startup_connection(req)

    def get_soccfg(self) -> SocCfgHandle | None:
        return self._inner.get_soccfg()

    def list_devices(self) -> list[DeviceEntry]:
        return self._inner.list_devices()

    def get_device_unit(self, name: str) -> str:
        return self._inner.get_device_unit(name)


class GuardedContextControl:
    """ContextControlPort wrapper that keeps Inspect read-only during a run."""

    def __init__(self, inner: ContextControlPort, guard: GuardFn) -> None:
        self._inner = inner
        self._guard = guard

    def has_project(self) -> bool:
        return self._inner.has_project()

    def use_context(self, label: str) -> None:
        self._guard("context")
        self._inner.use_context(label)

    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None:
        self._guard("context")
        self._inner.new_context(bind_device=bind_device, clone_from=clone_from)

    def get_context_labels(self) -> list[str]:
        return self._inner.get_context_labels()

    def get_active_context_label(self) -> str | None:
        return self._inner.get_active_context_label()

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        return self._inner.list_value_sources()

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        return self._inner.read_value_source(key, type_name)

    def get_current_md(self) -> MetaDict:
        return self._inner.get_current_md()

    def get_current_ml(self) -> ModuleLibrary:
        return self._inner.get_current_ml()

    def coerce_md_value(self, key: str, text: str) -> Any:
        return self._inner.coerce_md_value(key, text)

    def set_md_attr(self, key: str, value: Any) -> None:
        self._guard("context")
        self._inner.set_md_attr(key, value)

    def del_md_attr(self, key: str) -> None:
        self._guard("context")
        self._inner.del_md_attr(key)

    def rename_ml_module(self, old: str, new: str) -> None:
        self._guard("context")
        self._inner.rename_ml_module(old, new)

    def rename_ml_waveform(self, old: str, new: str) -> None:
        self._guard("context")
        self._inner.rename_ml_waveform(old, new)

    def del_ml_module(self, name: str) -> None:
        self._guard("context")
        self._inner.del_ml_module(name)

    def del_ml_waveform(self, name: str) -> None:
        self._guard("context")
        self._inner.del_ml_waveform(name)


class GuardedDeviceControl:
    """DeviceControlPort wrapper that blocks device lifecycle changes mid-run."""

    def __init__(self, inner: DeviceControlPort, guard: GuardFn) -> None:
        self._inner = inner
        self._guard = guard

    def on_device_changed(
        self, handler: Callable[[DeviceChangedPayload], None]
    ) -> Callable[[], None]:
        return self._inner.on_device_changed(handler)

    def on_device_setup_started(
        self, handler: Callable[[DeviceSetupStartedPayload], None]
    ) -> Callable[[], None]:
        return self._inner.on_device_setup_started(handler)

    def on_device_setup_finished(
        self, handler: Callable[[DeviceSetupFinishedPayload], None]
    ) -> Callable[[], None]:
        return self._inner.on_device_setup_finished(handler)

    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
        self._guard("device")
        return self._inner.start_connect_device(req)

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int:
        self._guard("device")
        return self._inner.start_disconnect_device(req)

    def start_reconnect_device(self, name: str) -> int:
        self._guard("device")
        return self._inner.start_reconnect_device(name)

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        self._guard("device")
        return self._inner.start_setup_device(req)

    def forget_device(self, name: str) -> None:
        self._guard("device")
        self._inner.forget_device(name)

    def cancel_device_operation(self, name: str) -> None:
        self._guard("device")
        self._inner.cancel_device_operation(name)

    def list_devices(self) -> list[DeviceEntry]:
        return self._inner.list_devices()

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        return self._inner.get_device_snapshot(name)

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        return self._inner.get_device_info(name)

    def get_cached_device_value(self, name: str) -> float | None:
        return self._inner.get_cached_device_value(name)

    def poll_device_info(self, name: str) -> None:
        try:
            self._guard("device")
        except RuntimeError:
            return
        self._inner.poll_device_info(name)

    def is_memory_device(self, name: str) -> bool:
        return self._inner.is_memory_device(name)

    def get_device_unit(self, name: str) -> str:
        return self._inner.get_device_unit(name)

    def get_active_device_operations(self) -> tuple[ActiveDeviceOperation, ...]:
        return self._inner.get_active_device_operations()

    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]:
        return self._inner.attach_progress(owner_id, listener)

    def progress_bars(self, owner_id: str) -> tuple[tuple[int, ProgressBarModel], ...]:
        return self._inner.progress_bars(owner_id)


class GuardedPredictorControl:
    """PredictorControlPort wrapper that allows reads but blocks calibration edits."""

    def __init__(
        self,
        inner: PredictorControlPort,
        guard: GuardFn,
        *,
        on_mutated: Callable[[], None] | None = None,
    ) -> None:
        self._inner = inner
        self._guard = guard
        self._on_mutated = on_mutated

    def _notify_mutated(self) -> None:
        if self._on_mutated is not None:
            self._on_mutated()

    def on_predictor_changed(
        self, handler: Callable[[PredictorChangedPayload], None]
    ) -> Callable[[], None]:
        return self._inner.on_predictor_changed(handler)

    def load_predictor(self, req: LoadPredictorRequest) -> None:
        self._guard("predictor")
        self._inner.load_predictor(req)
        self._notify_mutated()

    def set_predictor_model_params(self, req: SetModelParamsRequest) -> None:
        self._guard("predictor")
        self._inner.set_predictor_model_params(req)
        self._notify_mutated()

    def clear_predictor(self) -> None:
        self._guard("predictor")
        self._inner.clear_predictor()
        self._notify_mutated()

    def predict_freq(self, req: PredictFreqRequest) -> float:
        return self._inner.predict_freq(req)

    def predict_freq_curve(self, req: PredictCurveRequest) -> PredictCurveResult:
        return self._inner.predict_freq_curve(req)

    def predict_matrix_element_curve(
        self, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        return self._inner.predict_matrix_element_curve(req)

    def get_predictor_info(self) -> dict | None:
        return self._inner.get_predictor_info()
