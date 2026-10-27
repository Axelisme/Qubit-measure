"""Device construction and panel pairing shared by service and dialog."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from zcu_tools.device.base import BaseDeviceInfo

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from zcu_tools.gui.session.parameters import ParameterContext


@runtime_checkable
class DeviceProtocol(Protocol):
    """Structural contract a hardware driver must satisfy to be a GUI device.

    pyright only checks the *shape* (these methods exist with these signatures).
    The semantic invariants an implementer must also honour — and which a bug
    silently violates — are:

    - ``setup`` runs **off the main thread** (via ``BackgroundRunner``), never
      the Qt main thread. It MUST poll ``stop_event`` during any long operation:
      cancellation works by ``stop_event.set()``, and the worker reports
      "cancelled" only if it returns with the event set. A ``setup`` that ignores
      ``stop_event`` makes cancel a no-op (the worker blocks until natural
      completion). ``progress`` may drive a pbar via the ambient pbar factory.
    - ``get_info`` returns a fresh value snapshot. It is called on the worker
      right after ``setup``, by the synchronous idle read path, and by the
      off-main poll path that may refresh current values during setup/ramp. It
      must not mutate device state.
    - ``close`` releases the underlying resource and SHOULD be idempotent: it is
      called on disconnect, and on any rollback where a driver was constructed
      but never became the live registry entry.
    """

    def setup(
        self,
        cfg: Any,
        *,
        progress: bool = True,
        stop_event: threading.Event | None = None,
    ) -> None: ...
    def get_info(self) -> BaseDeviceInfo: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class DeviceCapabilities:
    """Admission and bias-query policy for one known instrument type.

    bias_source enables cached numeric bias and unit queries.
    user_connectable permits manual Add Device, not stored-row Reconnect.
    """

    bias_source: bool
    user_connectable: bool


@runtime_checkable
class DevicePanelProtocol(Protocol):
    """GUI-thread editing contract for a connected instrument's cached info."""

    def load(self, info: BaseDeviceInfo) -> None:
        """Hydrate from this instrument's Info; reject another Info type."""
        ...

    def read(self) -> Mapping[str, object]:
        """Return edited fields, retaining EvalRef values for Apply-time resolution."""
        ...

    def dispose(self) -> None:
        """Retire owned widget callbacks before terminal Qt child deletion.

        Successful return prevents further delivery. Repeated calls are safe;
        cleanup failures propagate and can be retried before deleting children.
        Hiding/reopening the panel is not terminal and must not call dispose.
        """
        ...

    def reset_eval_fields(self) -> None:
        """Reset expressions to direct inputs before switching devices."""
        ...

    def set_editing_enabled(self, *, enabled: bool) -> None:
        """Allow or prevent user edits without changing displayed values."""
        ...


class DevicePanelFactory(Protocol):
    """Create one GUI-thread panel with a fixed-context provider for expressions."""

    def __call__(
        self,
        parameter_provider: Callable[[], ParameterContext | None],
        parent: QWidget | None = None,
    ) -> QWidget:
        """Return a QWidget implementing DevicePanelProtocol, owned by parent."""
        ...


class BaseDeviceAdapter(ABC):
    """Pair an instrument driver, existing Qt panel and admission policy.

    driver_cls identifies the concrete driver; its class name is the Info type.
    panel_cls constructs its QWidget on the GUI thread, implementing the panel
    protocol. capabilities separates manual creation from cached bias queries.
    Adapters are stateless; callers own connected drivers and panel widgets.
    """

    driver_cls: type[DeviceProtocol]
    panel_cls: DevicePanelFactory
    capabilities: DeviceCapabilities

    @abstractmethod
    def connect(self, address: str) -> DeviceProtocol:
        """Open address off the GUI thread; caller owns close after success.

        Driver/backend errors propagate. This method does not grant manual
        admission; trusted stored-row reconnect uses it even when manual Add
        is disabled. No driver is constructed for an unknown instrument type.
        """
        ...

    def bias_unit(self, _info: BaseDeviceInfo | None, /) -> str | None:
        """Read a validated cached bias unit, or None for absent/non-bias Info.

        No hardware reads or state mutations occur. Bias adapters override this
        method to validate their own Info before exposing its unit.
        """
        return None
