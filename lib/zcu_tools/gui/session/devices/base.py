"""Device construction and panel pairing shared by service and dialog."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from zcu_tools.device.base import BaseDeviceInfo

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from zcu_tools.gui.session.parameters import DocumentSetCapture
    from zcu_tools.gui.session.ports import DeviceProtocol


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
        parameter_provider: Callable[[], DocumentSetCapture | None],
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
