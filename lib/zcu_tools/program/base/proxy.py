import warnings
from abc import ABC, abstractmethod
from typing import Optional

from myqick.qick_asm import AcquireMixin


class AbsProxy(ABC):
    @abstractmethod
    def test_remote_callback(self) -> bool:
        pass

    @abstractmethod
    def get_acc_buf(self) -> Optional[list]:
        pass

    @abstractmethod
    def set_early_stop(self) -> None:
        pass

    @abstractmethod
    def acquire(self, prog: AcquireMixin, decimated: bool, **kwargs) -> list:
        pass


class ProxyProgram(AcquireMixin):
    """
    Provide proxy support for acquire and acquire_decimated.
    """

    proxy: Optional[AbsProxy] = None

    @classmethod
    def init_proxy(cls, proxy: AbsProxy, test=False) -> None:
        if test:
            success = proxy.test_remote_callback()
            if not success:
                warnings.warn(
                    "Callback test failed, remote callback may not work, you should check your LOCAL_IP or LOCAL_PORT, it may be blocked by firewall"
                )

        cls.proxy = proxy

    @classmethod
    def clear_proxy(cls) -> None:
        cls.proxy = None

    @classmethod
    def is_use_proxy(cls) -> bool:
        return cls.proxy is not None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.proxy_buf_expired = False

    def set_early_stop(self) -> None:
        if self.is_use_proxy():
            # tell proxy to set early stop
            self.proxy.set_early_stop()
        super().set_early_stop()  # set locally for safe

    def get_acc_buf(self) -> Optional[list]:
        if self.is_use_proxy():
            if self.proxy_buf_expired:
                self.acc_buf = self.proxy.get_acc_buf()
                self.proxy_buf_expired = False
        return self.acc_buf

    def local_acquire(self, soc, decimated: bool, **kwargs) -> list:
        # non-override method, for ProgramServer to call
        if decimated:
            return super().acquire_decimated(soc, **kwargs)
        return super().acquire(soc, **kwargs)

    def proxy_acquire(self, decimated: bool, **kwargs) -> list:
        # acquire from proxy, also reset local acc_buf
        self.proxy_buf_expired = True
        return self.proxy.acquire(self, decimated=decimated, **kwargs)

    def acquire(self, soc, **kwargs) -> list:
        return self.local_acquire(soc, decimated=False, **kwargs)

    def acquire_decimated(self, soc, **kwargs) -> list:
        if self.is_use_proxy():
            return self.proxy_acquire(decimated=True, **kwargs)

        return self.local_acquire(soc, decimated=True, **kwargs)
