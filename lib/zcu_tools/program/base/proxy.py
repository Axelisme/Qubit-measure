import warnings
from abc import ABC, abstractmethod
from typing import Optional

from myqick.qick_asm import AcquireMixin


class AbsProxy(ABC):
    @abstractmethod
    def test_remote_callback(self) -> bool:
        pass

    @abstractmethod
    def get_acc_buf(self, prog) -> list:
        pass

    @abstractmethod
    def set_early_stop(self) -> None:
        pass

    @abstractmethod
    def acquire(self, prog, decimated, **kwargs) -> list:
        pass


class ProxyProgram(AcquireMixin):
    """
    Provide proxy support for acquire and acquire_decimated.
    """

    proxy: Optional[AbsProxy] = None

    @classmethod
    def init_proxy(cls, proxy: AbsProxy, test=False):
        if test:
            success = proxy.test_remote_callback()
            if not success:
                warnings.warn(
                    "Callback test failed, remote callback may not work, you should check your LOCAL_IP or LOCAL_PORT, it may be blocked by firewall"
                )

        cls.proxy = proxy

    @classmethod
    def clear_proxy(cls):
        cls.proxy = None

    @classmethod
    def is_use_proxy(cls):
        return cls.proxy is not None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.proxy_buf_expired = False

    def set_early_stop(self) -> None:
        if self.is_use_proxy():
            # tell proxy to set early stop
            self.proxy.set_early_stop()
        super().set_early_stop()  # set locally for safe

    def __getattribute__(self, name):
        # intercept acc_buf to fetch from proxy
        if name == "acc_buf":
            if self.is_use_proxy():
                # fetch acc_buf from proxy
                if self.proxy_buf_expired:
                    self.acc_buf = self.proxy.get_acc_buf(self)
                    self.proxy_buf_expired = False

        return object.__getattribute__(self, name)

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
        if self.is_use_proxy():
            return self.proxy_acquire(soc, decimated=False, **kwargs)

        return self.local_acquire(soc, decimated=False, **kwargs)

    def acquire_decimated(self, soc, **kwargs) -> list:
        if self.is_use_proxy():
            return self.proxy_acquire(soc, decimated=True, **kwargs)

        return self.local_acquire(soc, decimated=True, **kwargs)
