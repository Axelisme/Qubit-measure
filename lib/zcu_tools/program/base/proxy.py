import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from zcu_tools.config import config

from qick.qick_asm import AcquireMixin


class AbsProxy(ABC):
    @abstractmethod
    def test_remote_callback(self) -> bool:
        pass

    @abstractmethod
    def get_raw(self) -> Optional[list]:
        pass

    @abstractmethod
    def get_shots(self) -> Optional[list]:
        pass

    @abstractmethod
    def get_round_data(self) -> Tuple[Optional[list], Optional[list]]:
        """Return rounds_buf and stderr_buf"""
        pass

    @abstractmethod
    def set_early_stop(self) -> None:
        pass

    @abstractmethod
    def acquire(self, prog: AcquireMixin, **kwargs) -> list:
        pass

    @abstractmethod
    def acquire_decimated(self, prog: AcquireMixin, **kwargs) -> list:
        pass


class ProxyAcquireMixin(AcquireMixin):
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
        self.proxy_shots_expired = False
        self.proxy_round_expired = False

    def set_early_stop(self) -> None:
        if self.is_use_proxy():
            # tell proxy to set early stop
            self.proxy.set_early_stop()
        super().set_early_stop()  # set locally for safe

    def _set_expired(self) -> None:
        self.proxy_buf_expired = True
        self.proxy_shots_expired = True
        self.proxy_round_expired = True

    def get_raw(self) -> Optional[list]:
        if self.is_use_proxy():
            if self.proxy_buf_expired:  # check cache expiration
                self.acc_buf = self.proxy.get_raw()
                self.proxy_buf_expired = False
        return super().get_raw()

    def get_shots(self) -> Optional[list]:
        if self.is_use_proxy() and not config.ONLY_PROXY_DECIMATED:
            if self.proxy_shots_expired:  # check cache expiration
                self.shots = self.proxy.get_shots()
                self.proxy_shots_expired = False
        return super().get_shots()

    def get_stderr(self) -> Optional[list]:
        if self.is_use_proxy() and not config.ONLY_PROXY_DECIMATED:
            if self.proxy_round_expired:  # check cache expiration
                self.rounds_buf, self.stderr_buf = self.proxy.get_round_data()
                self.proxy_round_expired = False

        return super().get_stderr()

    def local_acquire(self, soc, **kwargs) -> list:
        # non-override method, for ProgramServer to call
        return super().acquire(soc, **kwargs)

    def local_acquire_decimated(self, soc, **kwargs) -> list:
        # non-override method, for ProgramServer to call
        return super().acquire_decimated(soc, **kwargs)

    def acquire(self, soc, **kwargs) -> list:
        if self.is_use_proxy() and not config.ONLY_PROXY_DECIMATED:
            self._set_expired()
            return self.proxy.acquire(self, **kwargs)

        return self.local_acquire(soc, **kwargs)

    def acquire_decimated(self, soc, **kwargs) -> list:
        if self.is_use_proxy():
            self._set_expired()
            return self.proxy.acquire_decimated(self, **kwargs)

        return self.local_acquire_decimated(soc, **kwargs)
