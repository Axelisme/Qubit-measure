from typing import Any, Dict

from zcu_tools.utils.async_func import AsyncFunc

from .improve_acquire import ImproveAcquireMixin


class MyProgram(ImproveAcquireMixin):
    """
    Add some functionality to the base program class
    including:
        parse config to dac/adc and pulse attributes
        wrap acqurie callback to be a coroutine
    """

    def __init__(self, soccfg, cfg: Dict[str, Any], **kwargs) -> None:
        self._parse_cfg(cfg)  # parse config first
        super().__init__(soccfg, cfg=cfg, **kwargs)

    def _parse_cfg(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def acquire(self, soc, **kwargs) -> list:
        # let callback be executd as a coroutine
        with AsyncFunc(kwargs.get("callback"), min_interval=0.1) as cb:
            kwargs["callback"] = cb

            return super().acquire(soc, **kwargs)

    def acquire_decimated(self, soc, **kwargs) -> list:
        # let callback be executd as a coroutine
        with AsyncFunc(kwargs.get("callback"), min_interval=0.1) as cb:
            kwargs["callback"] = cb

            return super().acquire_decimated(soc, **kwargs)


__all__ = ["MyProgram", "AbsProxy", "ProxyAcquireMixin"]
