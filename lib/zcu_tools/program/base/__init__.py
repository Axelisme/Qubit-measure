from typing import Any, Dict

from zcu_tools.auto import is_pulse_cfg
from zcu_tools.utils.async_func import AsyncFunc

from .improve_acquire import ImproveAcquireMixin
from .proxy import AbsProxy, ProxyAcquireMixin


class MyProgram(ProxyAcquireMixin, ImproveAcquireMixin):
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
        # set dac and adc config as attributes
        self.cfg = cfg
        self.dac: Dict[str, Any] = cfg.get("dac", {})
        self.adc: Dict[str, Any] = cfg.get("adc", {})
        if "sweep" in cfg:
            self.sweep_cfg = cfg["sweep"]

        # set dac pulse as attributes
        for name, pulse in self.dac.items():
            if not is_pulse_cfg(name, pulse):
                continue
            if hasattr(self, name):
                raise ValueError(f"Pulse name {name} already exists")
            setattr(self, name, pulse)

    def acquire(self, soc, **kwargs) -> list:
        # let callback be executd as a coroutine
        with AsyncFunc(kwargs.get("callback")) as cb:
            kwargs["callback"] = cb

            return super().acquire(soc, **kwargs)

    def acquire_decimated(self, soc, **kwargs) -> list:
        # let callback be executd as a coroutine
        with AsyncFunc(kwargs.get("callback")) as cb:
            kwargs["callback"] = cb

            return super().acquire_decimated(soc, **kwargs)


__all__ = ["MyProgram", "AbsProxy", "ProxyAcquireMixin"]
