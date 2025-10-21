from typing import Any, Dict



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


__all__ = ["MyProgram", "AbsProxy", "ProxyAcquireMixin"]
