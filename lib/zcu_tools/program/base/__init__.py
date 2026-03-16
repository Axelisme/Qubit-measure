from typing import Any, Dict

from typing_extensions import TypedDict

from .improve_acquire import AbsStatisticTracker, ImproveAcquireMixin


class SweepCfg(TypedDict, closed=True):
    start: float
    stop: float
    expts: int
    step: float


class MyProgram(ImproveAcquireMixin):
    """
    Add some functionality to the base program class
    including:
        parse config to dac/adc and pulse attributes
        wrap acqurie callback to be a coroutine
    """

    def __init__(self, soccfg, cfg: Dict[str, Any], **kwargs) -> None:
        self.cfg = cfg
        super().__init__(soccfg, cfg=cfg, **kwargs)


__all__ = ["MyProgram", "SweepCfg", "AbsStatisticTracker"]
