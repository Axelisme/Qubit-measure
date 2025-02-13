from collections import defaultdict
from typing import Any, Dict

from qick.asm_v2 import AveragerProgramV2

from .readout import make_readout
from .reset import make_reset


class MyProgram(AveragerProgramV2):
    def __init__(self, soccfg, cfg):
        self._parse_cfg(cfg)  # parse config first
        super().__init__(
            soccfg, reps=cfg["reps"], final_delay=cfg["adc"]["relax_delay"], cfg=cfg
        )

    def _initialize(self, cfg):
        if "sweep" in cfg:
            # add loops
            for name, sweep in cfg["sweep"].items():
                self.add_loop(name, count=sweep["expts"])

        super()._initialize(cfg)

    def _parse_cfg(self, cfg: dict):
        # dac and adc config
        self.dac: Dict[str, Any] = cfg.get("dac", {})
        self.adc: Dict[str, Any] = cfg.get("adc", {})

        # dac pulse
        for name, pulse in self.dac.items():
            if not isinstance(pulse, dict) or not name.endswith("_pulse"):
                continue
            if hasattr(self, name):
                raise ValueError(f"Pulse name {name} already exists")
            setattr(self, name, pulse)

        # dac pulse channel count
        self.ch_count = defaultdict(int)
        nqzs = dict()
        for pulse in self.dac.values():
            if not isinstance(pulse, dict) or "ch" not in pulse:
                continue
            ch, nqz = pulse["ch"], pulse["nqz"]
            self.ch_count[ch] += 1
            cur_nqz = nqzs.setdefault(ch, nqz)
            assert cur_nqz == nqz, "Found different nqz on the same channel"

        # reset and readout modules
        self.resetM = make_reset(self.dac["reset"])
        self.readoutM = make_readout(self.dac["readout"])

    def _local_acquire(self, soc, *args, **kwargs):
        return super().acquire(soc, *args, **kwargs)

    def _local_acquire_decimated(self, soc, *args, **kwargs):
        return super().acquire_decimated(soc, *args, **kwargs)
