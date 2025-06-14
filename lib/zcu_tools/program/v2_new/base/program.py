from typing import Any, Dict, Optional

from myqick import QickConfig
from myqick.asm_v2 import AveragerProgramV2
from zcu_tools.program.base import MyProgram

from .pulse import add_pulse, create_waveform
from .readout import make_readout
from .reset import make_reset


class MyProgramV2(MyProgram, AveragerProgramV2):
    def __init__(self, soccfg: QickConfig, cfg: Dict[str, Any], **kwargs) -> None:
        # v2 program need to pass reps and final_delay to init
        super().__init__(
            soccfg,
            cfg=cfg,
            reps=cfg["reps"],
            initial_delay=0.0,
            final_delay=cfg["adc"]["relax_delay"],
            **kwargs,
        )

    def _make_modules(self) -> None:
        self.resetM = make_reset(self.cfg["dac"]["reset"])
        self.readoutM = make_readout(self.cfg["dac"]["readout"])

    def _parse_cfg(self, cfg: Dict[str, Any]) -> None:
        # instaniate v2 reset and readout modules
        super()._parse_cfg(cfg)
        self._make_modules()

    def declare_pulse(
        self,
        pulse: Dict[str, Any],
        waveform: str,
        ro_ch: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.declare_gen(
            pulse["ch"],
            nqz=pulse["nqz"],
            mixer_freq=pulse.get("mixer_freq"),
            mux_freqs=pulse.get("mux_freqs"),
            mux_gains=pulse.get("mux_gains"),
            mux_phases=pulse.get("mux_phases"),
            ro_ch=pulse.get("ro_ch"),
        )
        create_waveform(self, waveform, pulse)

        add_pulse(self, pulse, waveform=waveform, ro_ch=ro_ch, **kwargs)

    def _init_modules(self) -> None:
        # initialize v2 reset and readout modules
        self.resetM.init(self)
        self.readoutM.init(self)

    def _initialize(self, cfg: Dict[str, Any]) -> None:
        # add v2 sweep loops
        if "sweep" in cfg:
            for name, sweep in cfg["sweep"].items():
                self.add_loop(name, count=sweep["expts"])

        self._init_modules()

    def acquire(self, soc, **kwargs) -> list:
        # v2 program need to pass soft_avgs to acquire
        return super().acquire(soc, soft_avgs=self.cfg["soft_avgs"], **kwargs)

    def acquire_decimated(self, soc, **kwargs) -> list:
        # v2 program need to pass soft_avgs to acquire_decimated
        return super().acquire_decimated(soc, soft_avgs=self.cfg["soft_avgs"], **kwargs)
