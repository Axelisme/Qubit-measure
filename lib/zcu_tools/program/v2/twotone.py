from typing import Callable

from myqick.asm_v2 import AsmV2, Macro

from .base import MyProgramV2, declare_pulse


class RecordTime(Macro):
    def __init__(self, register_fn: Callable[[int], None]) -> None:
        self.register_fn = register_fn

    def preprocess(self, prog):
        self.register_fn(prog.get_max_timestamp(gens=True, ros=False))

    def expand(self, prog):
        return []


class WithMinLength:
    def __init__(self, prog: AsmV2, length: float = None):
        self.prog = prog
        self.length = length
        self.start_t = 0.0

    def set_start_time(self, t):
        self.start_t = t

    def __enter__(self):
        self.prog.append_macro(RecordTime(self.set_start_time))

    def __exit__(self, exec_type, exec_value, exec_tb):
        if self.length is not None:
            self.prog.delay(t=self.start_t + self.length)


class TwoToneProgram(MyProgramV2):
    PULSE_DELAY = 0.01  # us

    def _initialize(self, cfg):
        declare_pulse(self, self.qub_pulse, "qub_pulse")
        super()._initialize(cfg)

    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # qubit pulse
        qub_pulse = self.qub_pulse

        # use sweep to align pulse length has bug
        # if qub_pulse + post_delay = constant, it may raise
        # RuntimeError: requested sweep step is smaller than the available resolution
        # so use WithMinLength for patched
        with WithMinLength(self, qub_pulse.get("force_total_length")):
            self.delay_auto(qub_pulse.get("pre_delay", self.PULSE_DELAY), ros=False)
            self.pulse(qub_pulse["ch"], "qub_pulse")
            self.delay_auto(qub_pulse.get("post_delay", self.PULSE_DELAY), ros=False)

        # measure
        self.readoutM.readout_qubit(self)
