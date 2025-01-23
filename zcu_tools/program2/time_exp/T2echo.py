from ..base import set_pulse
from .base import TimeProgram
from ..twotone import declare_pulse, PULSE_DELAY, SYNC_TIME


class T2EchoProgram(TimeProgram):
    def initialize(self):
        self.parse_sweep()
        self.resetM.init(self)
        self.readoutM.init(self)

        declare_pulse(self, self.pi_pulse, "pi")
        declare_pulse(self, self.pi2_pulse, "pi2")

        assert (
            self.pi_pulse["ch"] == self.pi2_pulse["ch"]
        ), "pi and pi/2 pulse must be on the same channel"
        self.declare_wait_reg(self.pi_pulse["ch"])

        self.synci(SYNC_TIME)

    def body(self):
        # reset
        self.resetM.reset_qubit(self)

        # pi/2 - wait - pi - wait - pi/2 sequence
        ch = self.pi_pulse["ch"]
        set_pulse(self, self.pi2_pulse, waveform="pi2")
        self.pulse(ch=ch)
        self.sync_all()

        self.sync(self.q_rp, self.q_wait)

        set_pulse(self, self.pi_pulse, waveform="pi")
        self.pulse(ch=ch)
        self.sync_all()

        self.sync(self.q_rp, self.q_wait)

        set_pulse(self, self.pi2_pulse, waveform="pi2")
        self.pulse(ch=ch)
        self.sync_all(self.us2cycles(PULSE_DELAY))

        # readout
        self.readoutM.readout_qubit(self)
