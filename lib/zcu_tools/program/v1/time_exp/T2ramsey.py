from ..base import SYNC_TIME, MyRAveragerProgram
from ..twotone import PULSE_DELAY, declare_pulse, set_pulse  # noqa


class T2RamseyProgram(MyRAveragerProgram):
    def declare_wait_reg(self):
        self.q_wait = 3
        self.regwi(self.q_rp, self.q_wait, self.us2cycles(self.cfg["start"]))
        self.q_wait_s = self.us2cycles(self.cfg["step"])

    def declare_phase_reg(self, detune):
        qub_ch = self.qub_pulse["ch"]
        self.q_phase = self.sreg(qub_ch, "phase")
        self.q_phase_0 = 4  # original phase
        self.q_phase_t = 5  # incremented phase
        self.mathi(self.q_rp, self.q_phase_0, self.q_phase, "+", 0)
        self.mathi(self.q_rp, self.q_phase_t, self.q_phase, "+", 0)
        self.q_phase_s = self.deg2reg(detune * self.cfg["step"], gen_ch=qub_ch)

    def initialize(self):
        self.resetM.init(self)
        self.readoutM.init(self)
        declare_pulse(self, self.qub_pulse, "qub_pulse")

        self.q_rp = self.ch_page(self.qub_pulse["ch"])

        self.declare_wait_reg()
        self.declare_phase_reg(self.cfg["detune"])

        self.synci(SYNC_TIME)

    def update(self):
        self.mathi(self.q_rp, self.q_wait, self.q_wait, "+", self.q_wait_s)
        self.mathi(self.q_rp, self.q_phase_t, self.q_phase_t, "+", self.q_phase_s)

    def overwrite_phase(self, r_phase):
        self.mathi(self.q_rp, self.q_phase, r_phase, "+", 0)

    def body(self):
        # reset
        self.resetM.reset_qubit(self)

        # pi/2 - wait - pi/2 sequence
        ch = self.qub_pulse["ch"]
        if self.ch_count[ch] > 1:
            set_pulse(self, self.qub_pulse, waveform="qub_pulse")
        self.overwrite_phase(self.q_phase_0)
        self.pulse(ch=ch)
        self.sync_all()

        self.sync(self.q_rp, self.q_wait)

        self.overwrite_phase(self.q_phase_t)
        self.pulse(ch=ch)
        self.sync_all(self.us2cycles(PULSE_DELAY))

        # measure
        self.readoutM.readout_qubit(self)
