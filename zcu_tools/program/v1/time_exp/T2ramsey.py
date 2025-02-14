from .base import TimeProgram, PULSE_DELAY, set_pulse


class T2RamseyProgram(TimeProgram):
    def body(self):
        # reset
        self.resetM.reset_qubit(self)

        # pi/2 - wait - pi/2 sequence
        ch = self.qub_pulse["ch"]  # type: ignore
        if self.ch_count[ch] > 1:
            set_pulse(self, self.qub_pulse, waveform="qub_pulse")  # type: ignore
        self.pulse(ch=ch)
        self.sync_all()

        self.sync(self.q_rp, self.q_wait)

        self.pulse(ch=ch)
        self.sync_all(self.us2cycles(PULSE_DELAY))  # type: ignore

        # measure
        self.readoutM.readout_qubit(self)
