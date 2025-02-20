from .base import TimeProgram, set_pulse


class T1Program(TimeProgram):
    def body(self):
        # reset
        self.resetM.reset_qubit(self)

        # pi pulse
        ch = self.qub_pulse["ch"]  # type: ignore
        if self.ch_count[ch] > 1:
            set_pulse(self, self.qub_pulse, waveform="qub_pulse")  # type: ignore
        self.pulse(ch=ch)
        self.sync_all()

        # wait for specified time
        self.sync(self.q_rp, self.q_wait)

        # measure
        self.readoutM.readout_qubit(self)
