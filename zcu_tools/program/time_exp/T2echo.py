from .base import TimeProgram
from ..base import set_pulse, create_waveform


class T2EchoProgram(TimeProgram):
    def parse_cfg(self):
        TimeProgram.parse_cfg()

        self.pi_pulse = self.dac_cfg["pi_pulse"]
        self.pi2_pulse = self.dac_cfg["pi2_pulse"]

    def setup_qubit(self):
        pi_pulse = self.pi_pulse
        pi2_pulse = self.pi2_pulse

        self.declare_gen(pi_pulse["ch"], nqz=pi_pulse["nqz"])
        if pi_pulse["ch"] != pi2_pulse["ch"]:
            self.declare_gen(pi2_pulse["ch"], nqz=pi2_pulse["nqz"])
        else:
            assert (
                pi2_pulse["nqz"] == pi_pulse["nqz"]
            ), "pi and pi/2 pulse on the same channel must have the same nqz"

        create_waveform(self, "pi_pulse", pi_pulse)
        set_pulse(self, pi_pulse, waveform="pi_pulse")

        create_waveform(self, "pi2_pulse", pi2_pulse)
        set_pulse(self, pi2_pulse, waveform="pi2_pulse")

    def body(self):
        pi_cfg = self.pi_pulse
        pi2_cfg = self.pi2_pulse

        # pi/2 - wait - pi - wait - pi/2 sequence
        set_pulse(self, pi2_cfg, waveform="pi2_pulse")
        self.pulse(ch=pi2_cfg["ch"])
        self.sync_all()

        self.sync(self.q_rp, self.r_wait)

        set_pulse(self, pi_cfg, waveform="pi_pulse")
        self.pulse(ch=pi_cfg["ch"])
        self.sync_all()

        self.sync(self.q_rp, self.r_wait)

        set_pulse(self, pi2_cfg, waveform="pi2_pulse")
        self.pulse(ch=pi2_cfg["ch"])
        self.sync_all(self.us2cycles(0.05))

        self.measure_pulse()
