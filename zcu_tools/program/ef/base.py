from ..base import BaseOneToneProgram, create_waveform, set_pulse


class BaseEFProgram(BaseOneToneProgram):
    def parse_cfg(self):
        BaseOneToneProgram.parse_cfg(self)

        self.qub_cfg = self.cfg["qubit"]
        self.ge_pulse = self.cfg["ge_pulse"]
        self.ef_pulse = self.cfg["ef_pulse"]

    def setup_qubit(self):
        qub_ch = self.qub_cfg["qub_ch"]
        self.declare_gen(ch=qub_ch, nqz=self.qub_cfg["nqz"])
        self.ge_wavform = create_waveform(self, qub_ch, self.ge_pulse)
        self.ef_wavform = create_waveform(self, qub_ch, self.ef_pulse)

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()
        self.setup_qubit()

        self.synci(200)

    def pulse_ge(self):
        qub_ch = self.qub_cfg["qub_ch"]

        set_pulse(self, qub_ch, self.ge_pulse, self.ge_wavform)
        self.pulse(ch=qub_ch)

    def pulse_ef(self):
        qub_ch = self.qub_cfg["qub_ch"]

        set_pulse(self, qub_ch, self.ef_pulse, self.ef_wavform)
        self.pulse(ch=qub_ch)
