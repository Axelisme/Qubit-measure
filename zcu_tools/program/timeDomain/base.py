import qick as qk

from ..base import BaseTwoToneProgram, set_pulse  # noqa: F401


class BaseTimeProgram(qk.RAveragerProgram, BaseTwoToneProgram):
    def parse_cfg(self):
        super().parse_cfg()

        sweep_cfg = self.cfg["sweep"]
        self.cfg["start"] = self.us2cycles(sweep_cfg["start"])
        self.cfg["step"] = self.us2cycles(sweep_cfg["step"])
        self.cfg["expts"] = sweep_cfg["expts"]

    def setup_waittime(self):
        # setup wait time register
        self.q_rp = self.ch_page(self.qub_cfg["qub_ch"])
        self.r_wait = 3
        self.regwi(self.q_rp, self.r_wait, self.cfg["start"])

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()
        self.setup_qubit()
        self.setup_waittime()

        self.synci(200)

    def update(self):
        # update wait time
        self.mathi(self.q_rp, self.r_wait, self.r_wait, "+", self.cfg["step"])
