import qick as qk  # type: ignore

from ..base import BaseTwoToneProgram, create_pulse  # noqa: F401


class BaseTimeProgram(qk.RAveragerProgram):
    def initialize(self):
        BaseTwoToneProgram.initialize(self)

        cfg = self.cfg
        self.qub_cfg = cfg["qubit"]

        sweep_cfg = cfg["sweep"]
        cfg["start"] = self.us2cycles(sweep_cfg["start"])
        cfg["step"] = self.us2cycles(sweep_cfg["step"])
        cfg["expts"] = sweep_cfg["expts"]

        # set the initial parameters
        self.q_rp = self.ch_page(self.qub_cfg["qub_ch"])
        self.r_wait = 3
        self.regwi(self.q_rp, self.r_wait, cfg["start"])

        # declare the resonator channel and readout channels
        self.declare_gen(ch=self.qub_cfg["qub_ch"], nqz=self.qub_cfg["nqz"])

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, "+", self.cfg["step"])
