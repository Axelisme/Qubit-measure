from .base import MyAveragerProgram, MyRAveragerProgram, SYNC_TIME


def onetone_body(prog: MyAveragerProgram, before_readout=None):
    # reset
    prog.resetM.reset_qubit(prog)

    # readout
    prog.readoutM.readout_qubit(prog, before_readout)


class OneToneProgram(MyAveragerProgram):
    def initialize(self):
        super().initialize()
        self.synci(SYNC_TIME)

    def body(self):
        onetone_body(self)


class RGainOneToneProgram(MyRAveragerProgram):
    def declare_gain_reg(self):
        # setup gain register
        ch = self.res_pulse["ch"]  
        self.r_rp = self.ch_page(ch)
        self.r_gain = self.sreg(ch, "gain")
        self.r_gain2 = self.sreg(ch, "gain2")
        self.r_gain_t = 3
        self.mathi(self.r_rp, self.r_gain_t, self.r_gain, "+", 0)
        self.r_step = self.cfg["step"]

    def initialize(self):
        self.res_pulse["gain"] = self.cfg["start"]  

        super().initialize()
        self.declare_gain_reg()

        self.synci(SYNC_TIME)

    def body(self):
        onetone_body(self, self.set_gain_reg)  

    def set_gain_reg(self):
        self.mathi(self.r_rp, self.r_gain, self.r_gain_t, "+", 0)
        self.mathi(self.r_rp, self.r_gain2, self.r_gain_t, ">>", 1)  # divide by 2

    def update(self):
        self.mathi(self.r_rp, self.r_gain_t, self.r_gain_t, "+", self.r_step)
