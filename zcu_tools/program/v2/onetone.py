from .base import MyProgramV2


def onetone_body(prog: MyProgramV2, _):
    # reset
    prog.resetM.reset_qubit(prog)

    # readout
    prog.readoutM.readout_qubit(prog)


class OneToneProgram(MyProgramV2):
    def _initialize(self, cfg):
        self.resetM.init(self)
        self.readoutM.init(self)

        super()._initialize(cfg)

    def _body(self, cfg):
        onetone_body(self, cfg)
