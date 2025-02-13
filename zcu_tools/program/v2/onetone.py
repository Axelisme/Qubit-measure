from .base import MyProgram


def onetone_body(prog: MyProgram, _):
    # reset
    prog.resetM.reset_qubit(prog)

    # readout
    prog.readoutM.readout_qubit(prog)


class OneToneProgram(MyProgram):
    def _initialize(self, cfg):
        self.resetM.init(self)
        self.readoutM.init(self)

        super()._initialize(cfg)

    def _body(self, cfg):
        onetone_body(self, cfg)
