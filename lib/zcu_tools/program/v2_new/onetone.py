from .base import MyProgramV2


class OneToneProgram(MyProgramV2):
    def _body(self, _):
        # reset
        self.resetM.reset_qubit(self)

        # readout
        self.readoutM.readout_qubit(self)
