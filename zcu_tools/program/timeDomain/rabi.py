from .base import BaseTimeProgram


class AmpRabiProgram(BaseTimeProgram):
    def initialize(self):
        # initialize qubit pulse gain
        self.cfg["qub_pulse"]["gain"] = self.cfg["sweep"]["start"]
        return super().initialize()

    def body(self):
        self.flux_ctrl.trigger()

        # qubit pulse
        self.pulse(ch=self.qub_cfg["qub_ch"])
        self.sync_all(self.us2cycles(0.05))

        # measure
        self.measure_pulse()
