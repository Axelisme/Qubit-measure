from myqick.asm_v2 import QickProgramV2, QickParam

from zcu_tools.program.base import MyProgram
from zcu_tools.program.base.simulate import Pulse, visualize_simulation


class SimulateV2(MyProgram, QickProgramV2):
    """
    Record the pulse sequence in a list of Pulse objects, So we can plot them later.
    It is performed by overriding the delay and pulse methods.
    It isn't very accurate, but it is enough for most cases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim_cur_t = 0.0
        self.pulse_end_t = 0.0
        self.pulse_list = []

    def delay(self, t, tag=None):
        super().delay(t, tag=tag)

        if isinstance(t, QickParam):
            t = t.start

        self.sim_cur_t = t

    def delay_auto(self, t=0, gens=True, ros=True, tag=None):
        super().delay_auto(t, gens=gens, ros=ros, tag=tag)

        if isinstance(t, QickParam):
            t = t.start

        if self.pulse_end_t > self.sim_cur_t:
            self.sim_cur_t = self.pulse_end_t

        self.sim_cur_t += t

    def pulse(self, ch, name, t=0, tag=None):
        super().pulse(ch, name, t=t, tag=tag)

        if isinstance(t, QickParam):
            t = t.start

        start_t = self.sim_cur_t + t
        pulse_cfg = self.pulse_map[name]
        self.pulse_list.append(Pulse(start_t, pulse_cfg))

        self.pulse_end_t = max(start_t + pulse_cfg["length"], self.pulse_end_t)

    def visualize(self):
        visualize_simulation(self.pulse_list, self.sim_cur_t)
