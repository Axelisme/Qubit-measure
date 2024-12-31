import Pyro4

from zcu_tools.schedule import measure_lookback


class RemoteSchedule:
    def __init__(self, soc):
        self.soc = soc

    def _override_cfg(self, cfg):
        # ignore flux control in remote mode
        cfg["flux_dev"] = "none"

    @Pyro4.expose
    def measure_lookback(self, cfg):
        self._override_cfg(cfg)
        return measure_lookback(self.soc, self.soc, cfg, progress=False)
