from .twotone import TwoToneProgram


class GEProgram(TwoToneProgram):
    def _initialize(self, cfg):
        super()._initialize(cfg)

        # add ge sweep to inner loop
        self.add_loop("ge_sweep", count=2)

    def acquire(self, soc, **kwargs):
        IQlist = super().acquire(soc, **kwargs)
        return [iq[..., 1, :] - iq[..., 0, :] for iq in IQlist]
