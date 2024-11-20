import qick as qk  # type: ignore

from .flux import make_fluxControl
from .pulse import create_pulse


class BaseOneToneProgram(qk.AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        self.res_cfg = cfg["resonator"]

        res_ch = self.res_cfg["res_ch"]

        # declare the resonator channel and readout channels
        self.declare_gen(ch=res_ch, nqz=self.res_cfg["nqz"])
        for ro_ch in self.res_cfg["ro_chs"]:
            self.declare_readout(
                ch=ro_ch,
                length=self.us2cycles(cfg["readout_length"]),
                freq=cfg["res_pulse"]["freq"],
                gen_ch=res_ch,
            )

        # prepare the flux control
        flux_cfg = cfg["flux"]
        self.flux_ctrl = make_fluxControl(self, flux_cfg["method"], flux_cfg)
        self.flux_ctrl.set_flux(flux=flux_cfg["value"])

        # set the pulse registers for resonator and qubit
        create_pulse(self, res_ch, cfg["res_pulse"], for_readout=True)

    def body(self):
        raise NotImplementedError("Please implement the body method in your program.")


class BaseTwoToneProgram(qk.AveragerProgram):
    def initialize(self):
        BaseOneToneProgram.initialize(self)

        self.qub_cfg = self.cfg["qubit"]

        self.declare_gen(ch=self.cfg["qubit"]["qub_ch"], nqz=self.cfg["qubit"]["nqz"])

    def body(self):
        raise NotImplementedError("Please implement the body method in your program.")
