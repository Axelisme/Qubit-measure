from qick.asm_v1 import AcquireProgram
from .flux import make_fluxControl
from .pulse import create_pulse


class BaseOneToneProgram(AcquireProgram):
    def parse_cfg(self):
        self.flux_cfg = self.cfg["flux"]
        self.res_cfg = self.cfg["resonator"]
        self.res_pulse = self.cfg["res_pulse"]

    def setup_readout(self):
        # declare the resonator channel and readout channels
        res_ch = self.res_cfg["res_ch"]
        self.declare_gen(ch=res_ch, nqz=self.res_cfg["nqz"])
        for ro_ch in self.res_cfg["ro_chs"]:
            self.declare_readout(
                ch=ro_ch,
                length=self.us2cycles(self.cfg["readout_length"]),
                freq=self.res_pulse["freq"],
                gen_ch=res_ch,
            )
        # create the resonator pulse
        self.res_wavform = create_pulse(
            self, self.res_cfg["res_ch"], self.res_pulse, for_readout=True
        )

    def setup_flux(self):
        flux_cfg = self.flux_cfg
        self.flux_ctrl = make_fluxControl(self, flux_cfg["method"], flux_cfg)
        self.flux_ctrl.set_flux(flux=flux_cfg["value"])

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()

        self.synci(200)

    def measure_pulse(self):
        cfg = self.cfg
        res_cfg = self.res_cfg

        self.measure(
            pulse_ch=res_cfg["res_ch"],
            adcs=res_cfg["ro_chs"],
            pins=[res_cfg["ro_chs"][0]],
            adc_trig_offset=self.us2cycles(cfg["adc_trig_offset"]),
            wait=True,
            syncdelay=self.us2cycles(cfg["relax_delay"]),
        )


class BaseTwoToneProgram(BaseOneToneProgram):
    def parse_cfg(self):
        super().parse_cfg()
        self.qub_cfg = self.cfg["qubit"]
        self.qub_pulse = self.cfg["qub_pulse"]

    def setup_qubit(self):
        self.declare_gen(ch=self.qub_cfg["qub_ch"], nqz=self.qub_cfg["nqz"])
        self.qub_wavform = create_pulse(self, self.qub_cfg["qub_ch"], self.qub_pulse)

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()
        self.setup_qubit()

        self.synci(200)
