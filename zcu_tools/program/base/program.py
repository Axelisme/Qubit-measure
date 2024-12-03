from qick.asm_v1 import AcquireProgram

from zcu_tools.device.flux import get_fluxControl
from .pulse import create_waveform, is_single_pulse, set_pulse


class BaseOneToneProgram(AcquireProgram):
    def parse_cfg(self):
        assert hasattr(self, "cfg"), "cfg is not defined"
        assert isinstance(self.cfg, dict), "cfg is not a dict"

        self.flux_dev = self.cfg["flux_dev"]
        self.res_cfg = self.cfg["resonator"]
        self.res_pulse = self.cfg["res_pulse"]

    def setup_readout(self):
        assert is_single_pulse(self.res_pulse), "Currently only support one pulse cfg"

        self.res_ch = self.res_cfg["res_ch"]
        self.ro_chs = self.res_cfg["ro_chs"]

        # declare the resonator channel and readout channels
        res_ch = self.res_ch
        ro_chs = self.ro_chs
        self.declare_gen(ch=res_ch, nqz=self.res_pulse["nqz"])
        for ro_ch in ro_chs:
            self.declare_readout(
                ch=ro_ch,
                length=self.us2cycles(self.cfg["readout_length"], ro_ch=ro_ch),
                freq=self.res_pulse["freq"],
                gen_ch=res_ch,
            )
        # create the resonator pulse
        self.res_wavform = create_waveform(self, res_ch, self.res_pulse)
        set_pulse(self, self.res_pulse, res_ch, ro_chs[0], self.res_wavform)

    def setup_flux(self):
        flux_cls = get_fluxControl(self.flux_dev)
        self.flux_ctrl = flux_cls(self)
        self.flux_ctrl.set_flux(self.cfg.get("flux"))

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
        qub_ch = self.qub_cfg["qub_ch"]
        self.declare_gen(ch=qub_ch, nqz=self.qub_pulse["nqz"])
        self.qub_wavform = create_waveform(self, qub_ch, self.qub_pulse)
        if isinstance(self.qub_wavform, str):
            set_pulse(self, self.qub_pulse, qub_ch, waveform=self.qub_wavform)

    def initialize(self):
        self.parse_cfg()
        self.setup_flux()
        self.setup_readout()
        self.setup_qubit()

        self.synci(200)
