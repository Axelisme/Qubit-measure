import qick as qk  # type: ignore

from .base import BaseOneToneProgram, create_pulse


class SingleShotProgram(qk.RAveragerProgram):
    def initialize(self):
        BaseOneToneProgram.initialize(self)

        cfg = self.cfg
        self.qub_cfg = cfg["qubit"]
        self.qub_pulse_cfg = cfg["qub_pulse"]

        # overwrite the number of experiments and repetitions
        if "expts" in cfg or "reps" in cfg:
            if "expts" in cfg:
                print(f"Find expts is {cfg['expts']}")
            if "reps" in cfg:
                print(f"Find reps is {cfg['reps']}")
            print("Warning: expts and reps are overwritten by 2 and shots")
        cfg["start"] = 0
        cfg["step"] = self.qub_pulse_cfg["gain"]
        cfg["expts"] = 2
        cfg["reps"] = cfg["shots"]

        # set intial gain to 0
        self.pi_gain = cfg["step"]
        self.qub_pulse_cfg["gain"] = 0

        qub_ch = self.qub_cfg["qub_ch"]
        self.q_rp = self.ch_page(qub_ch)
        self.r_gain = self.sreg(qub_ch, "gain")

        self.declare_gen(ch=qub_ch, nqz=self.qub_cfg["nqz"])

        create_pulse(self, qub_ch, self.qub_pulse_cfg)

        self.synci(200)

    def body(self):
        cfg = self.cfg
        res_cfg = self.res_cfg
        qub_cfg = self.qub_cfg

        self.flux_ctrl.trigger()

        # qubit pulse
        self.pulse(ch=qub_cfg["qub_ch"])
        self.sync_all(self.us2cycles(0.05))

        # measure
        self.measure(
            pulse_ch=res_cfg["res_ch"],
            adcs=res_cfg["ro_chs"],
            pins=[res_cfg["ro_chs"][0]],
            adc_trig_offset=self.us2cycles(cfg["adc_trig_offset"]),
            wait=True,
            syncdelay=self.us2cycles(cfg["relax_delay"]),
        )

    def update(self):
        self.mathi(self.q_rp, self.r_gain, self.r_gain, "+", self.pi_gain)

    def acquire(self, soc, progress=False):
        super().acquire(soc, progress=progress)
        return self.collect_shots()

    def collect_shots(self):
        cfg = self.cfg
        expts, reps = cfg["expts"], cfg["reps"]
        readout_length = cfg["readout_length"]
        i0 = self.di_buf[0].reshape((expts, reps)) / readout_length
        q0 = self.dq_buf[0].reshape((expts, reps)) / readout_length
        return i0, q0
