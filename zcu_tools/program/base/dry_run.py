import time

import numpy as np
from tqdm.auto import trange


class DryRunProgram:
    def __init__(self, soccfg, cfg):
        self.cfg = cfg
        self.make_program()
        self.soft_avgs = 1
        if "soft_avgs" in cfg:
            self.soft_avgs = cfg["soft_avgs"]
        if "rounds" in cfg:
            self.soft_avgs = cfg["rounds"]
        cfg["soft_avgs"] = self.soft_avgs

    def initialize(self):
        pass

    def body(self):
        pass

    def make_program(self):
        self.initialize()
        self.body()

    def declare_gen(self, *args, **kwargs):
        pass

    def declare_readout(self, *args, **kwargs):
        pass

    def us2cycles(self, us, **kwargs):
        if isinstance(us, np.ndarray):
            return (us * 1e3).astype(int)
        return int(us * 1e3)

    def cycles2us(self, cycles, **kwargs):
        if isinstance(cycles, np.ndarray):
            return cycles.astype(float) / 1e3
        return float(cycles) / 1e3

    def add_gauss(self, *args, **kwargs):
        pass

    def add_DRAG(self, *args, **kwargs):
        pass

    def add_cosine(self, *args, **kwargs):
        pass

    def freq2reg(self, freq, **kwargs):
        if isinstance(freq, np.ndarray):
            return freq.astype(int)
        return int(freq)

    def deg2reg(self, deg, **kwargs):
        if isinstance(deg, np.ndarray):
            return deg.astype(int)
        return int(deg)

    def default_pulse_registers(self, *args, **kwargs):
        pass

    def set_pulse_registers(self, *args, **kwargs):
        pass

    def synci(self, cycles):
        pass

    def sync_all(self, *args, **kwargs):
        pass

    def mathi(self, *args, **kwargs):
        pass

    def pulse(self, *args, **kwargs):
        pass

    def ch_page(self, ch):
        return ch

    def sreg(self, ch, name):
        return ch

    def measure(self, *args, **kwargs):
        pass

    def acquire_avg(self, soc, progress, round_callback, callback_period):
        # AveragerProgram
        avg_d = [np.zeros((1, 2))]
        for ir in trange(self.soft_avgs, leave=False, disable=not progress):
            avg_d[0][0] += np.mean(np.random.rand(self.cfg["reps"], 2), axis=0)
            if round_callback is not None and ir % callback_period == 0:
                cur_avg = [d / (ir + 1) for d in avg_d]
                if callable(round_callback):
                    round_callback(ir, cur_avg)
                else:
                    round_callback.oneway_callback(ir, cur_avg)
            time.sleep(0.01)
        for d in avg_d:
            d /= self.soft_avgs
        avg_i = [d[..., 0] for d in avg_d]
        avg_q = [d[..., 1] for d in avg_d]
        return avg_i, avg_q

    def acquire_ravg(self, soc, progress, round_callback, callback_period):
        # RAveragerProgram
        xs = self.cfg["start"] + np.arange(self.cfg["expts"]) * self.cfg["step"]
        ys = np.sin(xs / (xs.max() - xs.min()) * 2 * np.pi) * 0.1
        ys = np.stack((ys, ys), axis=-1)
        avg_d = [np.zeros((1, len(xs), 2))]
        for ir in trange(self.soft_avgs, leave=False, disable=not progress):
            y = ys + np.mean(np.random.rand(self.cfg["reps"], len(xs), 2), axis=0)
            avg_d[0][0] += y
            if round_callback is not None and ir % callback_period == 0:
                cur_avg = [d / (ir + 1) for d in avg_d]
                if callable(round_callback):
                    round_callback(ir, cur_avg)
                else:
                    round_callback.oneway_callback(ir, cur_avg)
            time.sleep(0.1)
        for d in avg_d:
            d /= self.soft_avgs
        avg_i = [d[..., 0] for d in avg_d]
        avg_q = [d[..., 1] for d in avg_d]
        return xs, avg_i, avg_q

    def acquire(
        self, soc, progress=False, round_callback=None, callback_period=100, **kwargs
    ):
        if hasattr(self, "update"):  # RAveragerProgram
            return self.acquire_ravg(
                soc,
                progress=progress,
                round_callback=round_callback,
                callback_period=callback_period,
                **kwargs,
            )
        else:
            return self.acquire_avg(
                soc,
                progress=progress,
                round_callback=round_callback,
                callback_period=callback_period,
                **kwargs,
            )

    def acquire_decimated(
        self, soc, progress=False, round_callback=None, callback_period=100
    ):
        # compare to acquire_avg, it have one more dimension at the end
        num = soc.us2cycles(3.32)
        ys = np.exp(-(((np.linspace(0, 3.32, num) - 1.66) * 2) ** 2))
        ys = np.stack((ys, ys), axis=-1)
        avg_d = np.zeros((num, 2))
        for ir in trange(self.soft_avgs, leave=False, disable=not progress):
            y = ys + np.mean(np.random.rand(self.cfg["reps"], num, 2), axis=0)
            avg_d += y
            time.sleep(0.01)
            if round_callback is not None and ir % callback_period == 0:
                cur_avg = [d / (ir + 1) for d in avg_d]
                if callable(round_callback):
                    round_callback(ir, cur_avg)
                else:
                    round_callback.oneway_callback(ir, cur_avg)
        avg_d /= self.soft_avgs

        avg_i = [d[..., 0] for d in avg_d]
        avg_q = [d[..., 1] for d in avg_d]
        return [[avg_i, avg_q]]
