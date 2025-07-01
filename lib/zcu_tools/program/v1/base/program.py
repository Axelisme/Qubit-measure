from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from myqick import AveragerProgram, NDAveragerProgram, RAveragerProgram
from zcu_tools.auto import is_pulse_cfg
from zcu_tools.program.base import MyProgram

from .readout import make_readout
from .reset import make_reset

SYNC_TIME = 200  # cycles


class MyProgramV1(MyProgram):
    def _parse_cfg(self, cfg: Dict[str, Any]) -> None:
        super()._parse_cfg(cfg)

        # dac pulse channel check
        self.ch_count = defaultdict(int)
        nqzs = dict()
        for name, pulse in self.dac.items():
            if not is_pulse_cfg(name, pulse):
                continue
            ch, nqz = pulse["ch"], pulse["nqz"]
            self.ch_count[ch] += 1
            cur_nqz = nqzs.setdefault(ch, nqz)
            assert cur_nqz == nqz, "Found different nqz on the same channel"

        self._make_modules()

    def _make_modules(self) -> None:
        self.resetM = make_reset(self.cfg["dac"]["reset"])
        self.readoutM = make_readout(self.cfg["dac"]["readout"])

    def initialize(self) -> None:
        self.resetM.init(self)
        self.readoutM.init(self)


class MyAveragerProgram(MyProgramV1, AveragerProgram):
    def acquire(
        self, soc, readouts_per_experiment=None, save_experiments=None, **kwargs
    ):
        if readouts_per_experiment is not None:
            self.set_reads_per_shot(readouts_per_experiment)

        avg_d, std_d = super().acquire(soc, soft_avgs=self.soft_avgs, **kwargs)

        # reformat the data into separate I and Q arrays
        # save results to class in case you want to look at it later or for analysis
        raw = [d.reshape((-1, 2)) for d in self.get_raw()]
        self.di_buf = [d[:, 0] for d in raw]
        self.dq_buf = [d[:, 1] for d in raw]

        n_ro = len(self.ro_chs)
        std_di, std_dq = None, None
        if save_experiments is None:
            avg_di = [d[:, 0] for d in avg_d]
            avg_dq = [d[:, 1] for d in avg_d]
            std_di = [d[:, 0] for d in std_d]
            std_dq = [d[:, 1] for d in std_d]

        else:
            avg_di = [np.zeros(len(save_experiments)) for _ in self.ro_chs]
            avg_dq = [np.zeros(len(save_experiments)) for _ in self.ro_chs]
            std_di = [np.zeros(len(save_experiments)) for _ in self.ro_chs]
            std_dq = [np.zeros(len(save_experiments)) for _ in self.ro_chs]
            for i_ch in range(n_ro):
                for nn, ii in enumerate(save_experiments):
                    avg_di[i_ch][nn] = avg_d[i_ch][ii, 0]
                    avg_dq[i_ch][nn] = avg_d[i_ch][ii, 1]
                    std_di[i_ch][nn] = std_d[i_ch][ii, 0]
                    std_dq[i_ch][nn] = std_d[i_ch][ii, 1]

        return avg_di, avg_dq, std_di, std_dq


class MyRAveragerProgram(MyProgramV1, RAveragerProgram):
    def _parse_cfg(self, cfg: Dict[str, Any]) -> None:
        super()._parse_cfg(cfg)
        self.cfg["start"] = self.sweep_cfg["start"]
        self.cfg["step"] = self.sweep_cfg["step"]
        self.cfg["expts"] = self.sweep_cfg["expts"]

    def acquire(
        self, soc, readouts_per_experiment=None, save_experiments=None, **kwargs
    ):
        if readouts_per_experiment is not None:
            self.set_reads_per_shot(readouts_per_experiment)

        avg_d, std_d = super().acquire(soc, soft_avgs=self.soft_avgs, **kwargs)

        # reformat the data into separate I and Q arrays
        # save results to class in case you want to look at it later or for analysis
        raw = [d.reshape((-1, 2)) for d in self.get_raw()]
        self.di_buf = [d[:, 0] for d in raw]
        self.dq_buf = [d[:, 1] for d in raw]

        expt_pts = self.get_expt_pts()

        n_ro = len(self.ro_chs)
        if save_experiments is None:
            avg_di = [d[..., 0] for d in avg_d]
            avg_dq = [d[..., 1] for d in avg_d]
            std_di = [d[..., 0] for d in std_d]
            std_dq = [d[..., 1] for d in std_d]
        else:
            avg_di = [np.zeros((len(save_experiments), *d.shape[1:])) for d in avg_d]
            avg_dq = [np.zeros((len(save_experiments), *d.shape[1:])) for d in avg_d]
            std_di = [np.zeros((len(save_experiments), *d.shape[1:])) for d in std_d]
            std_dq = [np.zeros((len(save_experiments), *d.shape[1:])) for d in std_d]
            for i_ch in range(n_ro):
                for nn, ii in enumerate(save_experiments):
                    avg_di[i_ch][nn] = avg_d[i_ch][ii, ..., 0]
                    avg_dq[i_ch][nn] = avg_d[i_ch][ii, ..., 1]
                    std_di[i_ch][nn] = std_d[i_ch][ii, ..., 0]
                    std_dq[i_ch][nn] = std_d[i_ch][ii, ..., 1]

        return expt_pts, avg_di, avg_dq, std_di, std_dq


class MyNDAveragerProgram(MyProgramV1, NDAveragerProgram):
    def acquire(
        self, soc, readouts_per_experiment=None, save_experiments: List = None, **kwargs
    ):
        if readouts_per_experiment is not None:
            self.set_reads_per_shot(readouts_per_experiment)

        avg_d, std_d = super().acquire(soc, soft_avgs=self.soft_avgs, **kwargs)

        # reformat the data into separate I and Q arrays
        # save results to class in case you want to look at it later or for analysis
        raw = [d.reshape((-1, 2)) for d in self.get_raw()]
        self.di_buf = [d[:, 0] for d in raw]
        self.dq_buf = [d[:, 1] for d in raw]

        expt_pts = self.get_expt_pts()

        n_ro = len(self.ro_chs)
        if save_experiments is None:
            avg_di = [d[..., 0] for d in avg_d]
            avg_dq = [d[..., 1] for d in avg_d]
            std_di = [d[..., 0] for d in std_d]
            std_dq = [d[..., 1] for d in std_d]
        else:
            avg_di = [np.zeros((len(save_experiments), *d.shape[1:])) for d in avg_d]
            avg_dq = [np.zeros((len(save_experiments), *d.shape[1:])) for d in avg_d]
            std_di = [np.zeros((len(save_experiments), *d.shape[1:])) for d in std_d]
            std_dq = [np.zeros((len(save_experiments), *d.shape[1:])) for d in std_d]
            for i_ch in range(n_ro):
                for nn, ii in enumerate(save_experiments):
                    avg_di[i_ch][nn] = avg_d[i_ch][ii, ..., 0]
                    avg_dq[i_ch][nn] = avg_d[i_ch][ii, ..., 1]
                    std_di[i_ch][nn] = std_d[i_ch][ii, ..., 0]
                    std_dq[i_ch][nn] = std_d[i_ch][ii, ..., 1]

        return expt_pts, avg_di, avg_dq, std_di, std_dq
