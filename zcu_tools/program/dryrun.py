import functools
import operator
import time

import numpy as np
from tqdm.auto import tqdm


class DryRunProxy:
    def __init__(self):
        self.acc_buf = []
        self._interrupt = False

    def get_acc_buf(self):
        return self.acc_buf

    def _average_buf(
        self,
        prog,
        d_reps: np.ndarray,
        ret_std: bool = False,
    ) -> np.ndarray:
        avg_d = []
        if ret_std:
            std_d = []
        for i_ch, (ch, ro) in enumerate(prog.ro_chs.items()):
            # average over the avg_level
            avg = d_reps[i_ch].sum(axis=prog.avg_level) / prog.loop_dims[prog.avg_level]
            avg /= ro["length"]
            if ret_std:
                std = d_reps[i_ch].std(axis=prog.avg_level)
                std /= ro["length"]

            # the reads_per_shot axis should be the first one
            avg_d.append(np.moveaxis(avg, -2, 0))
            if ret_std:
                std_d.append(np.moveaxis(std, -2, 0))

        if ret_std:
            return avg_d, std_d
        return avg_d

    def acquire(
        self,
        prog,
        soft_avgs=1,
        threshold=None,
        progress=True,
        ret_std=False,
        callback=None,
        **kwargs,
    ):
        total_count = functools.reduce(operator.mul, prog.loop_dims)
        self.acc_buf = [np.zeros((*prog.loop_dims, 1, 2), dtype=np.int64)]

        hiderounds = True
        hidereps = True
        if progress:
            if soft_avgs > 1:
                hiderounds = False
            else:
                hidereps = False

        avg_d = None
        std2_d = None
        for ir in tqdm(range(soft_avgs), disable=hiderounds):
            with tqdm(total=total_count, disable=hidereps) as pbar:
                if hasattr(prog, "_handle_early_stop"):
                    prog._handle_early_stop()

                time.sleep(0.1)  # simulate acquisition time

                pbar.update()

            # if we're thresholding, apply the threshold before averaging
            if threshold is None:
                d_reps = self.acc_buf
                if ret_std:
                    round_d, round_std = self._average_buf(prog, d_reps, ret_std=True)
                else:
                    round_d = self._average_buf(prog, d_reps)
            else:
                assert not ret_std, "ret_std is not supported with thresholding"
                raise NotImplementedError("Thresholding is not supported")

            # sum over rounds axis
            if avg_d is None:
                avg_d = round_d
            else:
                for ii, d in enumerate(round_d):
                    avg_d[ii] += d

            if ret_std:
                if std2_d is None:
                    std2_d = [d**2 + s**2 for d, s in zip(round_d, round_std)]
                else:
                    for ii, (d, s) in enumerate(zip(round_d, round_std)):
                        std2_d[ii] += d**2 + s**2

            # callback
            if callback is not None:
                callback(ir, avg_d)

        # divide total by rounds
        for d in avg_d:
            d /= soft_avgs

        if ret_std:
            for i in range(len(std2_d)):
                std2_d[i] = np.sqrt(std2_d[i] / soft_avgs - avg_d[i] ** 2)
            return avg_d, std2_d
        else:
            return avg_d

    def acquire_decimated(
        self,
        prog,
        soft_avgs=1,
        progress=True,
        callback=None,
        **kwargs,
    ):
        total_count = functools.reduce(operator.mul, prog.loop_dims)

        # Initialize data buffers
        # buffer for decimated data
        dec_buf = [np.zeros((total_count, 2), dtype=float)]

        # for each soft average, run and acquire decimated data
        for ir in tqdm(range(soft_avgs), disable=not progress):
            # buffer for accumulated data (for convenience/debug)
            self.acc_buf = []

            if hasattr(prog, "_handle_early_stop"):
                prog._handle_early_stop()

            dec_buf[0] += np.random.rand(total_count, 2)
            self.acc_buf.append(np.random.rand(*prog.loop_dims, 1, 2))

            # callback
            if callback is not None:
                callback(ir)

        # average the decimated data
        result = []
        d_avg = dec_buf[0] / soft_avgs
        if total_count == 1:
            # simple case: data is 1D (one rep and one shot), just average over rounds
            result.append(d_avg)
        else:
            # split the data into the individual reps
            d_reshaped = d_avg.reshape(total_count * 1, -1, 2)
            result.append(d_reshaped)
        return result

    def test_remote_callback(self) -> bool:
        return True
