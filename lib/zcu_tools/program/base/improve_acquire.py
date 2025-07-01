import functools
import operator
from typing import Callable, List, Optional

import numpy as np
from tqdm.auto import tqdm

from myqick.qick_asm import AcquireMixin, logger
from myqick import obtain

AcquireCallbackType = Callable[
    [int, List[np.ndarray], List[np.ndarray]], None
]  # round, avg_data, std_data

AcquireDecimatedCallbackType = Callable[[int], None]  # round


class ImproveAcquireMixin(AcquireMixin):
    """
    Add some functionality to the AcquireMixin class to allow early stopping
    Including:
    - Early stopping
    - Callback function to be called after each round of acquisition
    - Stadard error infomation for the acquired data
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stop = False

    def set_early_stop(self) -> None:
        # tell program to return as soon as possible
        print("Program received early stop signal")
        self.early_stop = True

    def acquire(
        self,
        soc,
        soft_avgs=1,
        load_pulses=True,
        start_src="internal",
        threshold=None,
        angle=None,
        progress=True,
        remove_offset=True,
        callback: Optional[AcquireCallbackType] = None,
    ):
        self.early_stop = False

        # don't load memories now, we'll do that later
        self.config_all(soc, load_pulses=load_pulses, load_mem=False)

        if any(
            [x is None for x in [self.counter_addr, self.loop_dims, self.avg_level]]
        ):
            raise RuntimeError(
                "data dimensions need to be defined with setup_acquire() before calling acquire()"
            )

        # configure tproc for internal/external start
        soc.start_src(start_src)

        total_count = functools.reduce(operator.mul, self.loop_dims)
        self.acc_buf = [
            np.zeros((*self.loop_dims, nreads, 2), dtype=np.int64)
            for nreads in self.reads_per_shot
        ]
        self.stats = []

        # select which tqdm progress bar to show
        hiderounds = True
        hidereps = True
        if progress:
            if soft_avgs > 1:
                hiderounds = False
            else:
                hidereps = False

        # avg_d doesn't have a specific shape here, so that it's easier for child programs to write custom _average_buf
        sum_d = None
        sum2_d = None
        for ir in tqdm(range(soft_avgs), disable=hiderounds):
            # Configure and enable buffer capture.
            self.config_bufs(soc, enable_avg=True, enable_buf=False)

            # Reload data memory.
            soc.reload_mem()

            count = 0
            with tqdm(total=total_count, disable=hidereps) as pbar:
                soc.start_readout(
                    total_count,
                    counter_addr=self.counter_addr,
                    ch_list=list(self.ro_chs),
                    reads_per_shot=self.reads_per_shot,
                )
                while count < total_count and not self.early_stop:
                    new_data = obtain(soc.poll_data())
                    for new_points, (d, s) in new_data:
                        for ii, nreads in enumerate(self.reads_per_shot):
                            # print(count, new_points, nreads, d[ii].shape, total_count)
                            if new_points * nreads != d[ii].shape[0]:
                                logger.error(
                                    "data size mismatch: new_points=%d, nreads=%d, data shape %s"
                                    % (new_points, nreads, d[ii].shape)
                                )
                            if count + new_points > total_count:
                                logger.error(
                                    "got too much data: count=%d, new_points=%d, total_count=%d"
                                    % (count, new_points, total_count)
                                )
                            # use reshape to view the acc_buf array in a shape that matches the raw data
                            # self.acc_buf[ii].reshape((-1,2))[count*nreads:(count+new_points)*nreads] = d[ii]
                            c_start = count * nreads
                            c_end = (count + new_points) * nreads
                            buf1d = self.acc_buf[ii].reshape((-1, 2))
                            try:
                                buf1d[c_start:c_end] = d[ii]
                            except Exception as e:
                                print(e)
                                num = buf1d[c_start:c_end].shape[0]
                                buf1d[c_start:c_end] = d[ii][:num]
                        count += new_points
                        self.stats.append(s)
                        pbar.update(new_points)

            # if we're thresholding, apply the threshold before averaging
            if threshold is None:
                d_reps = self.acc_buf
                round_d = self._average_buf(
                    d_reps,
                    self.reads_per_shot,
                    length_norm=True,
                    remove_offset=remove_offset,
                )
                round_std = self._stderr_buf(d_reps, length_norm=True)
            else:
                d_reps = [np.zeros_like(d) for d in self.acc_buf]
                self.shots = self._apply_threshold(
                    self.acc_buf, threshold, angle, remove_offset=remove_offset
                )
                for i, ch_shot in enumerate(self.shots):
                    d_reps[i][..., 0] = ch_shot
                round_d = self._average_buf(
                    d_reps, self.reads_per_shot, length_norm=False
                )
                round_std = self._stderr_buf(d_reps, length_norm=False)

            # sum over rounds axis
            if sum_d is None:
                sum_d = round_d
            else:
                for ii, u in enumerate(round_d):
                    sum_d[ii] += u

            if sum2_d is None:
                sum2_d = [u**2 + o**2 for u, o in zip(round_d, round_std)]
            else:
                for ii, (u, o) in enumerate(zip(round_d, round_std)):
                    sum2_d[ii] += u**2 + o**2

            # early stop
            if self.early_stop:
                soft_avgs = ir + 1  # set to the current round
                break

            # callback
            if callback is not None:
                callback(ir, sum_d, sum2_d)

        # divide total by rounds
        avg_d = [s / soft_avgs for s in sum_d]
        std_d = [np.sqrt(s2 / soft_avgs - u**2) for s2, u in zip(sum2_d, avg_d)]

        return avg_d, std_d

    def _stderr_buf(
        self,
        d_reps: np.ndarray,
        length_norm: bool = True,
    ) -> np.ndarray:
        std_d = []
        for i_ch, (_, ro) in enumerate(self.ro_chs.items()):
            # average over the avg_level
            std = d_reps[i_ch].std(axis=self.avg_level)
            if length_norm and not ro["edge_counting"]:
                std /= ro["length"]
            # the reads_per_shot axis should be the first one
            std_d.append(np.moveaxis(std, -2, 0))

        return std_d

    def acquire_decimated(
        self,
        soc,
        soft_avgs=1,
        load_pulses=True,
        start_src="internal",
        progress=True,
        remove_offset=True,
        callback: Optional[AcquireDecimatedCallbackType] = None,
    ):
        self.early_stop = False

        # don't load memories now, we'll do that later
        self.config_all(soc, load_pulses=load_pulses, load_mem=False)

        if any(
            [x is None for x in [self.counter_addr, self.loop_dims, self.avg_level]]
        ):
            raise RuntimeError(
                "data dimensions need to be defined with setup_acquire() before calling acquire_decimated()"
            )

        # configure tproc for internal/external start
        soc.start_src(start_src)

        total_count = functools.reduce(operator.mul, self.loop_dims)

        # Initialize data buffers
        # buffer for decimated data
        dec_buf = []
        for ch, ro in self.ro_chs.items():
            maxlen = self.soccfg["readouts"][ch]["buf_maxlen"]
            if ro["length"] * ro["trigs"] * total_count > maxlen:
                raise RuntimeError(
                    "Warning: requested readout length (%d x %d trigs x %d reps) exceeds buffer size (%d)"
                    % (ro["length"], ro["trigs"], total_count, maxlen)
                )
            dec_buf.append(
                np.zeros((ro["length"] * total_count * ro["trigs"], 2), dtype=float)
            )

        # for each soft average, run and acquire decimated data
        for ir in tqdm(range(soft_avgs), disable=not progress):
            # buffer for accumulated data (for convenience/debug)
            self.acc_buf = []

            # Configure and enable buffer capture.
            self.config_bufs(soc, enable_avg=True, enable_buf=True)

            # Reload data memory.
            soc.reload_mem()

            # make sure count variable is reset to 0
            soc.set_tproc_counter(addr=self.counter_addr, val=0)

            # run the assembly program
            # if start_src="external", you must pulse the trigger input once for every round
            soc.start_tproc()

            count = 0
            while count < total_count:
                if self.early_stop:
                    return []  # directly return empty list if early stop

                count = soc.get_tproc_counter(addr=self.counter_addr)

            for ii, (ch, ro) in enumerate(self.ro_chs.items()):
                dec_buf[ii] += obtain(
                    soc.get_decimated(
                        ch=ch,
                        address=0,
                        length=ro["length"] * ro["trigs"] * total_count,
                    )
                )
                self.acc_buf.append(
                    obtain(
                        soc.get_accumulated(
                            ch=ch, address=0, length=ro["trigs"] * total_count
                        ).reshape((*self.loop_dims, ro["trigs"], 2))
                    )
                )
            # callback
            if callback is not None:
                callback(ir)

        onetrig = all([ro["trigs"] == 1 for ro in self.ro_chs.values()])

        # average the decimated data
        result = []
        for ii, (ch, ro) in enumerate(self.ro_chs.items()):
            d_avg = dec_buf[ii] / soft_avgs
            if remove_offset:
                d_avg -= self._ro_offset(ch, ro.get("ro_config"))
            if total_count == 1 and onetrig:
                # simple case: data is 1D (one rep and one shot), just average over rounds
                result.append(d_avg)
            else:
                # split the data into the individual reps
                if onetrig or total_count == 1:
                    d_reshaped = d_avg.reshape(total_count * ro["trigs"], -1, 2)
                else:
                    d_reshaped = d_avg.reshape(total_count, ro["trigs"], -1, 2)
                result.append(d_reshaped)

        return result
