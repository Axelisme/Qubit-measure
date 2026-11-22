"""Corrected marginal IQ moments through the compiled acquisition seam."""

from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from qick import QickConfig
from zcu_tools.analysis.shot_uncertainty import iq_shot_sem
from zcu_tools.program import v2
from zcu_tools.program.acquisition import CancelFlagProtocol
from zcu_tools.program.v2.modules import MuxReadout

from tests.program.v2._mux_support import AcquisitionSoc, mux_soccfg, native_mux


def _mux_in_participant_order() -> tuple[QickConfig, MuxReadout]:
    soccfg = mux_soccfg()
    soccfg["readouts"][4]["iq_offset"] = 4.0
    adc = soccfg["readouts"][1]
    adc["iq_offset"] = 7.0
    adc["f_output"] /= 2
    cfg = native_mux((1, 4), mixer_freq=0.0)
    cfg.tones[0].freq = adc["f_dds"] * 9 / 8  # away from center: subtract 7
    cfg.tones[1].freq = 0.0  # channel center: subtract 2 × 4
    participants = [1, 0]
    return soccfg, cfg.with_updates(tones=[cfg.tones[i] for i in participants]).build(
        "mux"
    )


class _LiteralSoc(AcquisitionSoc):
    """Deliver literal IQ integrals, then interrupt a partially streamed round."""

    def __init__(self, cfg: QickConfig, stop: Event) -> None:
        super().__init__(cfg.get_cfg())
        self.stop = stop
        self.started = 0

    def start_readout(self, total_shots, counter_addr, ch_list, reads_per_shot):
        super().start_readout(total_shots, counter_addr, ch_list, reads_per_shot)
        self.started += 1

    def poll_data(self, totaltime=0.1, timeout=None):
        if self.delivered:
            self.stop.set()
            return []
        self.delivered = True
        shots = (
            {4: [[9, 10], [11, 6]], 1: [[5, 11], [11, 7]]},
            {4: [[13, 9], [7, 12]], 1: [[9, 5], [7, 10]]},
            {4: [[99, 99]], 1: [[99, 99]]},
        )[self.started - 1]
        integrals = [
            np.array(shots[ch], dtype=np.int64) * self.us2cycles(0.413, ro_ch=ch)
            for ch in self.channels
        ]
        return [(len(shots[4]), (integrals, {}))]


def test_marginals_keep_participant_order_and_only_completed_rounds() -> None:
    soccfg, module = _mux_in_participant_order()
    builder = v2.TimelineBuilder(soccfg, reps=2)
    builder.emit(module, tag="terminal")
    program = builder.build()
    stop = Event()
    soc = _LiteralSoc(soccfg, stop)
    published: list[int] = []

    def on_round(
        completed: int, data: NDArray[np.float64], _stop: CancelFlagProtocol
    ) -> None:
        published.append(completed)
        if completed == 1:
            np.testing.assert_array_equal(
                data, [[2, 4, 0, 10, -4, 8], [2, 2, 4, 20, -8, 16]]
            )

    result = program.acquire_summed(
        soc,
        reducer=v2.mux_iq_sums(program, "terminal"),
        rounds=4,
        cancel_flag=stop,
        round_hook=on_round,
        progress=False,
    )

    assert result is not None
    assert stop.is_set()
    assert soc.started == 3
    assert published == [1, 2]
    np.testing.assert_array_equal(
        result, [[4, 8, 5, 36, -3, 25], [4, 4, 5, 24, -12, 29]]
    )
    counts, means = v2.split_sums(result)
    np.testing.assert_array_equal(counts, [4, 4])
    np.testing.assert_array_equal(means, [2 + 1.25j, 1 + 1.25j])
    # For the two tones, the same calibrated I+Q shots are [3,1,6,3] and [2,4,0,3].
    np.testing.assert_allclose(
        iq_shot_sem(result, direction=(1.0, 1.0)), np.sqrt([17 / 16, 35 / 48])
    )


@pytest.mark.parametrize(
    ("unsupported", "message"),
    [
        ("mask", "shot masks"),
        ("extra read", "one integrated read"),
        ("edge", "edge counting"),
    ],
)
def test_marginals_reject_unsupported_acquisition(
    unsupported: str, message: str
) -> None:
    soccfg, module = _mux_in_participant_order()
    builder = v2.TimelineBuilder(soccfg, reps=2)
    builder.emit(module, tag="terminal")
    if unsupported in {"mask", "extra read"}:
        builder.timeline_inc(2.0)
        builder.emit(module, tag="second")
    if unsupported == "mask":
        builder.register_shot_mask_source({4: [0]}, lambda iq: iq[4][..., 0, 0] > 0)
    program = builder.build()
    if unsupported == "edge":
        # MuxReadout exposes no edge-counting author option; exercise native metadata.
        program.ro_chs[1]["edge_counting"] = True
    with pytest.raises(ValueError, match=message):
        v2.mux_iq_sums(program, "terminal")
