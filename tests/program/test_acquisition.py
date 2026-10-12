"""Public acquisition classifies normalized IQ without depending on lab cores."""

import numpy as np
import pytest
from zcu_tools.analysis.singleshot import classify_result
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("remove_offset", [False, True])
def test_acquire_population_matches_offline_classification(
    monkeypatch: pytest.MonkeyPatch, radius: float, remove_offset: bool
) -> None:
    signals = np.array([-1, 1, 0, 0.2, -0.2, 4, -1 + radius], dtype=np.complex128)
    expected = classify_result(signals, -1, 1, radius)
    assert np.all(sum(mask.astype(int) for mask in expected) == 1)
    assert expected[2][2]  # An exact tie belongs to neither population.
    soc, soccfg = make_mock_soc()
    offset = 0.25 if remove_offset else 0.0
    soccfg["readouts"][0]["iq_offset"] = offset
    program = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=1),
        modules=[
            DirectReadoutCfg(
                ro_ch=0,
                ro_length=soccfg.cycles2us(100, ro_ch=0),
                ro_freq=100.0,
            ).build("readout")
        ],
        sweep=[("sample", len(signals))],
    )
    raw = np.rint(
        (np.stack((signals.real, signals.imag), axis=-1) + offset)
        * program.ro_chs[0]["length"]
    ).astype(np.int64)
    delivered = False

    def poll_data(totaltime: float = 0.1, timeout: float | None = None):
        nonlocal delivered
        if delivered:
            return []
        delivered = True
        return [(len(signals), ([raw], {}))]

    monkeypatch.setattr(soc, "poll_data", poll_data)
    actual = program.acquire(
        soc,
        g_center=-1,
        e_center=1,
        ge_radius=radius,
        remove_offset=remove_offset,
        progress=False,
    )
    np.testing.assert_array_equal(actual[0][0], np.stack(expected[:2], axis=-1))
    assert actual[0].dtype == np.float64
