from unittest.mock import patch

from zcu_tools.program.v2.mocksoc import (
    _DEFAULT_POLL_LATENCY,
    make_mock_soc,
    make_mock_soccfg,
)
from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM


def test_make_mock_soccfg():
    cfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    assert cfg is not None
    assert (
        getattr(cfg, "gens", None) is None
        or getattr(cfg, "gens") is not None
        or "gens" in cfg._cfg
    )
    assert len(cfg["gens"]) == 2
    assert "readouts" in cfg._cfg
    assert len(cfg["readouts"]) == 1


def test_mock_soc_methods():
    soc, _ = make_mock_soc(n_gens=1, n_readouts=1)

    # Test getting tproc counter returns huge number
    assert soc.get_tproc_counter(0) == soc._BIG_COUNT

    # Test readout start and poll
    soc.start_readout(total_shots=100, counter_addr=0, ch_list=[0], reads_per_shot=[1])
    assert soc._readout_state == (100, [0], [1])
    assert soc._poll_done is False

    # Test poll_data with fake data
    data = soc.poll_data()
    assert len(data) == 1
    total_shots, (arr_list, _) = data[0]
    assert total_shots == 100
    assert len(arr_list) == 1
    assert arr_list[0].shape == (100, 2)
    assert soc._poll_done is True

    # Second poll should return empty
    assert soc.poll_data() == []


class TestPollLatency:
    """poll_latency controls whether time.sleep is called in poll_data."""

    def test_default_white_noise_sleep_called(self) -> None:
        # White-noise path (no SimParams) uses _DEFAULT_POLL_LATENCY; sleep is called.
        soc, _ = make_mock_soc(n_gens=1, n_readouts=1)
        soc.start_readout(
            total_shots=10, counter_addr=0, ch_list=[0], reads_per_shot=[1]
        )
        with patch("zcu_tools.program.v2.mocksoc.time.sleep") as mock_sleep:
            soc.poll_data()
        mock_sleep.assert_called_once()
        args, _ = mock_sleep.call_args
        # sleep argument should be proportional to _DEFAULT_POLL_LATENCY and data size
        assert args[0] > 0.0

    def test_poll_latency_zero_no_sleep(self) -> None:
        # poll_latency=0.0 must not call time.sleep at all (avoid sleep(0) syscall noise).
        sim = DEFAULT_SIMPARAM.model_copy(update={"poll_latency": 0.0})
        soc, _ = make_mock_soc(n_gens=1, n_readouts=1, sim=sim)
        soc.start_readout(
            total_shots=10, counter_addr=0, ch_list=[0], reads_per_shot=[1]
        )
        with patch("zcu_tools.program.v2.mocksoc.time.sleep") as mock_sleep:
            soc.poll_data()
        mock_sleep.assert_not_called()

    def test_poll_latency_default_sim_sleep_called(self) -> None:
        # Sim path with the default poll_latency (1e-7) must call time.sleep.
        sim = DEFAULT_SIMPARAM.model_copy(update={"poll_latency": 1e-7})
        soc, _ = make_mock_soc(n_gens=1, n_readouts=1, sim=sim)
        soc.start_readout(
            total_shots=10, counter_addr=0, ch_list=[0], reads_per_shot=[1]
        )
        with patch("zcu_tools.program.v2.mocksoc.time.sleep") as mock_sleep:
            soc.poll_data()
        mock_sleep.assert_called_once()

    def test_default_poll_latency_constant(self) -> None:
        # Sanity-check that the module-level constant matches the expected default.
        assert _DEFAULT_POLL_LATENCY == 1e-7


def test_mock_soc_decimated_accumulated():
    soc, _ = make_mock_soc()

    dec = soc.get_decimated(0, 0, 50)
    assert dec.shape == (50, 2)

    acc = soc.get_accumulated(0, 0, 20)
    assert acc.shape == (20, 2)
