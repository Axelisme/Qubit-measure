from unittest.mock import patch

from zcu_tools.program.v2.mocksoc import (
    make_mock_soc,
    make_mock_soccfg,
)
from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM


def test_make_mock_soccfg():
    cfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    assert cfg is not None
    assert len(cfg["gens"]) == 2
    assert len(cfg["readouts"]) == 1


def test_mock_soc_methods():
    soc, _ = make_mock_soc(n_gens=1, n_readouts=1)

    # Test getting tproc counter returns huge number
    assert soc.get_tproc_counter(0) >= 100

    # Test readout start and poll
    soc.start_readout(total_shots=100, counter_addr=0, ch_list=[0], reads_per_shot=[1])

    # Test poll_data with fake data
    data = soc.poll_data()
    assert len(data) == 1
    total_shots, (arr_list, _) = data[0]
    assert total_shots == 100
    assert len(arr_list) == 1
    assert arr_list[0].shape == (100, 2)

    # Second poll should return empty
    assert soc.poll_data() == []


class TestPollLatency:
    """poll_latency controls whether time.sleep is called in poll_data."""

    def test_default_white_noise_sleep_called(self) -> None:
        # White-noise path (no SimParams) uses the default pacing; sleep is called.
        soc, _ = make_mock_soc(n_gens=1, n_readouts=1)
        soc.start_readout(
            total_shots=10, counter_addr=0, ch_list=[0], reads_per_shot=[1]
        )
        with patch("zcu_tools.program.v2.mocksoc.time.sleep") as mock_sleep:
            soc.poll_data()
        mock_sleep.assert_called_once()
        args, _ = mock_sleep.call_args
        # sleep argument should be proportional to the default pacing and data size
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


def test_mock_soc_decimated_accumulated():
    soc, _ = make_mock_soc()

    dec = soc.get_decimated(0, 0, 50)
    assert dec.shape == (50, 2)

    acc = soc.get_accumulated(0, 0, 20)
    assert acc.shape == (20, 2)
