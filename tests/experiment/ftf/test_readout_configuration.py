"""Detached readout values and the public settings-to-author adapter contract."""

from dataclasses import replace

import pytest
from zcu_tools.experiment import ftf as ftf_module
from zcu_tools.experiment.ftf.readout import readout_configuration
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import dual_readout_document, ftf_document


def test_readout_configuration_preserves_pulse_without_mux() -> None:
    config = readout_configuration(parse_ftf(ftf_document()))
    expected = ftf_module.FTFReceiverConfig(
        ro_ch=0, gain=10000.0, noise_psd_adc2_ns=((0.0, 0.0), (0.0, 0.0))
    )
    assert getattr(config, "pulse", None) == expected
    assert getattr(config, "mux", "missing") is None


def test_readout_configuration_preserves_both_paths_and_all_calibrations() -> None:
    try:
        settings = parse_ftf(dual_readout_document())
    except ValueError as error:
        pytest.fail(f"dual-readout settings must adapt to detached values: {error}")
    config = readout_configuration(settings)
    assert config.pulse.ro_ch == 0
    mux = config.mux
    assert mux is not None
    assert mux.gen_ch == 1
    assert mux.input_per_gain == -2.0
    assert mux.receivers == (
        ftf_module.FTFReceiverConfig(
            ro_ch=4,
            gain=1000.0,
            phase_deg=10.0,
            offset_adc=(0.1, -0.2),
            noise_psd_adc2_ns=((1.0, 0.0), (0.0, 2.0)),
            time_of_flight_ns=20.0,
        ),
        ftf_module.FTFReceiverConfig(
            ro_ch=1,
            gain=1001.0,
            phase_deg=11.0,
            offset_adc=(1.1, -0.2),
            noise_psd_adc2_ns=((2.0, 0.0), (0.0, 2.0)),
            time_of_flight_ns=21.0,
        ),
    )
    assert config.pulse == readout_configuration(parse_ftf(ftf_document())).pulse


def mux_configuration(channels: tuple[int, ...] = (4, 1)):
    config_type = getattr(ftf_module, "FTFMuxReadoutConfig", None)
    assert config_type is not None, "FTF must expose the detached mux capability"
    return config_type(
        gen_ch=100,
        input_per_gain=-2.0,
        receivers=tuple(
            ftf_module.FTFReceiverConfig(
                ro_ch=ch, gain=1.0, noise_psd_adc2_ns=((0.0, 0.0), (0.0, 0.0))
            )
            for ch in channels
        ),
    )


@pytest.mark.parametrize("channels", [(1,), tuple(range(1, 9)), (100, 200)])
def test_detached_mux_accepts_one_to_eight_receivers_and_high_indices(
    channels: tuple[int, ...],
) -> None:
    mux = mux_configuration(channels)
    assert tuple(r.ro_ch for r in mux.receivers) == channels
    assert mux.gen_ch == 100
    assert mux.input_per_gain == -2.0


@pytest.mark.parametrize("channels", [(), tuple(range(1, 10)), (1, 1)])
def test_detached_mux_rejects_invalid_receiver_count_or_duplicates(
    channels: tuple[int, ...],
) -> None:
    valid = mux_configuration()
    assert len(valid.receivers) == 2
    with pytest.raises(ValueError, match="receiver"):
        mux_configuration(channels)


@pytest.mark.parametrize("gen_ch", [-1, 1.5, True])
def test_detached_mux_rejects_invalid_generator(gen_ch: object) -> None:
    mux = mux_configuration()
    with pytest.raises(ValueError, match="gen_ch"):
        replace(mux, gen_ch=gen_ch)


@pytest.mark.parametrize("input_per_gain", [float("nan"), float("inf")])
def test_detached_mux_rejects_nonfinite_calibration(input_per_gain: float) -> None:
    mux = mux_configuration()
    with pytest.raises(ValueError, match="input_per_gain"):
        replace(mux, input_per_gain=input_per_gain)


def test_detached_readout_rejects_receiver_shared_by_pulse_and_mux() -> None:
    mux = mux_configuration()
    pulse = ftf_module.FTFReceiverConfig(
        ro_ch=4, gain=1.0, noise_psd_adc2_ns=((0.0, 0.0), (0.0, 0.0))
    )
    with pytest.raises(ValueError, match="duplicate.*receiver|receiver.*duplicate"):
        ftf_module.FTFReadoutConfig(pulse=pulse, mux=mux)
