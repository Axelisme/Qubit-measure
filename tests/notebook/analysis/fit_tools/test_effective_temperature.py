from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.notebook.analysis.fit_tools import (
    ThermalAttenuatorStage,
    blackbody_photon_number,
    calculate_thermal_chain,
    effective_temperature_from_photon_number,
    evaluate_thermal_chain_at_frequency,
    plot_effective_temperature_vs_frequency,
    plot_thermal_chain_psd,
    thermal_psd_log10_v2_per_hz,
)


def test_same_temperature_attenuator_chain_preserves_thermal_equilibrium() -> None:
    frequencies = np.geomspace(1e6, 10e9, 50)
    stages = (
        ThermalAttenuatorStage("stage-1", Temp_K=300.0, attenuation_db=3.0),
        ThermalAttenuatorStage("stage-2", Temp_K=300.0, attenuation_db=17.0),
        ThermalAttenuatorStage("stage-3", Temp_K=300.0, attenuation_db=0.0),
    )

    result = calculate_thermal_chain(
        frequencies,
        stages,
        input_temperature_K=300.0,
    )

    assert result.effective_temperature_K == pytest.approx(
        np.full_like(frequencies, 300.0)
    )
    assert result.effective_photon_number == pytest.approx(
        blackbody_photon_number(300.0, frequencies)
    )


def test_thermal_chain_uses_output_referred_stage_emission() -> None:
    frequency = 5.0e9
    stages = (
        ThermalAttenuatorStage("4K", Temp_K=4.0, attenuation_db=20.0),
        ThermalAttenuatorStage("20mK", Temp_K=0.02, attenuation_db=10.0),
    )

    result = calculate_thermal_chain(
        [frequency],
        stages,
        input_temperature_K=300.0,
    )

    L_4K = 10.0 ** (20.0 / 10.0)
    L_20mK = 10.0 ** (10.0 / 10.0)
    expected_photons = (
        blackbody_photon_number(300.0, [frequency])[0] / (L_4K * L_20mK)
        + (1.0 - 1.0 / L_4K) * blackbody_photon_number(4.0, [frequency])[0] / L_20mK
        + (1.0 - 1.0 / L_20mK) * blackbody_photon_number(0.02, [frequency])[0]
    )

    assert result.effective_photon_number[0] == pytest.approx(expected_photons)
    assert result.effective_temperature_K[0] == pytest.approx(
        effective_temperature_from_photon_number([expected_photons], [frequency])[0]
    )


def test_evaluate_thermal_chain_at_frequency_matches_curve_calculation() -> None:
    frequency = 5.79313e9
    stages = (
        ThermalAttenuatorStage("50K", Temp_K=50.0, attenuation_db=13.0),
        ThermalAttenuatorStage("4K", Temp_K=4.0, attenuation_db=13.0),
        ThermalAttenuatorStage("100mK", Temp_K=0.1, attenuation_db=13.0),
        ThermalAttenuatorStage("20mK", Temp_K=0.02, attenuation_db=23.0),
    )

    curve = calculate_thermal_chain([frequency], stages)
    probe = evaluate_thermal_chain_at_frequency(frequency, stages)

    assert probe.effective_temperature_K == pytest.approx(
        curve.effective_temperature_K[0]
    )
    assert probe.photon_number == pytest.approx(curve.effective_photon_number[0])
    assert probe.log10_psd_v2_per_hz == pytest.approx(
        curve.total_log10_psd_v2_per_hz[0]
    )


def test_effective_temperature_validates_chain_inputs() -> None:
    with pytest.raises(ValueError, match="attenuation_db"):
        calculate_thermal_chain(
            [5e9],
            [ThermalAttenuatorStage("bad", Temp_K=4.0, attenuation_db=-1.0)],
        )

    with pytest.raises(ValueError, match="frequencies_hz"):
        calculate_thermal_chain(
            [0.0],
            [ThermalAttenuatorStage("4K", Temp_K=4.0, attenuation_db=20.0)],
        )


def test_effective_temperature_plots_include_probe_and_layers() -> None:
    result = calculate_thermal_chain(
        np.geomspace(1e6, 10e9, 20),
        (
            ThermalAttenuatorStage("4K", Temp_K=4.0, attenuation_db=20.0),
            ThermalAttenuatorStage("20mK", Temp_K=0.02, attenuation_db=20.0),
        ),
    )

    fig_psd, ax_psd = plot_thermal_chain_psd(result, probe_frequency_hz=5e9)
    psd_labels = [str(artist.get_label()) for artist in ax_psd.get_lines()]
    assert "300K" in psd_labels
    assert "Effective" in psd_labels
    assert any(label.startswith("T_eff =") for label in psd_labels)
    assert len(ax_psd.collections) == 1
    raw_300K = thermal_psd_log10_v2_per_hz(300.0, result.frequencies_hz)
    assert ax_psd.get_lines()[0].get_ydata() == pytest.approx(raw_300K)

    fig_temp, ax_temp = plot_effective_temperature_vs_frequency(
        result,
        probe_frequency_hz=5e9,
        highlight_range_hz=(1e9, 6e9),
    )
    temp_labels = [str(artist.get_label()) for artist in ax_temp.get_lines()]
    assert "effective temperature" in temp_labels
    assert any(label.startswith("range min =") for label in temp_labels)
    assert len(ax_temp.collections) == 1

    fig_psd.clear()
    fig_temp.clear()
