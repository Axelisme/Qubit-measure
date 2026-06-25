from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from zcu_tools.experiment.v2_gui.adapters.twotone import FluxDepAdapter
from zcu_tools.gui.app.main.adapter import LoadDataRequest
from zcu_tools.utils.datasaver import save_labber_data


def test_adapter_load_migrates_supported_legacy_single_file(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy_twotone_flux.hdf5"
    freqs = np.array([4300.0, 4350.0, 4400.0], dtype=np.float64)
    values = np.array([-0.25, 0.45], dtype=np.float64)
    signals = np.array(
        [[1.0 + 0.1j, 1.1 + 0.2j, 1.2 + 0.3j], [2.0 + 0.4j, 2.1 + 0.5j, 2.2 + 0.6j]],
        dtype=np.complex128,
    )
    save_labber_data(
        str(legacy_path),
        z=("Signal", "ADC unit", signals),
        axes=[
            ("Frequency", "Hz", freqs * 1e6),
            ("Yoko", "A", values),
        ],
        comment="legacy flux comment",
        tags=["TwoTone"],
    )

    loaded = FluxDepAdapter().load(
        LoadDataRequest(data_path=str(legacy_path), md=MagicMock(), ml=MagicMock())
    )

    np.testing.assert_allclose(loaded.freqs, freqs, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.values, values, rtol=0, atol=0)
    np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)
