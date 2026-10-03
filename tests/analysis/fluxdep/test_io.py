from __future__ import annotations

import h5py
import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.io import dump_spectrums, load_spectrums
from zcu_tools.analysis.fluxdep.models import SpectrumResult


@pytest.mark.parametrize("include_type", [False, True])
def test_spectrum_collection_roundtrips_values_shapes_and_dtypes(
    tmp_path, include_type
):
    first = SpectrumResult(
        flux_half=0.25,
        flux_int=0.75,
        flux_period=1.0,
        spectrum={
            "dev_values": np.array([0.0, 1.0]),
            "fluxs": np.array([0.25, 0.75]),
            "freqs": np.array([4.0, 5.0, 6.0]),
            "signals": np.array([[1 + 2j, 3 - 4j, 5j], [6j, 7 + 8j, 9 - 1j]]),
        },
        points={
            "dev_values": np.array([], dtype=np.float64),
            "fluxs": np.array([], dtype=np.float64),
            "freqs": np.array([], dtype=np.float64),
        },
    )
    second = SpectrumResult(
        flux_half=-0.5,
        flux_int=0.5,
        flux_period=2.0,
        spectrum=first["spectrum"],
        points={
            "dev_values": np.array([0.0, 1.0]),
            "fluxs": np.array([0.25, 0.75]),
            "freqs": np.array([4.0, 6.0]),
        },
    )
    if include_type:
        first["type"] = "OneTone"
        second["type"] = "TwoTone"
    expected = {"first": first, "second": second}
    path = str(tmp_path / "spectrums.hdf5")
    dump_spectrums(path, expected)
    actual = load_spectrums(path)
    assert actual.keys() == expected.keys()
    for name, source in expected.items():
        result = actual[name]
        assert result["flux_half"] == source["flux_half"]
        assert result["flux_int"] == source["flux_int"]
        assert result["flux_period"] == source["flux_period"]
        assert result.get("type") == source.get("type")
        pairs = (
            (result["spectrum"]["dev_values"], source["spectrum"]["dev_values"]),
            (result["spectrum"]["fluxs"], source["spectrum"]["fluxs"]),
            (result["spectrum"]["freqs"], source["spectrum"]["freqs"]),
            (result["spectrum"]["signals"], source["spectrum"]["signals"]),
            (result["points"]["dev_values"], source["points"]["dev_values"]),
            (result["points"]["fluxs"], source["points"]["fluxs"]),
            (result["points"]["freqs"], source["points"]["freqs"]),
        )
        for restored, array in pairs:
            np.testing.assert_array_equal(restored, array)
            assert restored.shape == array.shape
            assert restored.dtype == array.dtype


def test_load_spectrums_preserves_non_group_rejection(tmp_path):
    path = tmp_path / "raw.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("raw", data=np.array([1.0]))
    with pytest.raises(AssertionError):
        load_spectrums(str(path))


def test_load_spectrums_preserves_missing_group_error(tmp_path):
    path = tmp_path / "incomplete.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_group("entry")
    with pytest.raises(KeyError):
        load_spectrums(str(path))
