"""Regression tests for datasaver.save_local_data / load_local_data / load_comment.

These tests also lock in the bug-fix from Item 1: load_comment previously used
a bare `from labber_io import ...` which would raise ImportError at runtime; the
relative import fix is exercised here via load_comment.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.utils.datasaver import load_comment, load_local_data, save_local_data

# ---------------------------------------------------------------------------
# 1-D save / load round-trip via dict API
# ---------------------------------------------------------------------------


def test_save_load_1d(tmp_path):
    """1-D round-trip: z shape (Nx,) and x values survive through datasaver."""
    freq = np.linspace(4e9, 5e9, 41)
    z = np.exp(1j * np.linspace(0, np.pi, 41)) * 0.5

    x_info = {"name": "Frequency", "unit": "Hz", "values": freq}
    z_info = {"name": "S21", "unit": "arb", "values": z}

    path = str(tmp_path / "ds_1d")
    save_local_data(path, x_info=x_info, z_info=z_info)

    # load_local_data returns (z, x, y) with frequency-major convention:
    # 1-D -> z is (Nx,), y is None.
    z_out, x_out, y_out = load_local_data(path)

    assert y_out is None
    assert np.allclose(x_out, freq)
    # z must match (same 1-D shape, no transposition needed for 1-D)
    assert np.allclose(z_out, z)


# ---------------------------------------------------------------------------
# 2-D save / load round-trip via dict API
# ---------------------------------------------------------------------------


def test_save_load_2d(tmp_path):
    """2-D round-trip: load_local_data returns frequency-major (Nx, Ny) order."""
    freq = np.linspace(4e9, 5e9, 21)  # x (inner, length Nx=21)
    power = np.linspace(-30.0, 0.0, 7)  # y (outer, length Ny=7)
    rng = np.random.default_rng(5)
    # datasaver expects z as (Ny, Nx) for save, but returns (Nx, Ny) on load.
    z_ny_nx = rng.standard_normal((len(power), len(freq))) + 1j * rng.standard_normal(
        (len(power), len(freq))
    )

    x_info = {"name": "Frequency", "unit": "Hz", "values": freq}
    y_info = {"name": "Power", "unit": "dBm", "values": power}
    z_info = {"name": "S21", "unit": "arb", "values": z_ny_nx}

    path = str(tmp_path / "ds_2d")
    save_local_data(path, x_info=x_info, z_info=z_info, y_info=y_info)

    z_out, x_out, y_out = load_local_data(path)

    assert np.allclose(x_out, freq)
    assert y_out is not None
    assert np.allclose(y_out, power)
    # load_local_data transposes to frequency-major (Nx, Ny)
    assert z_out.shape == (len(freq), len(power))
    assert np.allclose(z_out, z_ny_nx.T)


# ---------------------------------------------------------------------------
# load_comment round-trip (also locks in the relative-import bug fix)
# ---------------------------------------------------------------------------


def test_load_comment(tmp_path):
    """load_comment reads back the comment written by save_local_data."""
    freq = np.linspace(4e9, 5e9, 11)
    z = np.ones(11, dtype=complex)

    x_info = {"values": freq}
    z_info = {"values": z}

    path = str(tmp_path / "comment_test")
    save_local_data(path, x_info=x_info, z_info=z_info, comment="hello world")

    assert load_comment(path) == "hello world"


def test_load_comment_missing_file(tmp_path):
    """load_comment returns None and does not raise on a missing file."""
    result = load_comment(str(tmp_path / "nonexistent"))
    assert result is None
