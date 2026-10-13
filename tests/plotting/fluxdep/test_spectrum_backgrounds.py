"""Native spectrum and line-pick figures draw OneTone like its sample picker."""

import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs, FluxPickState
from zcu_tools.analysis.fluxdep.models import PointsData
from zcu_tools.analysis.fluxdep.onetone import OneToneInputs
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.plotting.fluxdep.onetone import OneTonePickPlot
from zcu_tools.plotting.fluxdep.pick import make_flux_pick_figure
from zcu_tools.plotting.fluxdep.spectrum import make_spectrum_figure


@pytest.fixture
def raw() -> SpectrumData:
    """A OneTone resonator whose dip moves with the device value."""
    devs = np.linspace(0.0, 1.0, 30)
    freqs = np.linspace(5.0, 6.0, 25)
    center = 5.5 + 0.2 * np.sin(2 * np.pi * devs)
    dip = 0.6 * np.exp(-(((freqs[None, :] - center[:, None]) / 0.05) ** 2))
    return SpectrumData(
        dev_values=devs,
        fluxs=devs.copy(),
        freqs=freqs,
        signals=np.asarray(1.0 - dip, dtype=np.complex128),
    )


def _sample_picker_background(raw: SpectrumData) -> tuple[np.ndarray, tuple]:
    """The (frequency, device) background and limits of the OneTone sample picker."""
    inputs = OneToneInputs(
        FluxPickInputs(raw["signals"], raw["dev_values"], raw["freqs"])
    )
    figure = Figure()
    OneTonePickPlot(figure, inputs)
    image = figure.axes[0].get_images()[0]
    return np.asarray(image.get_array()), image.get_clim()


def _empty_points() -> PointsData:
    return PointsData(dev_values=np.empty(0), fluxs=np.empty(0), freqs=np.empty(0))


def _mesh(figure: Figure) -> QuadMesh:
    meshes = [c for c in figure.axes[0].collections if isinstance(c, QuadMesh)]
    assert len(meshes) == 1
    return meshes[0]


def test_onetone_spectrum_png_uses_the_sample_picker_background(raw):
    expected, expected_clim = _sample_picker_background(raw)
    mesh = _mesh(
        make_spectrum_figure(raw, _empty_points(), alignment=None, onetone=True)
    )
    np.testing.assert_allclose(
        np.asarray(mesh.get_array()).reshape(expected.shape), expected
    )
    assert mesh.get_clim() == pytest.approx(expected_clim)


def test_twotone_spectrum_png_keeps_the_raw_magnitude(raw):
    mesh = _mesh(
        make_spectrum_figure(raw, _empty_points(), alignment=None, onetone=False)
    )
    expected = np.abs(raw["signals"]).T
    np.testing.assert_allclose(
        np.asarray(mesh.get_array()).reshape(expected.shape), expected
    )


def test_onetone_line_pick_png_uses_the_sample_picker_background(raw):
    expected, expected_clim = _sample_picker_background(raw)
    inputs = FluxPickInputs(raw["signals"], raw["dev_values"], raw["freqs"])
    state = FluxPickState(flux_half=0.25, flux_int=0.75, magnitude_only=True)
    image = make_flux_pick_figure(inputs, state, onetone=True).axes[0].get_images()[0]
    np.testing.assert_allclose(np.asarray(image.get_array()), expected)
    assert image.get_clim() == pytest.approx(expected_clim)
