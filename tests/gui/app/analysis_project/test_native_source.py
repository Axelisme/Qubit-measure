"""The shared source reader keeps declared native content and metadata together."""

from dataclasses import replace

import pytest
from zcu_tools.datafile import Container, load, save
from zcu_tools.experiment.native_persistence import metadata_from_container
from zcu_tools.gui.app.analysis_project.native import read_spectrum_source

from tests.gui.app.analysis_project._native import native_spectrum_file


def test_native_read_keeps_acquisition_metadata(tmp_path):
    path = native_spectrum_file(tmp_path / "data.h5", {"coil": {"value": 0.2}})
    source = read_spectrum_source(str(path))
    assert isinstance(source, Container)
    assert metadata_from_container(source).device_snapshot == {"coil": {"value": 0.2}}


def test_declared_native_metadata_is_not_hidden_by_a_labber_fallback(tmp_path):
    path = native_spectrum_file(tmp_path / "invalid.h5", None)
    source = load(path)
    save(path, replace(source, docs={"metadata": {"run_id": "invalid"}}), replace=True)
    decoded = read_spectrum_source(str(path))
    assert isinstance(decoded, Container)
    with pytest.raises(ValueError, match="/docs/metadata"):
        metadata_from_container(decoded)
