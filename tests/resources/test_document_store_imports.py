"""Typed YAML edits and shared path helpers do not require HDF5."""

import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.requires_subprocess


def test_document_storage_and_path_helpers_without_hdf5(tmp_path: Path) -> None:
    lane = Path(__file__).resolve().parents[2]
    source = dedent(
        """
        import importlib.abc
        import sys
        from pathlib import Path

        class UnavailableDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "h5py" or fullname.startswith("h5py."):
                    raise ImportError("unrelated dependency requested: " + fullname)

        sys.meta_path.insert(0, UnavailableDependency())

        from pydantic import BaseModel
        from zcu_tools.datafile import reserve_labber_filepath
        from zcu_tools.resources.document_store import DocumentStore

        class Parameters(BaseModel):
            format: str
            format_version: str
            t1: float

        directory = Path(sys.argv[1])
        reserved = reserve_labber_filepath(str(directory / "measurement"))
        assert Path(reserved) == directory / "measurement.hdf5"

        path = directory / "parameters.yaml"
        path.write_text("format: synthetic\\nformat_version: '1.0'\\nt1: 2.5\\n", encoding="utf-8")
        store = DocumentStore(path, Parameters, format="synthetic")
        assert store.snapshot.t1 == 2.5
        with store.edit() as draft:
            draft.t1 = 4.5
        reopened = DocumentStore(path, Parameters, format="synthetic")
        assert reopened.snapshot.t1 == 4.5
        """
    )
    result = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(lane),
            "--no-sync",
            "--",
            "python",
            "-c",
            source,
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
