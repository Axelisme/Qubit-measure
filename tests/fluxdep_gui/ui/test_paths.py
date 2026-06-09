"""Tests for the file-dialog default directories (pure path logic, no Qt)."""

from __future__ import annotations

import os

import zcu_tools.gui.app.fluxdep.ui.paths as paths
from zcu_tools.gui.project import ProjectInfo


def test_processed_dir_derives_from_chip_qub_when_result_dir_unset():
    # no result_dir → derive result/<chip>/<qubit>/data/fluxdep, then fall back
    # to the nearest existing ancestor (so it is NOT the root / cwd-as-empty).
    p = ProjectInfo(chip_name="ZZ_chip", qub_name="ZZ_qub")  # not on disk
    d = paths.processed_spectrum_dir(p)
    # the intended path includes the chip/qub layout
    intended = os.path.join("result", "ZZ_chip", "ZZ_qub", "data", "fluxdep")
    assert os.path.abspath(intended).startswith(d) or d == ""
    # and d is an existing directory (or "") — never a non-existent path
    assert d == "" or os.path.isdir(d)


def test_processed_dir_uses_explicit_result_dir(tmp_path):
    target = tmp_path / "data" / "fluxdep"
    target.mkdir(parents=True)
    p = ProjectInfo(result_dir=str(tmp_path))
    assert paths.processed_spectrum_dir(p) == str(target)


def test_nearest_existing_falls_back_to_ancestor(tmp_path):
    # target deep under tmp_path that doesn't exist → nearest existing ancestor
    p = ProjectInfo(result_dir=str(tmp_path / "nope"))
    d = paths.params_dir(p)
    assert d == str(tmp_path)  # the existing ancestor


def test_database_dir_falls_back_to_bundled_when_unset():
    p = ProjectInfo()  # no database_path
    d = paths.database_dir(p)
    # either the bundled dir exists (→ returned) or "" when absent
    assert (
        d == ""
        or d.endswith(os.path.join("Database", "simulation"))
        or os.path.isdir(d)
    )


def test_database_dir_anchors_bundled_at_injected_root(tmp_path):
    # The bundled search database is repo-relative; the injected repo root anchors
    # it there instead of relative to cwd — the .bat-launcher fix (cwd is script/).
    sim = tmp_path / "Database" / "simulation"
    sim.mkdir(parents=True)
    p = ProjectInfo()
    assert paths.database_dir(p, str(tmp_path)) == str(sim)


def test_default_database_file_anchors_bundled_at_injected_root(tmp_path):
    sim = tmp_path / "Database" / "simulation"
    sim.mkdir(parents=True)
    (sim / "fluxonium_db.h5").touch()
    p = ProjectInfo()
    assert paths.default_database_file(p, str(tmp_path)) == str(sim / "fluxonium_db.h5")


def test_existing_dirs_are_returned_as_is(tmp_path):
    p = ProjectInfo(database_path=str(tmp_path))
    assert paths.raw_spectrum_dir(p) == str(tmp_path)
