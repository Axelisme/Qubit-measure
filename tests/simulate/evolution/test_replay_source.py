"""Replay keeps the original numeric source despite later exact publications."""

from hashlib import sha256

import numpy as np
from scipy.linalg import expm
from zcu_tools.simulate.evolution import (
    Basis,
    ControlPulse,
    ControlSequence,
    CPUReference,
    DensityState,
    DriveSource,
    Envelope,
    Evolution,
    EvolutionLibrary,
    LibrarySnapshot,
    NumericModel,
    PlateauFamily,
)
from zcu_tools.simulate.evolution.archive import (
    ArchiveHeader,
    NumericArchive,
    encode_archive,
)


def test_table_replay_ignores_later_exact_record(tmp_path):
    basis = Basis("replay", ("g", "e"), np.eye(2, dtype=complex))
    model = NumericModel(
        "replay",
        basis,
        np.diag([0.0, 0.017]).astype(complex),
        (DriveSource("a", np.array([[0, 0.003], [0.003, 0]], dtype=complex)),),
    )
    family = PlateauFamily(model, 8.0, (1 + 0j,))
    nodes = np.linspace(-0.1, 0.1, 5)
    # A tiny synthetic CPU table exercises runtime replay, not GPU construction.
    values = CPUReference().compute(family, nodes).propagators
    payload = encode_archive(
        NumericArchive(
            ArchiveHeader(
                kind="table",
                key="synthetic",
                family=family.identity,
                lower=-0.1,
                upper=0.1,
                degree=4,
                duration_ns=8.0,
                ratios=((1.0, 0.0),),
                evidence=(("synthetic_fixture", 0.0),),
            ),
            nodes,
            values,
            np.ones(5, dtype=bool),
        )
    )
    folder = tmp_path / "tables"
    folder.mkdir()
    (folder / f"{sha256(payload).hexdigest()}.npz").write_bytes(payload)
    library = EvolutionLibrary(tmp_path)
    snapshot = library.snapshot()
    controls = ControlSequence((ControlPulse("a", 0, 8, Envelope.flat(0.037)),))
    initial = DensityState(basis, np.full((2, 2), 0.5, dtype=complex))
    original = Evolution(model, library=library, tables=snapshot)
    expected = original.evolve(initial, controls, end_ns=8)
    assert original.last_evaluation.origin == "table"
    manifest = library.replay_manifest(
        snapshot, original.last_evaluation.content_identities
    )

    direct = Evolution(model, library=library, tables=LibrarySnapshot(()))
    exact = direct.evolve(initial, controls, end_ns=8)
    assert direct.last_evaluation.origin == "direct"
    u = expm(-2j * np.pi * 8 * (model.hamiltonian_ghz + 0.037 * family.drive_ghz))
    np.testing.assert_allclose(
        exact.density, u @ initial.density @ u.conj().T, atol=1e-12
    )
    direct_manifest = library.replay_manifest(
        LibrarySnapshot(()), direct.last_evaluation.content_identities
    )

    replay = Evolution(model, library=library, tables=library.restore_replay(manifest))
    actual = replay.evolve(initial, controls, end_ns=8)
    assert replay.last_evaluation.origin == "table"
    assert (
        replay.last_evaluation.content_identities
        == original.last_evaluation.content_identities
    )
    np.testing.assert_array_equal(actual.density, expected.density)
    exact_replay = Evolution(
        model, library=library, tables=library.restore_replay(direct_manifest)
    )
    np.testing.assert_array_equal(
        exact_replay.evolve(initial, controls, end_ns=8).density, exact.density
    )
    assert exact_replay.last_evaluation.origin == "direct-record"
