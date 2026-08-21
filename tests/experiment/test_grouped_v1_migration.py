from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import pytest
from zcu_tools.experiment.v2.jpa.jpa_auto_optimize import (
    JPA_AUTO_GROUPED_ROLES,
    JPAOptimizeResult,
    jpa_auto_result_to_grouped_payloads,
    load_jpa_auto_grouped_result,
)
from zcu_tools.experiment.v2.twotone.ro_optimize.auto_optimize import (
    RO_AUTO_GROUPED_ROLES,
    AutoOptResult,
    auto_opt_result_to_grouped_payloads,
    load_auto_opt_grouped_result,
)
from zcu_tools.experiment.v2.twotone.time_domain.cpmg import (
    CPMG_GROUPED_ROLES,
    CPMG_Result,
    cpmg_result_to_grouped_payloads,
    load_cpmg_grouped_result,
)
from zcu_tools.utils.datasaver import (
    DatasetRole,
    GroupedLabberData,
    LabberMetadata,
    LabberPayload,
    load_grouped_labber_data,
)

from script.migrate_experiment_data import migrate_experiment_data
from tests.experiment.grouped_v1_fixture import write_grouped_v1


def _with_timestamps(
    roles: Mapping[str, LabberPayload], timestamps: np.ndarray
) -> dict[str, LabberPayload]:
    return {
        role: LabberPayload(
            payload.data,
            axes=payload.axes,
            timestamps=timestamps,
        )
        for role, payload in roles.items()
    }


def _assert_grouped_payloads_equal(
    actual: GroupedLabberData,
    expected_roles: Mapping[str, LabberPayload],
    expected_metadata: LabberMetadata,
) -> None:
    assert list(actual.roles) == [DatasetRole(role) for role in expected_roles]
    assert actual.metadata == expected_metadata
    for role, expected in expected_roles.items():
        payload = actual.roles[DatasetRole(role)]
        assert payload.data.name == expected.data.name
        assert payload.data.unit == expected.data.unit
        np.testing.assert_array_equal(payload.data.values, expected.data.values)
        assert [(axis.name, axis.unit) for axis in payload.axes] == [
            (axis.name, axis.unit) for axis in expected.axes
        ]
        for axis, expected_axis in zip(payload.axes, expected.axes, strict=True):
            np.testing.assert_array_equal(axis.values, expected_axis.values)
        np.testing.assert_array_equal(payload.timestamps, expected.timestamps)


def test_migrate_grouped_v1_cpmg_to_grouped_v2(tmp_path: Path) -> None:
    result = CPMG_Result(
        ns=np.array([1, 3], dtype=np.int64),
        delays=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        signals=np.array(
            [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]],
            dtype=np.complex128,
        ),
    )
    roles = _with_timestamps(
        cpmg_result_to_grouped_payloads(result),
        np.array([1239.75, 1240.75], dtype=np.float64),
    )
    metadata = LabberMetadata(
        comment="cpmg cfg snapshot",
        tags=["twotone/ge/cpmg"],
        project="project-a",
        user="user-a",
        creation_time=1234.5,
    )
    source = write_grouped_v1(tmp_path / "cpmg_v1.hdf5", roles, metadata=metadata)
    source_bytes = source.read_bytes()

    output = migrate_experiment_data(
        experiment="grouped/v1",
        input_path=source,
        output_path=tmp_path / "cpmg_v2.hdf5",
    )

    assert source.read_bytes() == source_bytes
    loaded = load_cpmg_grouped_result(str(output))
    np.testing.assert_array_equal(loaded.ns, result.ns)
    np.testing.assert_allclose(loaded.delays, result.delays, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(loaded.signals, result.signals)

    grouped = load_grouped_labber_data(str(output), required_roles=CPMG_GROUPED_ROLES)
    _assert_grouped_payloads_equal(grouped, roles, metadata)


def test_migrate_grouped_v1_ro_auto_optimize_to_grouped_v2(tmp_path: Path) -> None:
    result = AutoOptResult(
        params=np.array([[6000.0, 0.1, 1.0], [6100.0, 0.2, 1.5]], dtype=np.float64),
        signals=np.array([2.5, 3.5], dtype=np.float64),
    )
    roles = _with_timestamps(
        auto_opt_result_to_grouped_payloads(result),
        np.array([2239.75], dtype=np.float64),
    )
    metadata = LabberMetadata(
        comment="ro cfg",
        tags=["twotone/ge/ro_optimize/auto"],
        project="project-b",
        user="user-b",
        creation_time=2234.5,
    )
    source = write_grouped_v1(tmp_path / "ro_v1.hdf5", roles, metadata=metadata)

    output = migrate_experiment_data(
        experiment="grouped/v1",
        input_path=source,
        output_path=tmp_path / "ro_v2.hdf5",
    )

    loaded = load_auto_opt_grouped_result(str(output))
    np.testing.assert_allclose(loaded.params, result.params, rtol=0, atol=1e-15)
    assert loaded.params.dtype == np.float64
    np.testing.assert_array_equal(loaded.signals, result.signals)
    assert loaded.signals.dtype == np.float64
    grouped = load_grouped_labber_data(
        str(output), required_roles=RO_AUTO_GROUPED_ROLES
    )
    _assert_grouped_payloads_equal(grouped, roles, metadata)


def test_migrate_grouped_v1_jpa_auto_optimize_to_grouped_v2(tmp_path: Path) -> None:
    result = JPAOptimizeResult(
        params=np.array(
            [[-0.01, 7000.0, -10.0], [-0.02, 7100.0, -9.5]], dtype=np.float64
        ),
        phases=np.array([1, 3], dtype=np.int32),
        signals=np.array([4.5, 5.5], dtype=np.float64),
    )
    roles = _with_timestamps(
        jpa_auto_result_to_grouped_payloads(result),
        np.array([3239.75], dtype=np.float64),
    )
    metadata = LabberMetadata(
        comment="jpa cfg",
        tags=["jpa/auto_optimize"],
        project="project-c",
        user="user-c",
        creation_time=3234.5,
    )
    source = write_grouped_v1(tmp_path / "jpa_v1.hdf5", roles, metadata=metadata)

    output = migrate_experiment_data(
        experiment="grouped/v1",
        input_path=source,
        output_path=tmp_path / "jpa_v2.hdf5",
    )

    loaded = load_jpa_auto_grouped_result(str(output))
    np.testing.assert_allclose(loaded.params, result.params, rtol=0, atol=1e-15)
    assert loaded.params.dtype == np.float64
    np.testing.assert_array_equal(loaded.phases, result.phases)
    assert loaded.phases.dtype == np.int32
    np.testing.assert_array_equal(loaded.signals, result.signals)
    assert loaded.signals.dtype == np.float64
    grouped = load_grouped_labber_data(
        str(output), required_roles=JPA_AUTO_GROUPED_ROLES
    )
    _assert_grouped_payloads_equal(grouped, roles, metadata)
    assert grouped.roles[DatasetRole("jpa_flux")].data.unit == "a.u."


def test_grouped_v1_jpa_flux_a_uses_only_legacy_a_converter(tmp_path: Path) -> None:
    result = JPAOptimizeResult(
        params=np.array(
            [[-0.01, 7000.0, -10.0], [-0.02, 7100.0, -9.5]], dtype=np.float64
        ),
        phases=np.array([1, 3], dtype=np.int32),
        signals=np.array([4.5, 5.5], dtype=np.float64),
    )
    roles = jpa_auto_result_to_grouped_payloads(result)
    flux = roles["jpa_flux"]
    roles["jpa_flux"] = LabberPayload(
        (flux.data.name, "A", flux.data.values), axes=flux.axes
    )
    source = write_grouped_v1(tmp_path / "jpa_a_v1.hdf5", roles)
    source_bytes = source.read_bytes()

    grouped_v1_output = tmp_path / "wrong_route.hdf5"
    with pytest.raises(ValueError, match="unit"):
        migrate_experiment_data(
            experiment="grouped/v1",
            input_path=source,
            output_path=grouped_v1_output,
        )
    assert not grouped_v1_output.exists()

    output = migrate_experiment_data(
        experiment="jpa/jpa_auto_optimize/legacy_a",
        input_path=source,
        output_path=tmp_path / "jpa_a_v2.hdf5",
    )

    assert source.read_bytes() == source_bytes
    loaded = load_jpa_auto_grouped_result(str(output))
    np.testing.assert_allclose(loaded.params, result.params, rtol=0, atol=1e-15)
    grouped = load_grouped_labber_data(
        str(output), required_roles=JPA_AUTO_GROUPED_ROLES
    )
    assert grouped.roles[DatasetRole("jpa_flux")].data.unit == "a.u."


def test_grouped_v1_rejects_unsupported_role_set_before_output(tmp_path: Path) -> None:
    result = CPMG_Result(
        ns=np.array([1], dtype=np.int64),
        delays=np.array([[0.1, 0.2]], dtype=np.float64),
        signals=np.array([[1 + 1j, 2 + 2j]], dtype=np.complex128),
    )
    roles = cpmg_result_to_grouped_payloads(result)
    roles.pop("signals")
    source = write_grouped_v1(tmp_path / "incomplete_v1.hdf5", roles)
    output = tmp_path / "output.hdf5"

    with pytest.raises(ValueError, match="unsupported grouped v1 dataset role set"):
        migrate_experiment_data(
            experiment="grouped/v1", input_path=source, output_path=output
        )

    assert not output.exists()


def test_grouped_v1_rejects_streaming_artifact(tmp_path: Path) -> None:
    result = CPMG_Result(
        ns=np.array([1], dtype=np.int64),
        delays=np.array([[0.1, 0.2]], dtype=np.float64),
        signals=np.array([[1 + 1j, 2 + 2j]], dtype=np.complex128),
    )
    source = write_grouped_v1(
        tmp_path / "streaming_v1.hdf5", cpmg_result_to_grouped_payloads(result)
    )
    with h5py.File(source, "a") as file:
        file.attrs["zcu_tools.streaming_grouped_dataset_version"] = 1

    with pytest.raises(ValueError, match="streaming grouped dataset"):
        migrate_experiment_data(
            experiment="grouped/v1",
            input_path=source,
            output_path=tmp_path / "output.hdf5",
        )


def test_grouped_v1_typed_validation_removes_partial_output(tmp_path: Path) -> None:
    result = CPMG_Result(
        ns=np.array([1], dtype=np.int64),
        delays=np.array([[0.1, 0.2]], dtype=np.float64),
        signals=np.array([[1 + 1j, 2 + 2j]], dtype=np.complex128),
    )
    roles = cpmg_result_to_grouped_payloads(result)
    lengths = roles["lengths"]
    roles["lengths"] = LabberPayload(
        (lengths.data.name, "ms", lengths.data.values), axes=lengths.axes
    )
    source = write_grouped_v1(tmp_path / "bad_schema_v1.hdf5", roles)
    output = tmp_path / "output.hdf5"

    with pytest.raises(ValueError, match="unit"):
        migrate_experiment_data(
            experiment="grouped/v1", input_path=source, output_path=output
        )

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.*.hdf5")) == []
