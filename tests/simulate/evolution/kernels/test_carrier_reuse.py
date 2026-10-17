"""Physical-block interpolation through its callback boundary, not eigengauges."""

import numpy as np
import pytest


def test_narrow_signed_scan_evaluates_original_carriers_within_probe_tolerance() -> (
    None
):
    from zcu_tools.simulate.evolution.kernels._carrier_reuse import CarrierWorkspace

    frequencies = -np.linspace(5.117, 5.141, 101)
    calls = []

    def physical(frequency):
        calls.append(frequency)
        return np.array([[[np.exp(3j / frequency), frequency**2]]], dtype=complex)

    workspace = CarrierWorkspace(frequencies)
    actual = np.array(
        [workspace.evaluate(("physical",), f, physical) for f in frequencies]
    )
    expected = np.array([[[[np.exp(3j / f), f**2]]] for f in frequencies])
    assert (
        np.max(np.sum(np.linalg.norm(actual - expected, axis=(-2, -1)), axis=-1))
        <= 1e-7
    )
    assert len(calls) < len(frequencies)
    assert workspace.nbytes > 0
    assert not np.array_equal(actual[0], actual[-1])
    # Returned arrays are owned by the caller, not reusable workspace storage.
    actual[0] = 0
    np.testing.assert_allclose(
        workspace.evaluate(("physical",), frequencies[0], physical),
        expected[0],
        atol=1e-7,
        rtol=0,
    )


def test_failed_probes_refine_or_use_exact_carriers() -> None:
    from zcu_tools.simulate.evolution.kernels._carrier_reuse import CarrierWorkspace

    frequencies = np.linspace(5.0, 5.04, 201)
    calls = []

    def physical(frequency):
        calls.append(frequency)
        return np.array([[[abs(frequency - 5.017321)]]], dtype=complex)

    workspace = CarrierWorkspace(frequencies)
    actual = [workspace.evaluate(("kink",), f, physical).item() for f in frequencies]
    np.testing.assert_allclose(
        actual, np.abs(frequencies - 5.017321), atol=1e-7, rtol=0
    )
    # The initial nine builds cannot represent this kink to the probe threshold.
    assert len(calls) > 9


@pytest.mark.parametrize(
    "frequencies", [np.linspace(5.0, 5.04, 9), np.linspace(0.05, 7.0, 101)]
)
def test_unprofitable_scan_uses_exact_builds_without_workspace_arrays(
    frequencies,
) -> None:
    from zcu_tools.simulate.evolution.kernels._carrier_reuse import CarrierWorkspace

    calls = []

    def physical(frequency):
        calls.append(frequency)
        return np.array([[[np.exp(13j / frequency)]]], dtype=complex)

    workspace = CarrierWorkspace(frequencies)
    for frequency in frequencies:
        np.testing.assert_array_equal(
            workspace.evaluate(("exact",), frequency, physical),
            np.array([[[np.exp(13j / frequency)]]]),
        )
    np.testing.assert_array_equal(calls, frequencies)
    assert workspace.nbytes == 0


def test_families_do_not_mix_physical_build_inputs() -> None:
    from zcu_tools.simulate.evolution.kernels._carrier_reuse import CarrierWorkspace

    frequencies = np.linspace(5.0, 5.04, 21)
    workspace = CarrierWorkspace(frequencies)
    for identity, scale in [(("coarse",), 1), (("fine",), 2), (("polynomial",), 3)]:
        actual = workspace.evaluate(
            identity,
            frequencies[4],
            lambda f, scale=scale: np.full((2, 2), scale * f, dtype=complex),
        )
        np.testing.assert_allclose(actual, scale * frequencies[4], atol=1e-7, rtol=0)


def test_distinct_physical_groups_do_not_pay_for_unshared_carrier_tiles() -> None:
    from zcu_tools.simulate.evolution.kernels._carrier_reuse import CarrierWorkspace

    frequencies = np.linspace(5.0, 5.04, 21)
    groups = [("bound", index) for index in range(len(frequencies))]
    workspace = CarrierWorkspace(frequencies, groups=groups)
    calls = []

    def physical(frequency):
        calls.append(frequency)
        return np.full((2, 2), frequency, dtype=complex)

    for group, frequency in zip(groups, frequencies, strict=True):
        np.testing.assert_array_equal(
            workspace.evaluate(("peak", group), frequency, physical, group=group),
            np.full((2, 2), frequency, dtype=complex),
        )
    np.testing.assert_array_equal(calls, frequencies)
    assert workspace.nbytes == 0
