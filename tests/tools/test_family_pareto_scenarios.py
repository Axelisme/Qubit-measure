"""Public fixed-scenario construction and reproducible manifest freezing."""

from tools.family_pareto_inputs import canonical_json, freeze_manifest
from tools.family_pareto_scenarios import make_real_inputs


def test_fixed_scenarios_freeze_byte_identically_from_fresh_construction() -> None:
    first_inputs = make_real_inputs()
    assert first_inputs, "fixed-scenario construction must yield usable numeric inputs"
    first = canonical_json(freeze_manifest(first_inputs)) + "\n"
    second = canonical_json(freeze_manifest(make_real_inputs())) + "\n"
    assert first.encode() == second.encode()
