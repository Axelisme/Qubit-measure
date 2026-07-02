"""The data-driven ROLE_TABLE + builders reproduce the role-factory golden.

Proves ``role_blank`` / ``role_ref`` over ``ROLE_TABLE`` produce the golden value
trees for every role. For the 14 non-``readout_dpm`` roles the table is
byte-identical to the legacy per-role factories; ``readout_dpm`` is in the table
too, pinned to its normalized all-live golden (the one deliberate live-eval
change). The golden's two md cases (empty / fully populated) do not reach the
``readout_dpm`` freq fallback path, so a separate focused test below locks it.
"""

from __future__ import annotations

from typing import Any

import pytest
from zcu_tools.experiment.v2_gui.adapters.shared.defaults.role_table import (
    ROLE_TABLE,
    Pulse,
    RoleDef,
    Source,
    role_blank,
    role_ref,
)
from zcu_tools.gui.app.main.adapter import CfgSectionValue, DirectValue, EvalValue
from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec
from zcu_tools.gui.session.value_lookup import ValueKey, ValueRegistry

from .test_role_default_characterization import (
    _POPULATED_MD,
    _load_golden,
    _mk_ctx,
    _serialize,
)


def _compute_via_table(role_id: str) -> dict[str, Any]:
    role = ROLE_TABLE[role_id]
    entry: dict[str, Any] = {}
    for fx_name, md in (("empty", {}), ("pop", _POPULATED_MD)):
        entry[f"blank/{fx_name}"] = _serialize(role_blank(role, _mk_ctx(md)))
        if role.lib is not None:
            entry[f"ref/{fx_name}/opt=False"] = _serialize(role_ref(role, _mk_ctx(md)))
            entry[f"ref/{fx_name}/opt=True"] = _serialize(
                role_ref(role, _mk_ctx(md), optional=True)
            )
    return entry


@pytest.mark.parametrize("role_id", sorted(ROLE_TABLE))
def test_role_table_reproduces_golden_payload(role_id: str) -> None:
    golden = _load_golden()
    assert _compute_via_table(role_id) == golden[role_id], (
        f"ROLE_TABLE[{role_id!r}] payload differs from the role-factory golden — "
        "the data-driven builder is not behaviour-identical to the legacy factory"
    )


def test_role_table_declares_exactly_the_golden_roles() -> None:
    """The table owns every role that the golden characterizes."""

    assert set(ROLE_TABLE) == set(_load_golden())


def test_readout_dpm_freq_falls_back_to_live_r_f_when_best_ro_freq_absent() -> None:
    """Partial md (``r_f`` present, ``best_ro_freq`` absent): readout_dpm's pulse
    and ro freq fall back to a *live* ``EvalValue("r_f")`` — mirroring how the
    plain readout role seeds freq from ``r_f``, not a snapshot. The nested
    ``Md("best_ro_freq", Md("r_f", 6000.0))`` seed expresses "prefer best_ro_freq,
    else the same r_f the plain readout uses". This is the all-live normalization
    (the golden's empty/populated md cases never reach this fallback)."""
    node = role_blank(ROLE_TABLE["readout_dpm"], _mk_ctx({"r_f": 6100.0}))
    readout = node.value
    assert isinstance(readout, CfgSectionValue)

    pulse_cfg = readout.fields["pulse_cfg"]
    ro_cfg = readout.fields["ro_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert isinstance(ro_cfg, CfgSectionValue)

    pulse_freq = pulse_cfg.fields["freq"]
    ro_freq = ro_cfg.fields["ro_freq"]
    assert isinstance(pulse_freq, EvalValue)
    assert pulse_freq.expr == "r_f"
    assert isinstance(ro_freq, EvalValue)
    assert ro_freq.expr == "r_f"


def test_role_source_seed_resolves_against_value_registry() -> None:
    registry = ValueRegistry()
    registry.register(
        ValueKey("predictor.qubit_freq", float),
        lambda: 4123.0,
        owner="test",
    )
    ctx = _mk_ctx({})
    ctx.values = registry
    role = RoleDef(
        make_pulse_spec,
        "<Custom:Pulse>",
        pulses=(Pulse(Source("predictor.qubit_freq", "float"), 0, 0.05, 0.1),),
    )

    node = role_blank(role, ctx)

    assert node.value.fields["freq"] == DirectValue(4123.0)
