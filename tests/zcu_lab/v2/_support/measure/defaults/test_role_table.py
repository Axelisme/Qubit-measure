"""Canonical role defaults retain deferred links and fresh inline value trees.

The manually reviewed golden characterizes empty and populated fixed captures.
Focused cases verify L01-only wiring and missing optimized readout calibration
through the actual public default leaves, without static or other-band fallback.
"""

from __future__ import annotations

from typing import Any

import pytest
from zcu_tools.gui.app.measure.specs import make_pulse_spec
from zcu_tools.gui.cfg import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
)
from zcu_tools.gui.session.expression import ExpressionError, evaluate_scalar_expr
from zcu_tools.gui.session.value_lookup import ValueKey, ValueRegistry

from tests.zcu_lab.v2._support.measure._role_characterization import (
    POPULATED_QUBIT,
    POPULATED_RESONATOR,
)
from tests.zcu_lab.v2._support.measure._role_characterization import (
    load_golden as _load_golden,
)
from tests.zcu_lab.v2._support.measure._role_characterization import (
    make_context as _mk_ctx,
)
from tests.zcu_lab.v2._support.measure._role_characterization import (
    serialize as _serialize,
)
from zcu_lab.components import (
    ChannelSchema,
    QubitSchema,
    QubitWiringSchema,
    ResonatorSchema,
)
from zcu_lab.v2._support.measure.defaults.role_table import (
    ROLE_TABLE,
    Pulse,
    RoleDef,
    Source,
    role_blank,
    role_ref,
)


def _compute_via_table(role_id: str) -> dict[str, Any]:
    role = ROLE_TABLE[role_id]
    entry: dict[str, Any] = {}
    for fx_name, ctx in (
        ("empty", _mk_ctx()),
        ("pop", _mk_ctx(qubit=POPULATED_QUBIT, resonator=POPULATED_RESONATOR)),
    ):
        entry[f"blank/{fx_name}"] = _serialize(role_blank(role, ctx))
        if role.lib is not None:
            entry[f"ref/{fx_name}/opt=False"] = _serialize(role_ref(role, ctx))
            entry[f"ref/{fx_name}/opt=True"] = _serialize(
                role_ref(role, ctx, optional=True)
            )
    return entry


@pytest.mark.parametrize("role_id", sorted(ROLE_TABLE))
def test_role_table_reproduces_golden_payload(role_id: str) -> None:
    golden = _load_golden()
    assert _compute_via_table(role_id) == golden[role_id], (
        f"ROLE_TABLE[{role_id!r}] payload differs from the manually reviewed "
        "canonical golden"
    )


def test_role_table_declares_exactly_the_golden_roles() -> None:
    """The table owns every role that the golden characterizes."""

    assert set(ROLE_TABLE) == set(_load_golden())


def test_readout_dpm_freq_requires_optimized_calibration_at_resolution() -> None:
    """A resonator frequency cannot substitute for missing optimized calibration."""
    ctx = _mk_ctx(resonator=ResonatorSchema(kind="resonator", freq=6100.0))
    node = role_blank(ROLE_TABLE["readout_dpm"], ctx)
    readout = node.value
    assert isinstance(readout, CfgSectionValue)

    pulse_cfg = readout.fields["pulse_cfg"]
    ro_cfg = readout.fields["ro_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert isinstance(ro_cfg, CfgSectionValue)

    pulse_freq = pulse_cfg.fields["freq"]
    ro_freq = ro_cfg.fields["ro_freq"]
    for freq in (pulse_freq, ro_freq):
        assert isinstance(freq, EvalValue)
        assert freq == EvalValue("$qubit.ext.best_ro_freq")
        with pytest.raises(ExpressionError) as caught:
            evaluate_scalar_expr(freq.expr, ctx.parameters)
        assert str(caught.value) == (
            "Invalid parameter path '$qubit.ext.best_ro_freq': missing parameter"
        )


@pytest.mark.parametrize(
    ("wiring", "resolved"),
    [
        (
            QubitWiringSchema(
                L01=ChannelSchema(ch=3),
                L14=ChannelSchema(ch=4),
                L45=ChannelSchema(ch=5),
            ),
            3,
        ),
        (
            QubitWiringSchema(L14=ChannelSchema(ch=4), L45=ChannelSchema(ch=5)),
            None,
        ),
        (QubitWiringSchema(L45=ChannelSchema(ch=5)), None),
        (QubitWiringSchema(), None),
    ],
    ids=["md0-expected0", "md1-expected1", "md2-expected2", "md3-expected3"],
)
def test_qubit_pulse_channel_uses_only_l01_wiring(
    wiring: QubitWiringSchema, resolved: int | None
) -> None:
    ctx = _mk_ctx(qubit=QubitSchema(kind="qubit/transmon", wiring=wiring))
    node = role_blank(ROLE_TABLE["qub_probe"], ctx)
    channel = node.value.fields["ch"]

    assert isinstance(channel, EvalValue)
    assert channel == EvalValue("$qubit.wiring.L01.ch")
    if resolved is not None:
        assert evaluate_scalar_expr(channel.expr, ctx.parameters) == resolved
    else:
        with pytest.raises(ExpressionError) as caught:
            evaluate_scalar_expr(channel.expr, ctx.parameters)
        assert str(caught.value) == (
            "Invalid parameter path '$qubit.wiring.L01.ch': missing parameter"
        )


def test_role_source_seed_resolves_against_value_registry() -> None:
    registry = ValueRegistry()
    registry.register(
        ValueKey("predictor.qubit_freq", float),
        lambda: 4123.0,
        owner="test",
    )
    ctx = _mk_ctx(values=registry)
    role = RoleDef(
        make_pulse_spec,
        "<Custom:Pulse>",
        pulses=(Pulse(Source("predictor.qubit_freq", "float"), 0, 0.05, 0.1),),
    )

    node = role_blank(role, ctx)

    assert node.value.fields["freq"] == DirectValue(4123.0)
