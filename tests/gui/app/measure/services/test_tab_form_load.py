"""Native Load backfills direct module/range input without recapturing tab sources."""

from collections.abc import Mapping

from zcu_tools.cfg_model import ConfigBase, Expr
from zcu_tools.gui.app.measure.adapter import CfgInputRejected, Form
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv
from zcu_tools.program.v2.modules import PulseCfg
from zcu_tools.program.v2.sweep import SweepCfg


class Modules(ConfigBase):
    """A native core declares a concrete pulse module."""

    probe: PulseCfg


class LoadedCfg(ConfigBase):
    """Native evidence contains materialized modules and a complete sweep."""

    modules: Modules
    sweep: SweepCfg
    amount: float = 1.0


def test_load_repairs_invalid_draft_with_native_modules_and_range_without_capture() -> (
    None
):
    captures = 0

    def capture(_aliases: Mapping[str, str] | None) -> TabFormSources:
        nonlocal captures
        captures += 1
        return TabFormSources(FormEnv())

    resource = TabFormResource(
        Form(LoadedCfg, {"amount": Expr("unfinished +")}),
        capture=capture,
        mutation_allowed=lambda: True,
    )
    loaded = LoadedCfg(
        modules=Modules(
            probe=PulseCfg.model_validate(
                {
                    "ch": 1,
                    "nqz": 1,
                    "freq": 123.0,
                    "gain": 0.25,
                    "waveform": {"style": "const", "length": 0.1},
                }
            )
        ),
        sweep=SweepCfg(start=10, stop=14, expts=3, step=2),
        amount=4.0,
    )
    assert hasattr(resource, "load_cfg"), "Missing resource-owned native Load backfill"
    outcome = resource.load_cfg(loaded)
    assert outcome is None
    assert resource.result() == loaded
    assert captures == 1
    assert resource.input.data["amount"] == 4.0
    loaded.modules.probe.gain = 0.75
    accepted = resource.accept().values
    assert accepted["amount"] == 4.0
    assert captures == 1


def test_load_rejects_double_lower_and_keeps_the_previous_input() -> None:
    from pydantic import BaseModel

    class Scalar(ConfigBase):
        amount: float = 2.0

    class Extras(BaseModel):
        multiplier: int = 2

    def lower(extras: Extras, cfg: Scalar) -> Scalar:
        return Scalar(amount=cfg.amount * extras.multiplier)

    resource = TabFormResource(
        Form(Scalar, {}, extras=Extras, lower=lower),
        capture=lambda _aliases: TabFormSources(FormEnv()),
        mutation_allowed=lambda: True,
    )
    assert hasattr(resource, "load_cfg"), "Missing resource-owned native Load backfill"
    before = resource.input.data
    outcome = resource.load_cfg(Scalar(amount=4.0))
    assert isinstance(outcome, CfgInputRejected)
    assert outcome.differences[0].path == ("amount",)
    assert (outcome.differences[0].loaded, outcome.differences[0].would_run) == (
        4.0,
        8.0,
    )
    assert resource.input.data == before
    resource.write({("extras", "multiplier"): 1})
    assert resource.load_cfg(Scalar(amount=4.0)) is None
    assert resource.result() == Scalar(amount=4.0)
