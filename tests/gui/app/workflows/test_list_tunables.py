"""Native list Apply uses the registered converter, not preview leaf paths."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.workflows import Actor, Done, Step, WorkflowEnv, workflow
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.app.workflows.catalog import WorkflowCatalog
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.resources.entry import ResultEntry

from tests.experiment.workflows._engine_fakes import Plan, Rig, State
from tests.gui._user_input import ComponentDriver

from ._controller_support import identity
from ._list_tunables import ListForm, NativeTunables
from ._reload_source import InjectedCatalog
from ._start_fakes import StartHarness
from ._synthetic import Context, PlanForm, context_seed, init_sample
from ._synthetic import Plan as NativePlan


@pytest.fixture
def list_host(tmp_path: Path) -> tuple[FormState[ListForm], Rig]:
    rig = Rig(tmp_path)

    @workflow("lists", plan=Plan, tunables=NativeTunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: NativeTunables, state: State
    ) -> Step[State, int]:
        del env, plan, tun, state
        yield from ()
        return Done()

    env = FormEnv()
    rig.engine.start(
        step,
        plan=Plan(),
        tunables=ListForm().to_cfg(),
        paths=rig.paths,
        identity=identity(),
    )
    return FormState(ListForm.default_input(env)), rig


def binding(
    source: tuple[FormState[ListForm], Rig],
) -> tuple[TunablesHost[ListForm, int], Rig]:
    state, rig = source
    host: TunablesHost[ListForm, int] = TunablesHost(
        state, FormEnv(), "run", to_cfg=ListForm.to_cfg
    )
    host.accept(rig.engine)
    return host, rig


def test_indexed_draft_is_converted_to_complete_native_model(
    list_host: tuple[FormState[ListForm], Rig],
) -> None:
    host, rig = binding(list_host)
    host.state.write({("tones", 1, "gain"): {"expr": "3 * 2"}}, host.env)
    applied = host.apply(actor=Actor("agent", "rpc"))
    assert not isinstance(applied, FormErrors)
    assert applied.values == {
        "channels": [{"name": "A", "amplitude": 1.0}, {"name": "B", "amplitude": 6.0}],
        "participants": ["B"],
    }
    assert applied == rig.engine.tunables("run")
    assert host.state.base.same_as(host.accepted_input)
    assert host.accepted_input.data["tones"] == [
        {"label": "A", "gain": 1.0},
        {"label": "B", "gain": {"expr": "3 * 2"}},
    ]
    host.state.write({("tones", 0, "gain"): 4}, host.env)
    host.discard()
    assert host.state.input.same_as(host.accepted_input)


def test_native_conversion_error_preserves_correctable_draft_and_base(
    list_host: tuple[FormState[ListForm], Rig],
) -> None:
    from pydantic import ValidationError

    host, rig = binding(list_host)
    before = host.state.base
    host.state.write({("tones", 1, "gain"): 11}, host.env)
    with pytest.raises(ValidationError):
        host.apply(actor=Actor("user", "gui"))
    assert host.state.base.same_as(before) and host.accepted_input.same_as(before)
    assert not host.state.input.same_as(before)
    assert rig.engine.tunables("run").revision == 0
    host.state.write({("tones", 1, "gain"): 8}, host.env)
    applied = host.apply(actor=Actor("user", "gui"))
    assert not isinstance(applied, FormErrors) and applied.revision == 1


@pytest.fixture
def list_driver(
    attachment: tuple[WorkflowWindow[Context], StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    qapp: QApplication,
) -> Iterator[ComponentDriver]:
    window, harness, _entry = attachment

    @workflow("list-tuning", plan=NativePlan, tunables=NativeTunables, init=init_sample)
    def step(
        env: WorkflowEnv[Context], plan: NativePlan, tun: NativeTunables, state: int
    ) -> Step[int, float]:
        del plan, tun, state
        window.pause("list-tuning")
        yield from env.wait_until(env.started_at)
        return Done()

    catalog_source.catalog = WorkflowCatalog(
        (
            WorkflowAttachment(
                step=step,
                plan_model=PlanForm,
                knobs_model=ListForm,
                plan_to_cfg=PlanForm.to_cfg,
                knobs_to_cfg=ListForm.to_cfg,
                seed=context_seed,
            ),
        )
    )
    window.reload_source()
    window.start("list-tuning")
    harness.complete_segment()
    window.resize(1100, 900)
    window.show()
    qapp.processEvents()
    panel = window.tuning_panel
    assert panel is not None
    yield ComponentDriver(panel)


def test_visible_list_apply_discard_and_native_error_correction(
    list_driver: ComponentDriver, tmp_path: Path
) -> None:
    driver = list_driver
    driver.screenshot(tmp_path / "01-list-ready.png")
    # The picker identifies the original second element, not participant order.
    driver.expand_section("form.fold.tones")
    # The picked element shows its fields directly, without a sub-section fold.
    driver.select_combo("1: B", object_name="form.picker.tones")
    driver.enter_text(object_name="form.input.tones.1.gain", value="6")
    driver.click(object_name="workflow.tuning.apply")
    driver.screenshot(tmp_path / "02-list-applied.png")
    assert (
        driver.observe(object_name="workflow.tuning.message").text
        == "Applied revision 1"
    )
    driver.enter_text(object_name="form.input.tones.1.gain", value="9")
    driver.click(object_name="workflow.tuning.discard")
    driver.screenshot(tmp_path / "03-list-discarded.png")
    assert float(driver.observe(object_name="form.input.tones.1.gain").text) == 6
    driver.enter_text(object_name="form.input.tones.1.gain", value="11")
    driver.click(object_name="workflow.tuning.apply")
    driver.screenshot(tmp_path / "04-list-native-error.png")
    assert (
        "less than or equal to 10"
        in driver.observe(object_name="workflow.tuning.message").text
    )
    assert float(driver.observe(object_name="form.input.tones.1.gain").text) == 11
    driver.enter_text(object_name="form.input.tones.1.gain", value="8")
    driver.click(object_name="workflow.tuning.apply")
    driver.screenshot(tmp_path / "05-list-corrected.png")
    assert (
        driver.observe(object_name="workflow.tuning.message").text
        == "Applied revision 2"
    )
