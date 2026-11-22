"""The shared GUI/MCP script seeds an offline entry without a window."""

from collections.abc import Iterator
from pathlib import Path
from shutil import copyfile

import pytest
from qtpy.QtCore import QCoreApplication
from zcu_tools.gui.app.measure import Registry, State
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.events.startup import StartupChangedPayload
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form.model import FormErrors
from zcu_tools.resources.entry import component_registry
from zcu_tools.simulators.ftf import environment

from tests.gui._qt_waits import qt_until
from zcu_lab.assembly import lab_definitions
from zcu_lab.simulation_envs import FTF


@pytest.fixture
def ftf_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Controller]:
    app = QCoreApplication.instance() or QCoreApplication([])
    assert app is not None
    source = Path(FTF.__file__).parent
    project = tmp_path / "project"
    project.mkdir()
    script_dir = tmp_path / "FTF"
    script_dir.mkdir()
    for asset in source.glob("entry-*.yaml"):
        copyfile(asset, script_dir / asset.name)
    script = script_dir / "startup.py"
    copyfile(source / script.name, script)
    copyfile(source / "default.yaml", script_dir / "ftf.yaml")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    profile = "qick_216_v2.yaml"
    copyfile(source.parent / "profiles" / profile, profiles / profile)

    build = environment.ftf

    def cpu_only(*args, **kwargs):
        # Keep the actual offline SoC and devices; model warmup is not this seam.
        return build(*args, **(kwargs | {"warmup": False, "workers": 1}))

    monkeypatch.setattr(environment, "ftf", cpu_only)
    lab = lab_definitions()
    old_kinds = set(component_registry.kinds())
    old_roles = set(component_registry.roles.names())
    old_shorthand = component_registry.roles.shorthand
    old_focus = component_registry.roles.focus_kinds
    old_policy = component_registry.schema_policy
    try:
        lab.register_components(component_registry)
        registry = Registry(templates=lab.templates)
        lab.register_experiments(registry)
        bus = BaseEventBus()
        ctrl = Controller(
            state=State(SessionEnv(parameters=None, soc=None, soccfg=None)),
            registry=registry,
            view=None,
            bus=bus,
            project_root=str(project),
        )
        try:
            events: list[StartupChangedPayload] = []
            subscription = bus.subscribe(StartupChangedPayload, events.append)
            try:
                ctrl.start_startup(script)
                qt_until(lambda: events and not events[-1].running, label="FTF startup")
                assert events[-1].error is None
            finally:
                subscription.unsubscribe()
            yield ctrl
        finally:
            ctrl.finish_close()
    finally:
        component_registry.configure_schema(old_policy)
        component_registry.roles.configure(
            shorthand=old_shorthand, focus_kinds=old_focus
        )
        for role in set(component_registry.roles.names()) - old_roles:
            component_registry.roles.unregister(role)
        for kind in set(component_registry.kinds()) - old_kinds:
            component_registry.unregister(kind)


def test_ftf_startup_creates_a_runnable_initial_demo(ftf_project: Controller) -> None:
    context = ftf_project.context_control
    assert context.current_point_label() == "initial"
    initial = context.read_context(point="initial")
    global_ = context.read_context()
    assert initial.value("$resonator") == "R1"
    assert initial.value("$qubit1") == "Q1"
    assert initial.value("$qubit2") == "Q2"
    assert initial.value("$qubit.wiring.lo_flux_ch") == 1
    assert initial.value("$qubit2.wiring.lo_flux_ch") == 4
    assert initial.value("$mux.flux_dev") == "coil"
    for alias in ("$qubit1", "$qubit2"):
        period = initial.value(f"{alias}.flux.period.coil")
        assert isinstance(period, float) and period != 0.0
    assert initial.value("$qubit.ext.flux_reset_gain") == pytest.approx(0.32)
    assert initial.value("$qubit.ext.flux_reset_length") == pytest.approx(20.0)
    linewidth = initial.value("$qubit.kappa")
    assert isinstance(linewidth, float) and linewidth > 0.0
    for capture in (initial, global_):
        assert capture.has_library_key("MG12.module.readout")
        assert capture.has_library_key("Q1.module.x180")
        assert capture.has_library_key("Q2.module.x180")

    devices = ftf_project.device_control
    assert devices.get_device_unit("coil") == "A"
    assert devices.get_cached_device_value("f1_bias") == pytest.approx(156.64e-6)
    assert devices.get_cached_device_value("f2_bias") == pytest.approx(83.24e-6)
    assert devices.get_cached_device_value("coupler_bias") == 0.0


def test_initial_demo_opens_fast_flux_and_mux_forms(ftf_project: Controller) -> None:
    failures = {}
    for adapter in (
        "fastflux/distortion/acc_phase",
        "fastflux/distortion/freq",
        "fastflux/distortion/phase",
        "fastflux/mist",
        "fastflux/t1",
        "fastflux/twotone",
        "twotone/reset/flux/gain",
        "twotone/reset/flux/length",
        "mux/onetone_flux_global",
        "mux/twotone_flux_global",
        "mux/singleshot_ge",
    ):
        tab = ftf_project.tab_control.new_tab(adapter)
        result = ftf_project.cfg_resources.lookup(tab).result()
        if isinstance(result, FormErrors):
            failures[adapter] = result
    assert not failures
