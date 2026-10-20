"""Create a fresh FTF demo entry and install offline resources in a measure GUI."""

from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.startup_context import StartupContext
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.resources.entry import load_entry_template

from zcu_lab.simulation_envs.FTF.environment import build_environment


def startup(ctx: StartupContext) -> None:
    """Create ftf-demo/initial from the adjacent template and open a frequency tab.

    ctx is the GUI's typed startup capability, called on its background worker
    after lab components/adapters are registered. Use the GUI's data project
    roots, not the shell cwd. Existing ftf-demo destinations fail rather than
    overwrite. Template, entry, environment and binding failures propagate to
    the startup runner, which retains completed work and registered cleanup.

    Build with one worker and no numeric warmup; this script never acquires or
    controls hardware. Successfully registered drivers become GUI-owned. The
    close callback releases borrowed references and closes only untransferred
    drivers, after background work drains and before GUI driver cleanup.
    """
    ctx.progress("Reading FTF entry template")
    template = load_entry_template(Path(__file__).with_name("entry-template.yaml"))
    ctx.setup.new_entry("ftf-demo")
    for name, component in template.components.items():
        fields = TypeAdapter(YamlMap).validate_python(
            component.model_dump(exclude_unset=True)
        )
        fields.pop("kind")
        ctx.context.add_component(name, component.kind, fields, setup=True)
    ctx.setup.new_point("initial")
    ctx.context.apply_alias_rows(
        tuple(AliasRow(name, value) for name, value in template.aliases.aliases.items())
    )

    ctx.progress("Building offline FTF resources")
    environment = build_environment(
        settings_path=Path(__file__).with_name("ftf.yaml"), workers=1, warmup=False
    )
    transferred: set[str] = set()

    def close_environment() -> None:
        # Runtime reads this manager; transferred drivers are borrowed until close.
        # Only the GUI closes them. Rejected drivers remain script-owned.
        for name in transferred:
            environment.devices.drop_device(name)
        environment.devices.close_all_devices()

    ctx.on_close(close_environment)
    ctx.soc.install(environment.soc, environment.soccfg)
    for name, driver in environment.devices.get_all_devices().items():
        ctx.devices.register(name, driver)
        transferred.add(name)
    ctx.progress("Opening FTF frequency tab")
    ctx.tabs.new_tab("onetone/freq")
