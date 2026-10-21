"""Assemble the single-fluxonium demo entry and offline measure-GUI resources."""

from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.experiment.single_fluxonium import single_fluxonium
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.startup_context import StartupContext
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.resources.entry import DestinationExistsError, load_entry_template


def startup(ctx: StartupContext) -> None:
    """Open single-fluxonium-demo/initial and install mock SoC and DC driver.

    ctx is the GUI's typed startup capability on its background worker, after
    component/adapter registration. New entries use adjacent entry-template.yaml
    and seed point aliases once; existing entries reopen unchanged. Physical
    values come only from adjacent default.yaml. No hardware, acquire or numeric
    warmup runs. Acquisition remains unavailable pending single-cavity readout
    integration; successful assembly does not certify MCP/measurement acceptance.

    Successfully registered drivers become GUI-owned. The on-close callback
    drops borrowed driver references before closing any untransferred devices.
    Partial entries and startup errors propagate; no fallback environment runs.
    """
    try:
        ctx.setup.new_entry("single-fluxonium-demo")
    except DestinationExistsError as exists:
        ctx.progress("Reopening single-fluxonium demo entry")
        try:
            ctx.setup.open_entry("single-fluxonium-demo")
        except FileNotFoundError as missing:
            raise RuntimeError(
                f"single-fluxonium-demo is partially present ({exists.filename} exists, "
                f"{missing.filename} is missing). Restore the missing half or remove "
                "both demo directories under result/ and Database/ before retrying."
            ) from missing
        ctx.setup.use_point("initial")
    else:
        _seed_entry(ctx)

    ctx.progress(
        "Building offline single-fluxonium resources; acquisition not connected"
    )
    environment = single_fluxonium(Path(__file__).with_name("default.yaml"))
    transferred: set[str] = set()

    def close_environment() -> None:
        for name in transferred:
            environment.devices.drop_device(name)
        environment.devices.close_all_devices()

    ctx.on_close(close_environment)
    ctx.soc.install(environment.soc, environment.soccfg)
    for name, driver in environment.devices.get_all_devices().items():
        ctx.devices.register(name, driver)
        transferred.add(name)


def _seed_entry(ctx: StartupContext) -> None:
    template = load_entry_template(Path(__file__).with_name("entry-template.yaml"))
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
