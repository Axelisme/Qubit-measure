"""Assemble the single-fluxonium demo entry and offline measure-GUI resources."""

import logging
from pathlib import Path
from shlex import quote

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
    values come only from adjacent single_fluxonium.yaml. If it is missing,
    raise FileNotFoundError with a template copy command before creating an entry.
    No hardware, acquire or numeric warmup runs during startup. The mock SoC runs
    complete-timeline evolution and single-cavity acquisition on demand.

    Successfully registered drivers become GUI-owned. The on-close callback
    closes the backend after acquire drain, then drops borrowed driver references
    before closing any untransferred devices. Backend failure does not skip device
    cleanup; simultaneous cleanup failures are preserved in an ExceptionGroup.
    Partial entries and startup errors propagate; no fallback environment runs.
    """
    settings_path = Path(__file__).resolve().with_name("single_fluxonium.yaml")
    if not settings_path.exists():
        template_path = settings_path.with_name("default.yaml")
        raise FileNotFoundError(
            f"Missing simulation working copy: {settings_path}. Create it with: "
            f"cp {quote(str(template_path))} {quote(str(settings_path))}"
        )

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

    ctx.progress("Building offline single-fluxonium resources")
    environment = single_fluxonium(settings_path)
    transferred: set[str] = set()

    def close_environment() -> None:
        errors: list[Exception] = []
        if environment.close_backend is not None:
            try:
                environment.close_backend()
            except Exception as error:
                logging.exception("Single-fluxonium backend cleanup failed")
                errors.append(error)
        try:
            for name in transferred:
                environment.devices.drop_device(name)
            environment.devices.close_all_devices()
        except Exception as error:
            logging.exception("Single-fluxonium device cleanup failed")
            errors.append(error)
        if errors:
            raise ExceptionGroup("single-fluxonium resource cleanup failed", errors)

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
