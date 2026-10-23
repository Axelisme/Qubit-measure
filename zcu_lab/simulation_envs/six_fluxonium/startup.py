"""Assemble the six-fluxonium demo entry and offline measure-GUI resources."""

import logging
from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.experiment.six_fluxonium import six_fluxonium
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.startup_context import StartupContext
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.resources.entry import DestinationExistsError, load_entry_template


def startup(ctx: StartupContext) -> None:
    """Open six-fluxonium-demo/initial and install mock SoC and seven DC drivers.

    ctx is the GUI's typed startup capability on its background worker, after
    component/adapter registration. New entries use adjacent entry-template.yaml
    and seed point aliases once; existing entries reopen unchanged. Physical
    values come only from adjacent default.yaml. No hardware, acquire or numeric
    warmup runs during startup. The mock SoC evolves each qubit independently
    on demand. Q1_bias..Q6_bias each affect only their own qubit. The shared
    coil_bias affects all six. Both contributions use per-qubit offsets and
    periods in A. No fast-flux line is installed. Single-tone scans use pulse
    gen13/ADC6; mux uses gen0/ADC0..5.

    Successfully registered drivers become GUI-owned. The on-close callback
    closes the backend after acquire drain, then drops borrowed driver references
    before closing any untransferred devices. Backend failure does not skip device
    cleanup; simultaneous cleanup failures are preserved in an ExceptionGroup.
    Partial entries and startup errors propagate; no fallback environment runs.
    """
    try:
        ctx.setup.new_entry("six-fluxonium-demo")
    except DestinationExistsError as exists:
        ctx.progress("Reopening six-fluxonium demo entry")
        try:
            ctx.setup.open_entry("six-fluxonium-demo")
        except FileNotFoundError as missing:
            raise RuntimeError(
                f"six-fluxonium-demo is incomplete ({exists.filename} exists, "
                f"{missing.filename} is missing). Restore the missing Result entry "
                "files before retrying, or select a new project root."
            ) from missing
        ctx.setup.use_point("initial")
    else:
        _seed_entry(ctx)

    ctx.progress("Building offline six-fluxonium resources")
    environment = six_fluxonium(Path(__file__).with_name("default.yaml"))
    transferred: set[str] = set()

    def close_environment() -> None:
        errors: list[Exception] = []
        if environment.close_backend is not None:
            try:
                environment.close_backend()
            except Exception as error:
                logging.exception("Six-fluxonium backend cleanup failed")
                errors.append(error)
        try:
            for name in transferred:
                environment.devices.drop_device(name)
            environment.devices.close_all_devices()
        except Exception as error:
            logging.exception("Six-fluxonium device cleanup failed")
            errors.append(error)
        if errors:
            raise ExceptionGroup("six-fluxonium resource cleanup failed", errors)

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
